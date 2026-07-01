from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from .config import BrokerConfig
from .store import BrokerStore, Lease

AcquireStatus = Literal["granted", "queued"]


@dataclass
class AcquireResult:
    status: AcquireStatus
    lease: Lease
    evicted: list[str]


class Policy:
    def __init__(self, store: BrokerStore, config: BrokerConfig):
        self.store = store
        self.config = config

    def _reserved_mb(self) -> float:
        # Critical: evicting leases still reserve VRAM until release/ack/dead timeout.
        return sum(self.store.footprint(l.model_id, self.config.stat_mode) for l in self.store.reserving_leases())

    def _available_mb(self) -> float:
        gpu = self.store.gpu
        if gpu.total_mb <= 0:
            return 0.0
        live_room = gpu.free_mb
        reserved_room = gpu.total_mb - self._reserved_mb()
        return max(0.0, min(live_room, reserved_room))

    def _fits(self, model_id: str) -> bool:
        f = self.store.footprint(model_id, self.config.stat_mode)
        return self._available_mb() >= f + self.config.headroom_mb

    @staticmethod
    def _evictable(lease: Lease) -> bool:
        # Do not evict while loading or inside an active inference. Only idle loaded models.
        return lease.status == "active" and lease.state == "loaded_idle"

    def _evict_idle_by_ttl(self) -> list[str]:
        now = time.time()
        evicted: list[str] = []
        for lease in self.store.active_leases():
            if not self._evictable(lease):
                continue
            model = self.store.models.get(lease.model_id)
            ttl = model.ttl_seconds if model else self.config.default_ttl_seconds
            if now - lease.last_active > ttl:
                self.store.mark_evicting(lease.lease_id)
                evicted.append(lease.lease_id)
        return evicted

    def _evict_for(self, model_id: str) -> list[str]:
        evicted = self._evict_idle_by_ttl()
        if self._fits(model_id):
            return evicted
        candidates = []
        now = time.time()
        for lease in self.store.active_leases():
            model = self.store.models.get(lease.model_id)
            if not model or not model.cpu_capable or not self._evictable(lease):
                continue
            idle = now - lease.last_active > model.ttl_seconds
            candidates.append((0 if idle else 1, model.gpu_priority, lease.granted_at or lease.created_at, lease))
        for _, _, _, lease in sorted(candidates, key=lambda x: x[:3]):
            self.store.mark_evicting(lease.lease_id)
            evicted.append(lease.lease_id)
            # _fits still includes evicting leases; this only becomes true if enough room existed anyway.
            if self._fits(model_id):
                break
        return evicted

    def acquire(self, model_id: str, client_id: str, worker_id: str | None = None, request_id: str | None = None) -> AcquireResult:
        with self.store.lock:
            existing = self.store.find_request_lease(model_id, client_id, worker_id, request_id)
            if existing is not None:
                if existing.status == "active":
                    return AcquireResult("granted", existing, [])
                return AcquireResult("queued", existing, [])

            self._evict_idle_by_ttl()
            if self.config.strict:
                if not self.store.reserving_leases():
                    return AcquireResult("granted", self.store.new_lease(model_id, client_id, "active", worker_id, request_id), [])
                return AcquireResult("queued", self.store.new_lease(model_id, client_id, "queued", worker_id, request_id), [])

            if self.config.fits_concurrent and self._fits(model_id):
                return AcquireResult("granted", self.store.new_lease(model_id, client_id, "active", worker_id, request_id), [])

            evicted = self._evict_for(model_id)
            if self.config.fits_concurrent and self._fits(model_id):
                return AcquireResult("granted", self.store.new_lease(model_id, client_id, "active", worker_id, request_id), evicted)
            return AcquireResult("queued", self.store.new_lease(model_id, client_id, "queued", worker_id, request_id), evicted)

    def try_promote_queue(self) -> None:
        with self.store.lock:
            self._evict_idle_by_ttl()
            for lease in sorted(self.store.queued_leases(), key=lambda l: l.created_at):
                if self.config.strict and self.store.reserving_leases():
                    return
                if (not self.config.strict) and (not self._fits(lease.model_id)):
                    return
                lease.status = "active"
                lease.state = "loading"
                lease.granted_at = time.time()
                lease.last_heartbeat = time.time()
                lease.last_active = time.time()
