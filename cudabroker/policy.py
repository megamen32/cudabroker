from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from .config import BrokerConfig
from .store import BrokerStore, Lease

AcquireStatus = Literal["granted", "queued"]
MAX_QUEUE_SKIPS = 3


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
        # Evicting leases still reserve VRAM until release/ack/dead timeout.
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

    def _evictable(self, lease: Lease) -> bool:
        # Never evict while loading or inside an active inference.
        return lease.status == "active" and lease.state == "loaded_idle"

    def _mark_one_idle(self, leases: list[Lease]) -> list[str]:
        candidates = [l for l in leases if self._evictable(l)]
        if not candidates:
            return []
        # Equal service class: reclaim least-recently-used idle residency first.
        lease = min(candidates, key=lambda l: (l.last_active, l.granted_at or l.created_at))
        self.store.mark_evicting(lease.lease_id)
        return [lease.lease_id]

    def _evict_idle_by_ttl(self) -> list[str]:
        now = time.time()
        expired = []
        for lease in self.store.active_leases():
            if not self._evictable(lease):
                continue
            model = self.store.models.get(lease.model_id)
            ttl = model.ttl_seconds if model else self.config.default_ttl_seconds
            if now - lease.last_active > ttl:
                expired.append(lease)
        return self._mark_one_idle(expired)

    def _evict_for(self, model_id: str) -> list[str]:
        # Eviction is asynchronous: VRAM remains reserved until the worker ACKs.
        # Mark one model only; the next ACK/maintenance pass recalculates fit.
        if self._fits(model_id):
            return []
        return self._mark_one_idle(self.store.active_leases())

    def acquire(self, model_id: str, client_id: str, worker_id: str | None = None, request_id: str | None = None) -> AcquireResult:
        with self.store.lock:
            existing = self.store.find_request_lease(model_id, client_id, worker_id, request_id)
            if existing is not None:
                # Repeated acquire is also a heartbeat for a waiter. Do not let
                # a healthy client lose its FIFO position to lease cleanup.
                existing.last_heartbeat = time.time()
                if existing.status == "active":
                    return AcquireResult("granted", existing, [])
                return AcquireResult("queued", existing, [])

            queued = self.store.queued_leases()
            if queued:
                lease = self.store.new_lease(model_id, client_id, "queued", worker_id, request_id)
                self.try_promote_queue()
                return AcquireResult(lease.status if lease.status == "active" else "queued", lease, [])

            ttl_evicted = self._evict_idle_by_ttl()
            if self.config.strict:
                if not self.store.reserving_leases():
                    return AcquireResult("granted", self.store.new_lease(model_id, client_id, "active", worker_id, request_id), [])
                return AcquireResult("queued", self.store.new_lease(model_id, client_id, "queued", worker_id, request_id), [])

            if self.config.fits_concurrent and self._fits(model_id):
                return AcquireResult("granted", self.store.new_lease(model_id, client_id, "active", worker_id, request_id), [])

            evicted = ttl_evicted or self._evict_for(model_id)
            return AcquireResult("queued", self.store.new_lease(model_id, client_id, "queued", worker_id, request_id), evicted)

    def try_promote_queue(self) -> None:
        with self.store.lock:
            ttl_evicted = self._evict_idle_by_ttl()
            queued = sorted(self.store.queued_leases(), key=lambda l: (l.created_at, l.lease_id))
            if not queued:
                return
            if self.config.strict and self.store.reserving_leases():
                return

            selected = None
            for index, lease in enumerate(queued):
                if self.config.strict or self._fits(lease.model_id):
                    selected = (index, lease)
                    break
            if selected is None:
                # Free one idle model per ACK cycle for the oldest waiter.
                if not ttl_evicted:
                    self._evict_for(queued[0].model_id)
                return

            index, lease = selected
            oldest = queued[0]
            if index > 0:
                oldest.queue_skips += 1
                total = self.store.gpu.total_mb
                need = self.store.footprint(oldest.model_id, self.config.stat_mode) + self.config.headroom_mb
                # After a few work-conserving skips, let active work drain and
                # reserve the next admission for the oldest feasible waiter.
                if oldest.queue_skips >= MAX_QUEUE_SKIPS and need <= total:
                    self._evict_for(oldest.model_id)
                    return
            lease.status = "active"
            lease.state = "loading"
            lease.granted_at = time.time()
            lease.last_heartbeat = time.time()
            lease.last_active = time.time()
