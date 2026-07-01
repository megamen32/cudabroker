from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from statistics import mean
from threading import RLock
from typing import Deque, Literal

LeaseStatus = Literal["active", "queued", "evicted", "released"]


@dataclass
class ModelInfo:
    model_id: str
    client_id: str
    declared_vram_mb: float | None = None
    gpu_priority: int = 0
    cpu_capable: bool = False
    ttl_seconds: float = 600


@dataclass
class Lease:
    lease_id: str
    model_id: str
    client_id: str
    status: LeaseStatus
    granted_at: float | None = None
    created_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    last_vram_sample: float | None = None


@dataclass
class GpuSnapshot:
    total_mb: float = 0
    free_mb: float = 0
    ts: float = 0


class BrokerStore:
    def __init__(self, stats_limit: int = 200):
        self.lock = RLock()
        self.models: dict[str, ModelInfo] = {}
        self.leases: dict[str, Lease] = {}
        self.stats: dict[str, Deque[float]] = {}
        self.gpu = GpuSnapshot()
        self.stats_limit = stats_limit

    def register(self, model: ModelInfo) -> ModelInfo:
        with self.lock:
            prev = self.models.get(model.model_id)
            if prev and model.declared_vram_mb is None:
                model.declared_vram_mb = prev.declared_vram_mb
            self.models[model.model_id] = model
            self.stats.setdefault(model.model_id, deque(maxlen=self.stats_limit))
            return model

    def ensure_model(self, model_id: str, client_id: str, declared_vram_mb: float | None = None,
                     gpu_priority: int = 0, cpu_capable: bool = False, ttl_seconds: float = 600) -> ModelInfo:
        with self.lock:
            if model_id not in self.models:
                self.register(ModelInfo(model_id, client_id, declared_vram_mb, gpu_priority, cpu_capable, ttl_seconds))
            return self.models[model_id]

    def new_lease(self, model_id: str, client_id: str, status: LeaseStatus) -> Lease:
        now = time.time()
        lease = Lease(
            lease_id=str(uuid.uuid4()),
            model_id=model_id,
            client_id=client_id,
            status=status,
            granted_at=now if status == "active" else None,
            created_at=now,
            last_heartbeat=now,
            last_active=now,
        )
        self.leases[lease.lease_id] = lease
        return lease

    def active_leases(self) -> list[Lease]:
        return [l for l in self.leases.values() if l.status == "active"]

    def queued_leases(self) -> list[Lease]:
        return [l for l in self.leases.values() if l.status == "queued"]

    def set_gpu(self, total_mb: float, free_mb: float) -> None:
        with self.lock:
            self.gpu = GpuSnapshot(total_mb=total_mb, free_mb=free_mb, ts=time.time())

    def add_sample(self, model_id: str, mb: float | None) -> None:
        if mb is None or mb <= 0:
            return
        with self.lock:
            self.stats.setdefault(model_id, deque(maxlen=self.stats_limit)).append(float(mb))

    def footprint(self, model_id: str, mode: str = "p98") -> float:
        with self.lock:
            samples = list(self.stats.get(model_id, ()))
            model = self.models.get(model_id)
            declared = float(model.declared_vram_mb or 0) if model else 0
        if len(samples) < 3:
            return declared
        mode = (mode or "p98").lower()
        if mode == "mean":
            return max(declared, float(mean(samples)))
        if mode == "max":
            return max(declared, float(max(samples)))
        samples.sort()
        idx = min(len(samples) - 1, max(0, int(round(0.98 * (len(samples) - 1)))))
        return max(declared, float(samples[idx]))

    def mark_evicted(self, lease_id: str) -> None:
        with self.lock:
            if lease_id in self.leases and self.leases[lease_id].status == "active":
                self.leases[lease_id].status = "evicted"

    def release(self, lease_id: str) -> bool:
        with self.lock:
            lease = self.leases.get(lease_id)
            if not lease:
                return False
            lease.status = "released"
            return True

    def cleanup_dead(self, lease_timeout: float) -> None:
        now = time.time()
        with self.lock:
            for lease in self.leases.values():
                if lease.status in {"active", "queued", "evicted"} and now - lease.last_heartbeat > lease_timeout:
                    lease.status = "released"
