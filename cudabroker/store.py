from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from statistics import mean
from threading import RLock
from typing import Deque, Literal

LeaseStatus = Literal["queued", "active", "evicting", "released", "dead"]
LeaseState = Literal[
    "queued",
    "loading",
    "loaded_idle",
    "inference_active",
    "unloading",
    "cpu",
    "released",
]

TERMINAL_STATUSES = {"released", "dead"}
RESERVED_STATUSES = {"active", "evicting"}


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
    worker_id: str | None = None
    request_id: str | None = None
    state: LeaseState = "queued"
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

    def ensure_model(
        self,
        model_id: str,
        client_id: str,
        declared_vram_mb: float | None = None,
        gpu_priority: int = 0,
        cpu_capable: bool = False,
        ttl_seconds: float = 600,
    ) -> ModelInfo:
        with self.lock:
            if model_id not in self.models:
                self.register(ModelInfo(model_id, client_id, declared_vram_mb, gpu_priority, cpu_capable, ttl_seconds))
            return self.models[model_id]

    def find_request_lease(
        self,
        model_id: str,
        client_id: str,
        worker_id: str | None,
        request_id: str | None,
    ) -> Lease | None:
        if not request_id:
            return None
        with self.lock:
            for lease in self.leases.values():
                if lease.status in TERMINAL_STATUSES:
                    continue
                if (
                    lease.model_id == model_id
                    and lease.client_id == client_id
                    and lease.worker_id == worker_id
                    and lease.request_id == request_id
                ):
                    return lease
        return None

    def new_lease(
        self,
        model_id: str,
        client_id: str,
        status: LeaseStatus,
        worker_id: str | None = None,
        request_id: str | None = None,
    ) -> Lease:
        now = time.time()
        lease = Lease(
            lease_id=str(uuid.uuid4()),
            model_id=model_id,
            client_id=client_id,
            worker_id=worker_id,
            request_id=request_id,
            status=status,
            state="loading" if status == "active" else "queued",
            granted_at=now if status == "active" else None,
            created_at=now,
            last_heartbeat=now,
            last_active=now,
        )
        self.leases[lease.lease_id] = lease
        return lease

    def active_leases(self) -> list[Lease]:
        return [l for l in self.leases.values() if l.status == "active"]

    def reserving_leases(self) -> list[Lease]:
        return [l for l in self.leases.values() if l.status in RESERVED_STATUSES]

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

    def mark_evicting(self, lease_id: str) -> None:
        with self.lock:
            lease = self.leases.get(lease_id)
            if lease and lease.status == "active":
                lease.status = "evicting"
                lease.state = "unloading"
                lease.last_heartbeat = time.time()

    def release(self, lease_id: str) -> bool:
        with self.lock:
            lease = self.leases.get(lease_id)
            if not lease:
                return False
            lease.status = "released"
            lease.state = "released"
            lease.last_heartbeat = time.time()
            return True

    def cleanup_dead(self, lease_timeout: float) -> None:
        now = time.time()
        with self.lock:
            for lease in self.leases.values():
                if lease.status in {"active", "queued", "evicting"} and now - lease.last_heartbeat > lease_timeout:
                    lease.status = "dead"
                    lease.state = "released"
