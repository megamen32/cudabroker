from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import get_config
from .gpu import GpuSampler
from .policy import Policy
from .store import BrokerStore, LeaseState, ModelInfo

log = logging.getLogger("cudabroker")
config = get_config()
store = BrokerStore(stats_limit=config.stats_limit)
policy = Policy(store, config)
sampler = GpuSampler(store, config.sample_interval)


class RegisterRequest(BaseModel):
    model_id: str
    client_id: str
    vram_mb: float | None = None
    gpu_priority: int = 0
    cpu_capable: bool = False
    ttl_seconds: float = Field(default_factory=lambda: config.default_ttl_seconds)


class AcquireRequest(RegisterRequest):
    worker_id: str | None = None
    request_id: str | None = None
    wait_seconds: float = 15


class AcquireResponse(BaseModel):
    lease_id: str
    status: Literal["granted", "queued"]
    evicted: list[str] = []


class HeartbeatRequest(BaseModel):
    lease_id: str
    active: bool = False
    state: LeaseState | None = None
    vram_sample_mb: float | None = None


class HeartbeatResponse(BaseModel):
    status: Literal["active", "queued", "evicted", "released", "dead", "unknown"]


class ReleaseRequest(BaseModel):
    lease_id: str


async def maint_loop() -> None:
    while True:
        store.cleanup_dead(config.lease_timeout)
        policy.try_promote_queue()
        await asyncio.sleep(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    sampler.start()
    task = asyncio.create_task(maint_loop())
    try:
        yield
    finally:
        task.cancel()
        sampler.stop()


app = FastAPI(title="cudabroker", version="0.1.0", lifespan=lifespan)


@app.post("/v1/register")
def register(req: RegisterRequest):
    model = store.register(ModelInfo(
        model_id=req.model_id,
        client_id=req.client_id,
        declared_vram_mb=req.vram_mb,
        gpu_priority=req.gpu_priority,
        cpu_capable=req.cpu_capable,
        ttl_seconds=req.ttl_seconds,
    ))
    return {"ok": True, "model": model.__dict__}


@app.post("/v1/acquire", response_model=AcquireResponse)
async def acquire(req: AcquireRequest):
    store.ensure_model(req.model_id, req.client_id, req.vram_mb, req.gpu_priority, req.cpu_capable, req.ttl_seconds)
    deadline = time.time() + max(0.0, req.wait_seconds)
    result = policy.acquire(req.model_id, req.client_id, req.worker_id, req.request_id)
    if result.status == "granted":
        return AcquireResponse(lease_id=result.lease.lease_id, status="granted", evicted=result.evicted)
    lease = result.lease
    while time.time() < deadline:
        await asyncio.sleep(0.25)
        policy.try_promote_queue()
        current = store.leases.get(lease.lease_id)
        if current and current.status == "active":
            return AcquireResponse(lease_id=lease.lease_id, status="granted", evicted=result.evicted)
    return AcquireResponse(lease_id=lease.lease_id, status="queued", evicted=result.evicted)


@app.post("/v1/heartbeat", response_model=HeartbeatResponse)
def heartbeat(req: HeartbeatRequest):
    with store.lock:
        lease = store.leases.get(req.lease_id)
        if not lease:
            return HeartbeatResponse(status="unknown")
        lease.last_heartbeat = time.time()
        if req.vram_sample_mb is not None:
            lease.last_vram_sample = req.vram_sample_mb
            store.add_sample(lease.model_id, req.vram_sample_mb)
        if lease.status == "evicting":
            return HeartbeatResponse(status="evicted")
        if lease.status in {"released", "dead"}:
            return HeartbeatResponse(status=lease.status)
        if lease.status == "queued":
            return HeartbeatResponse(status="queued")
        if req.state is not None:
            lease.state = req.state
        elif req.active:
            lease.state = "inference_active"
        else:
            lease.state = "loaded_idle"
        if req.active:
            lease.last_active = time.time()
        return HeartbeatResponse(status="active")


@app.post("/v1/release")
def release(req: ReleaseRequest):
    ok = store.release(req.lease_id)
    policy.try_promote_queue()
    return {"ok": ok}


@app.post("/v1/evict_ack")
def evict_ack(req: ReleaseRequest):
    ok = store.release(req.lease_id)
    policy.try_promote_queue()
    return {"ok": ok}


@app.get("/v1/status")
def status():
    with store.lock:
        return {
            "gpu": {"total_mb": store.gpu.total_mb, "free_mb": store.gpu.free_mb, "ts": store.gpu.ts},
            "active_leases": [{
                "lease_id": l.lease_id,
                "model_id": l.model_id,
                "client_id": l.client_id,
                "worker_id": l.worker_id,
                "request_id": l.request_id,
                "status": l.status,
                "state": l.state,
                "gpu_priority": store.models.get(l.model_id).gpu_priority if store.models.get(l.model_id) else None,
                "last_active": l.last_active,
                "footprint_mb": store.footprint(l.model_id, config.stat_mode),
            } for l in store.reserving_leases()],
            "queue": [{
                "lease_id": l.lease_id,
                "model_id": l.model_id,
                "client_id": l.client_id,
                "worker_id": l.worker_id,
                "request_id": l.request_id,
            } for l in store.queued_leases()],
            "models": {k: v.__dict__ for k, v in store.models.items()},
        }


@app.get("/v1/health")
def health():
    return {"ok": True}


def main() -> None:
    uvicorn.run("cudabroker.server:app", host=config.host, port=config.port, reload=False)


if __name__ == "__main__":
    main()
