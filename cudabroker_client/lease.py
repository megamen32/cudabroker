from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from .config import ClientConfig
from .transport import BrokerTransport


@dataclass
class LeaseHandle:
    lease_id: str
    transport: BrokerTransport
    local_mode: bool = False
    evicted: bool = False

    def touch(self, active: bool = True, vram_sample_mb: float | None = None, state: str | None = None) -> str:
        if self.local_mode:
            return "active"
        resp = self.transport.post("/v1/heartbeat", {
            "lease_id": self.lease_id,
            "active": active,
            "state": state,
            "vram_sample_mb": vram_sample_mb,
        })
        if resp.local_mode:
            self.local_mode = True
            return "active"
        status = resp.data.get("status", "unknown")
        if status == "evicted":
            self.evicted = True
        return status

    def release(self) -> None:
        if not self.local_mode:
            self.transport.post("/v1/release", {"lease_id": self.lease_id})

    def evict_ack(self) -> None:
        if not self.local_mode:
            self.transport.post("/v1/evict_ack", {"lease_id": self.lease_id})


@contextmanager
def gpu_lease(model_id: str, vram_mb: float | None = None, gpu_priority: int = 0,
              cpu_capable: bool = False, ttl_seconds: float = 600,
              client_id: str | None = None, wait_seconds: float = 15,
              worker_id: str | None = None, request_id: str | None = None) -> Iterator[LeaseHandle]:
    cfg = ClientConfig()
    transport = BrokerTransport(cfg)
    cid = client_id or cfg.client_id
    wid = worker_id or cfg.worker_id
    rid = request_id or str(uuid.uuid4())
    payload = {
        "model_id": model_id,
        "client_id": cid,
        "worker_id": wid,
        "request_id": rid,
        "vram_mb": vram_mb,
        "gpu_priority": gpu_priority,
        "cpu_capable": cpu_capable,
        "ttl_seconds": ttl_seconds,
        "wait_seconds": wait_seconds,
    }
    resp = transport.post("/v1/acquire", payload)
    if resp.local_mode:
        if cfg.required:
            raise RuntimeError("cudabroker is required but unavailable")
        handle = LeaseHandle("local-" + str(uuid.uuid4()), transport, local_mode=True)
    else:
        while resp.data.get("status") != "granted":
            time.sleep(min(2.0, max(0.25, wait_seconds / 5)))
            resp = transport.post("/v1/acquire", payload)
            if resp.local_mode:
                if cfg.required:
                    raise RuntimeError("cudabroker is required but unavailable")
                handle = LeaseHandle("local-" + str(uuid.uuid4()), transport, local_mode=True)
                break
        else:
            handle = LeaseHandle(resp.data["lease_id"], transport, local_mode=False)
    try:
        yield handle
    finally:
        handle.release()
