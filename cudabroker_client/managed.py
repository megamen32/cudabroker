from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Generic, TypeVar

from .config import ClientConfig
from .lease import LeaseHandle, gpu_lease
from .sampler import memory_allocated_mb

T = TypeVar("T")
log = logging.getLogger(__name__)


class ManagedModel(Generic[T]):
    def __init__(self, model_id: str, loader: Callable[[], T], vram_mb: float | None = None,
                 gpu_priority: int = 0, cpu_capable: bool = False, ttl_seconds: float = 600,
                 cpu_fallback: Callable[[], T] | None = None, client_id: str | None = None,
                 worker_id: str | None = None):
        self.model_id = model_id
        self.loader = loader
        self.vram_mb = vram_mb
        self.gpu_priority = gpu_priority
        self.cpu_capable = cpu_capable
        self.ttl_seconds = ttl_seconds
        self.cpu_fallback = cpu_fallback
        self.client_id = client_id
        self.worker_id = worker_id
        self.current_device = "unloaded"
        self.model: T | None = None
        self.lease: LeaseHandle | None = None
        self._cm = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._active_until = 0.0

    def acquire(self) -> T:
        with self._lock:
            if self.model is not None and self.current_device == "cuda":
                return self.model
            self._cm = gpu_lease(
                self.model_id,
                self.vram_mb,
                self.gpu_priority,
                self.cpu_capable,
                self.ttl_seconds,
                self.client_id,
                worker_id=self.worker_id,
            )
            self.lease = self._cm.__enter__()
            self.lease.touch(active=False, state="loading")
            before = memory_allocated_mb()
            self.model = self.loader()
            after = memory_allocated_mb()
            sample = (after - before) if before is not None and after is not None and after >= before else self.vram_mb
            self.current_device = "cuda"
            self.touch(active=False, vram_sample_mb=sample, state="loaded_idle")
            self._start_heartbeat()
            return self.model

    def _start_heartbeat(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._heartbeat_loop, name=f"cudabroker-heartbeat-{self.model_id}", daemon=True)
        self._thread.start()

    def _heartbeat_loop(self) -> None:
        try:
            cfg = ClientConfig()
            interval = max(1.0, float(cfg.heartbeat_interval))
        except Exception:
            interval = 5.0
        while not self._stop.wait(interval):
            try:
                now = time.time()
                active = now < self._active_until
                state = "inference_active" if active else "loaded_idle"
                status = self.touch(active=active, state=state)
                if status == "evicted":
                    self._evict_to_cpu()
                    return
            except Exception as e:
                log.warning("cudabroker heartbeat failed for %s: %s", self.model_id, e)

    def touch(self, active: bool = True, vram_sample_mb: float | None = None, state: str | None = None) -> str:
        with self._lock:
            if not self.lease:
                return "unloaded"
            if active:
                try:
                    self._active_until = max(self._active_until, time.time() + ClientConfig().active_grace_seconds)
                except Exception:
                    self._active_until = max(self._active_until, time.time() + 30.0)
                state = state or "inference_active"
            elif state is None:
                state = "loaded_idle"
            status = self.lease.touch(active=active, vram_sample_mb=vram_sample_mb, state=state)
            if status == "evicted":
                self._evict_to_cpu()
            return status

    def _evict_to_cpu(self) -> None:
        with self._lock:
            if self.current_device != "cuda":
                return
            self.model = None
            self.current_device = "evicted"
            if self.lease:
                self.lease.evict_ack()
            if self._cm:
                try:
                    self._cm.__exit__(None, None, None)
                except Exception:
                    pass
            self.lease = None
            self._cm = None
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
            except Exception:
                pass
            if self.cpu_fallback:
                self.model = self.cpu_fallback()
                self.current_device = "cpu"

    def release(self) -> None:
        with self._lock:
            self._stop.set()
            if self.lease:
                try:
                    self.lease.touch(active=False, state="unloading")
                except Exception:
                    pass
            self.model = None
            self.current_device = "unloaded"
            if self.lease:
                self.lease.release()
            if self._cm:
                self._cm.__exit__(None, None, None)
            self.lease = None
            self._cm = None
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
            except Exception:
                pass
