from __future__ import annotations

import asyncio
import gc
import inspect
import logging
import time
from contextlib import asynccontextmanager
from typing import Awaitable, Callable, Generic, TypeVar

from .config import ClientConfig
from .lease import AsyncLeaseHandle, gpu_lease_async
from .sampler import memory_allocated_mb

T = TypeVar("T")
MaybeAwaitable = T | Awaitable[T]
log = logging.getLogger(__name__)
LOST_LEASE_STATUSES = {"evicted", "unknown", "dead", "released"}


async def _maybe_await(value: MaybeAwaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value  # type: ignore[no-any-return]
    return value


class AsyncManagedModel(Generic[T]):
    """Async mirror of ManagedModel for asyncio/FastAPI services."""

    def __init__(self, model_id: str, loader: Callable[[], MaybeAwaitable[T]], vram_mb: float | None = None,
                 gpu_priority: int = 0, cpu_capable: bool = False, ttl_seconds: float = 600,
                 cpu_fallback: Callable[[], MaybeAwaitable[T]] | None = None, client_id: str | None = None,
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
        self.lease: AsyncLeaseHandle | None = None
        self._cm = None
        self._stop: asyncio.Event | None = None
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._active_until = 0.0
        self._active_count = 0

    async def __aenter__(self) -> T:
        return await self.acquire()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.release()

    async def acquire(self) -> T:
        async with self._lock:
            if self.model is not None and self.current_device == "cuda":
                return self.model
            self._cm = gpu_lease_async(
                self.model_id,
                self.vram_mb,
                self.gpu_priority,
                self.cpu_capable,
                self.ttl_seconds,
                self.client_id,
                worker_id=self.worker_id,
            )
            self.lease = await self._cm.__aenter__()
            if self.lease.local_mode:
                cm = self._cm
                self.lease = None
                self._cm = None
                await cm.__aexit__(None, None, None)
                if self.cpu_fallback is None:
                    raise RuntimeError(
                        f"cudabroker unavailable; unmanaged GPU load refused for {self.model_id}"
                    )
                self.model = await _maybe_await(self.cpu_fallback())
                self.current_device = "cpu"
                return self.model
            await self.lease.touch(active=False, state="loading")
            before = memory_allocated_mb()
            self.model = await _maybe_await(self.loader())
            after = memory_allocated_mb()
            sample = (after - before) if before is not None and after is not None and after >= before else self.vram_mb
            self.current_device = "cuda"
            await self._touch_locked(active=False, vram_sample_mb=sample, state="loaded_idle")
            self._start_heartbeat()
            return self.model

    def _start_heartbeat(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._heartbeat_loop(), name=f"cudabroker-heartbeat-{self.model_id}")

    async def _heartbeat_loop(self) -> None:
        try:
            interval = max(1.0, float(ClientConfig().heartbeat_interval))
        except Exception:
            interval = 5.0
        assert self._stop is not None
        while True:
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
                return
            except asyncio.TimeoutError:
                pass
            try:
                now = time.time()
                active = self._active_count > 0 or now < self._active_until
                state = "inference_active" if active else "loaded_idle"
                status = await self.touch(active=active, state=state)
                if status in LOST_LEASE_STATUSES:
                    return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("cudabroker heartbeat failed for %s: %s", self.model_id, e)

    async def _touch_locked(self, active: bool = True, vram_sample_mb: float | None = None, state: str | None = None) -> str:
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
        return await self.lease.touch(active=active, vram_sample_mb=vram_sample_mb, state=state)

    async def touch(self, active: bool = True, vram_sample_mb: float | None = None, state: str | None = None) -> str:
        async with self._lock:
            status = await self._touch_locked(active=active, vram_sample_mb=vram_sample_mb, state=state)
        if status in LOST_LEASE_STATUSES:
            await self._evict_to_cpu()
        return status

    async def begin_inference(self) -> T:
        lost_status: str | None = None
        async with self._lock:
            if self.model is None:
                raise RuntimeError(f"model {self.model_id} is not loaded")
            if self.current_device == "cpu":
                return self.model
            if self.current_device != "cuda" or not self.lease:
                raise RuntimeError(f"model {self.model_id} has no active GPU lease")
            status = await self.lease.touch(active=True, state="inference_active")
            if status in LOST_LEASE_STATUSES:
                lost_status = status
            else:
                self._active_count += 1
                return self.model

        assert lost_status is not None
        await self._evict_to_cpu()
        async with self._lock:
            if self.current_device == "cpu" and self.model is not None:
                return self.model
        raise RuntimeError(f"GPU lease lost for {self.model_id}: {lost_status}")

    async def end_inference(self) -> None:
        lost_status: str | None = None
        async with self._lock:
            if self._active_count > 0:
                self._active_count -= 1
            if self._active_count > 0 or not self.lease:
                return
            status = await self.lease.touch(active=False, state="loaded_idle")
            if status in LOST_LEASE_STATUSES:
                lost_status = status
        if lost_status is not None:
            await self._evict_to_cpu()

    @asynccontextmanager
    async def inference(self):
        model = await self.begin_inference()
        try:
            yield model
        finally:
            await self.end_inference()

    async def _evict_to_cpu(self) -> None:
        async with self._lock:
            if self.current_device != "cuda":
                return
            self.model = None
            self.current_device = "evicted"
            self._active_count = 0
            lease = self.lease
            cm = self._cm
            self.lease = None
            self._cm = None

        # The VRAM release must happen before evict_ack permits queued work.
        self._empty_cuda_cache()
        if lease:
            await lease.evict_ack()
        if cm:
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                pass
        if self.cpu_fallback:
            model = await _maybe_await(self.cpu_fallback())
            async with self._lock:
                self.model = model
                self.current_device = "cpu"

    async def release(self) -> None:
        task: asyncio.Task[None] | None = None
        async with self._lock:
            if self._stop:
                self._stop.set()
            task = self._task
            if self.lease:
                try:
                    await self.lease.touch(active=False, state="unloading")
                except Exception:
                    pass
            self.model = None
            self.current_device = "unloaded"
            self._active_count = 0
            lease = self.lease
            cm = self._cm
            self.lease = None
            self._cm = None
            self._task = None

        if lease:
            await lease.release()
        if cm:
            await cm.__aexit__(None, None, None)
        if task and task is not asyncio.current_task():
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()
        self._empty_cuda_cache()

    @staticmethod
    def _empty_cuda_cache() -> None:
        # Native CUDA runtimes may free VRAM only when their model object's
        # finalizer runs.  Collect before clearing PyTorch's allocator cache.
        gc.collect()
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
