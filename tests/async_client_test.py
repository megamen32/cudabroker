import pytest

from cudabroker_client import AsyncManagedModel, gpu_lease_async
from cudabroker_client.lease import AsyncLeaseHandle
from cudabroker_client.transport import TransportResponse


@pytest.mark.asyncio
async def test_async_gpu_lease_local_mode(monkeypatch):
    async def fake_post(self, path, payload):
        return TransportResponse(False, {}, True)

    monkeypatch.setattr("cudabroker_client.transport.AsyncBrokerTransport.post", fake_post)
    monkeypatch.delenv("CUDABROKER_REQUIRED", raising=False)

    async with gpu_lease_async("m") as lease:
        assert lease.local_mode is True
        assert await lease.touch() == "active"


@pytest.mark.asyncio
async def test_async_managed_model_local_mode(monkeypatch):
    gpu_loads = []

    async def fake_post(self, path, payload):
        return TransportResponse(False, {}, True)

    async def loader():
        gpu_loads.append(True)
        return {"gpu": True}

    monkeypatch.setattr("cudabroker_client.transport.AsyncBrokerTransport.post", fake_post)
    monkeypatch.delenv("CUDABROKER_REQUIRED", raising=False)

    mm = AsyncManagedModel("m", loader=loader, cpu_fallback=lambda: {"cpu": True})
    assert await mm.acquire() == {"cpu": True}
    assert mm.current_device == "cpu"
    assert gpu_loads == []
    await mm.release()
    assert mm.current_device == "unloaded"


@pytest.mark.asyncio
async def test_async_managed_model_gpu_only_fails_closed_when_broker_down(monkeypatch):
    async def fake_post(self, path, payload):
        return TransportResponse(False, {}, True)

    monkeypatch.setattr("cudabroker_client.transport.AsyncBrokerTransport.post", fake_post)
    monkeypatch.delenv("CUDABROKER_REQUIRED", raising=False)
    mm = AsyncManagedModel("m", loader=lambda: "gpu")
    with pytest.raises(RuntimeError, match="GPU load refused"):
        await mm.acquire()


@pytest.mark.asyncio
async def test_async_granted_lease_recovers_after_temporary_broker_unavailability():
    class Transport:
        def __init__(self):
            self.calls = 0

        async def post(self, path, payload):
            self.calls += 1
            if self.calls == 1:
                return TransportResponse(False, {}, True)
            return TransportResponse(True, {"status": "active"}, False)

    handle = AsyncLeaseHandle("l1", Transport())
    assert await handle.touch(active=False) == "unavailable"
    assert handle.local_mode is False
    assert await handle.touch(active=False) == "active"


@pytest.mark.asyncio
async def test_async_managed_model_evict_to_cpu(monkeypatch):
    calls = {"n": 0}

    async def fake_post(self, path, payload):
        if path == "/v1/acquire":
            return TransportResponse(True, {"lease_id": "l1", "status": "granted"}, False)
        if path == "/v1/heartbeat":
            calls["n"] += 1
            status = "evicted" if calls["n"] >= 3 else "active"
            return TransportResponse(True, {"status": status}, False)
        return TransportResponse(True, {"ok": True}, False)

    monkeypatch.setattr("cudabroker_client.transport.AsyncBrokerTransport.post", fake_post)
    mm = AsyncManagedModel("m", loader=lambda: "gpu", cpu_fallback=lambda: "cpu")
    assert await mm.acquire() == "gpu"
    assert await mm.touch() == "evicted"
    assert mm.model == "cpu"
    assert mm.current_device == "cpu"


@pytest.mark.asyncio
async def test_async_inference_guard_never_yields_evicted_gpu_reference(monkeypatch):
    heartbeats = 0

    async def fake_post(self, path, payload):
        nonlocal heartbeats
        if path == "/v1/acquire":
            return TransportResponse(True, {"lease_id": "l1", "status": "granted"}, False)
        if path == "/v1/heartbeat":
            heartbeats += 1
            status = "active" if heartbeats <= 2 else "evicted"
            return TransportResponse(True, {"status": status}, False)
        return TransportResponse(True, {"ok": True}, False)

    monkeypatch.setattr("cudabroker_client.transport.AsyncBrokerTransport.post", fake_post)
    mm = AsyncManagedModel("m", loader=lambda: "stale-gpu", cpu_fallback=lambda: "safe-cpu")
    assert await mm.acquire() == "stale-gpu"

    async with mm.inference() as model:
        assert model == "safe-cpu"
        assert mm.current_device == "cpu"


@pytest.mark.asyncio
async def test_async_inference_guard_tracks_active_scope_and_idle_transition(monkeypatch):
    states = []

    async def fake_post(self, path, payload):
        if path == "/v1/acquire":
            return TransportResponse(True, {"lease_id": "l1", "status": "granted"}, False)
        if path == "/v1/heartbeat":
            states.append((payload["active"], payload["state"]))
            return TransportResponse(True, {"status": "active"}, False)
        return TransportResponse(True, {"ok": True}, False)

    monkeypatch.setattr("cudabroker_client.transport.AsyncBrokerTransport.post", fake_post)
    mm = AsyncManagedModel("m", loader=lambda: "gpu")
    await mm.acquire()
    async with mm.inference() as model:
        assert model == "gpu"
        assert mm._active_count == 1
    assert mm._active_count == 0
    assert (True, "inference_active") in states
    assert states[-1] == (False, "loaded_idle")


@pytest.mark.asyncio
@pytest.mark.parametrize("lost_status", ["unknown", "dead", "released"])
async def test_async_managed_model_lost_lease_unloads_gpu_and_falls_back(monkeypatch, lost_status):
    events = []
    heartbeats = 0

    async def fake_post(self, path, payload):
        nonlocal heartbeats
        if path == "/v1/acquire":
            return TransportResponse(True, {"lease_id": "l1", "status": "granted"}, False)
        if path == "/v1/heartbeat":
            heartbeats += 1
            status = "active" if heartbeats <= 2 else lost_status
            return TransportResponse(True, {"status": status}, False)
        if path == "/v1/evict_ack":
            events.append("ack")
        return TransportResponse(True, {"ok": True}, False)

    monkeypatch.setattr("cudabroker_client.transport.AsyncBrokerTransport.post", fake_post)
    monkeypatch.setattr(AsyncManagedModel, "_empty_cuda_cache", staticmethod(lambda: events.append("empty")))
    mm = AsyncManagedModel("m", loader=lambda: "gpu", cpu_fallback=lambda: "cpu")
    assert await mm.acquire() == "gpu"

    assert await mm.touch() == lost_status
    assert mm.model == "cpu"
    assert mm.current_device == "cpu"
    assert events[:2] == ["empty", "ack"]

@pytest.mark.asyncio
async def test_async_managed_model_context_manager_releases(monkeypatch):
    async def fake_post(self, path, payload):
        if path == "/v1/acquire":
            return TransportResponse(True, {"lease_id": "l1", "status": "granted"}, False)
        if path == "/v1/heartbeat":
            return TransportResponse(True, {"status": "active"}, False)
        return TransportResponse(True, {"ok": True}, False)

    monkeypatch.setattr("cudabroker_client.transport.AsyncBrokerTransport.post", fake_post)
    mm = AsyncManagedModel("m", loader=lambda: "gpu")
    async with mm as model:
        assert model == "gpu"
        assert mm.current_device == "cuda"
    assert mm.current_device == "unloaded"
