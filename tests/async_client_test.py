import pytest

from cudabroker_client import AsyncManagedModel, gpu_lease_async
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
    async def fake_post(self, path, payload):
        return TransportResponse(False, {}, True)

    async def loader():
        return {"gpu": True}

    monkeypatch.setattr("cudabroker_client.transport.AsyncBrokerTransport.post", fake_post)
    monkeypatch.delenv("CUDABROKER_REQUIRED", raising=False)

    mm = AsyncManagedModel("m", loader=loader, cpu_fallback=lambda: {"cpu": True})
    assert await mm.acquire() == {"gpu": True}
    assert mm.current_device == "cuda"
    await mm.release()
    assert mm.current_device == "unloaded"


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
async def test_async_managed_model_context_manager_releases(monkeypatch):
    async def fake_post(self, path, payload):
        return TransportResponse(False, {}, True)

    monkeypatch.setattr("cudabroker_client.transport.AsyncBrokerTransport.post", fake_post)
    monkeypatch.delenv("CUDABROKER_REQUIRED", raising=False)

    mm = AsyncManagedModel("m", loader=lambda: "gpu")
    async with mm as model:
        assert model == "gpu"
        assert mm.current_device == "cuda"
    assert mm.current_device == "unloaded"
