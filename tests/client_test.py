import pytest

from cudabroker_client.managed import ManagedModel
from cudabroker_client.transport import TransportResponse


def test_managed_model_local_mode(monkeypatch):
    def fake_post(self, path, payload):
        return TransportResponse(False, {}, True)
    monkeypatch.setattr("cudabroker_client.transport.BrokerTransport.post", fake_post)
    monkeypatch.delenv("CUDABROKER_REQUIRED", raising=False)
    mm = ManagedModel("m", loader=lambda: {"gpu": True}, cpu_fallback=lambda: {"cpu": True})
    assert mm.acquire() == {"gpu": True}
    assert mm.current_device == "cuda"
    mm.release()
    assert mm.current_device == "unloaded"


def test_required_mode_fails_when_broker_down(monkeypatch):
    def fake_post(self, path, payload):
        return TransportResponse(False, {}, True)
    monkeypatch.setattr("cudabroker_client.transport.BrokerTransport.post", fake_post)
    monkeypatch.setenv("CUDABROKER_REQUIRED", "1")
    mm = ManagedModel("m", loader=lambda: "gpu")
    with pytest.raises(RuntimeError):
        mm.acquire()


def test_managed_model_evict_to_cpu(monkeypatch):
    calls = {"n": 0}
    def fake_post(self, path, payload):
        if path == "/v1/acquire":
            return TransportResponse(True, {"lease_id": "l1", "status": "granted"}, False)
        if path == "/v1/heartbeat":
            calls["n"] += 1
            status = "evicted" if calls["n"] >= 3 else "active"
            return TransportResponse(True, {"status": status}, False)
        return TransportResponse(True, {"ok": True}, False)
    monkeypatch.setattr("cudabroker_client.transport.BrokerTransport.post", fake_post)
    mm = ManagedModel("m", loader=lambda: "gpu", cpu_fallback=lambda: "cpu")
    assert mm.acquire() == "gpu"
    assert mm.touch() == "evicted"
    assert mm.model == "cpu"
    assert mm.current_device == "cpu"
