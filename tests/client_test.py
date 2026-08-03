import sys
from types import SimpleNamespace

import pytest

from cudabroker_client.managed import ManagedModel
from cudabroker_client.lease import LeaseHandle
from cudabroker_client.transport import TransportResponse


def test_empty_cuda_cache_collects_native_models_before_torch_cache(monkeypatch):
    events = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: events.append("torch-empty"),
            ipc_collect=lambda: events.append("torch-ipc"),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr("cudabroker_client.managed.gc.collect", lambda: events.append("gc"))

    ManagedModel._empty_cuda_cache()

    assert events == ["gc", "torch-empty", "torch-ipc"]


def test_managed_model_local_mode(monkeypatch):
    gpu_loads = []

    def fake_post(self, path, payload):
        return TransportResponse(False, {}, True)
    monkeypatch.setattr("cudabroker_client.transport.BrokerTransport.post", fake_post)
    monkeypatch.delenv("CUDABROKER_REQUIRED", raising=False)
    mm = ManagedModel(
        "m",
        loader=lambda: gpu_loads.append(True) or {"gpu": True},
        cpu_fallback=lambda: {"cpu": True},
    )
    assert mm.acquire() == {"cpu": True}
    assert mm.current_device == "cpu"
    assert gpu_loads == []
    mm.release()
    assert mm.current_device == "unloaded"


def test_managed_model_gpu_only_fails_closed_when_broker_down(monkeypatch):
    def fake_post(self, path, payload):
        return TransportResponse(False, {}, True)

    monkeypatch.setattr("cudabroker_client.transport.BrokerTransport.post", fake_post)
    monkeypatch.delenv("CUDABROKER_REQUIRED", raising=False)
    mm = ManagedModel("m", loader=lambda: "gpu")
    with pytest.raises(RuntimeError, match="GPU load refused"):
        mm.acquire()


def test_required_mode_fails_when_broker_down(monkeypatch):
    def fake_post(self, path, payload):
        return TransportResponse(False, {}, True)
    monkeypatch.setattr("cudabroker_client.transport.BrokerTransport.post", fake_post)
    monkeypatch.setenv("CUDABROKER_REQUIRED", "1")
    mm = ManagedModel("m", loader=lambda: "gpu")
    with pytest.raises(RuntimeError):
        mm.acquire()


def test_granted_lease_recovers_after_temporary_broker_unavailability():
    class Transport:
        def __init__(self):
            self.calls = 0

        def post(self, path, payload):
            self.calls += 1
            if self.calls == 1:
                return TransportResponse(False, {}, True)
            return TransportResponse(True, {"status": "active"}, False)

    handle = LeaseHandle("l1", Transport())
    assert handle.touch(active=False) == "unavailable"
    assert handle.local_mode is False
    assert handle.touch(active=False) == "active"


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


def test_inference_guard_never_yields_evicted_gpu_reference(monkeypatch):
    heartbeats = 0

    def fake_post(self, path, payload):
        nonlocal heartbeats
        if path == "/v1/acquire":
            return TransportResponse(True, {"lease_id": "l1", "status": "granted"}, False)
        if path == "/v1/heartbeat":
            heartbeats += 1
            status = "active" if heartbeats <= 2 else "evicted"
            return TransportResponse(True, {"status": status}, False)
        return TransportResponse(True, {"ok": True}, False)

    monkeypatch.setattr("cudabroker_client.transport.BrokerTransport.post", fake_post)
    mm = ManagedModel("m", loader=lambda: "stale-gpu", cpu_fallback=lambda: "safe-cpu")
    assert mm.acquire() == "stale-gpu"

    with mm.inference() as model:
        assert model == "safe-cpu"
        assert mm.current_device == "cpu"


def test_inference_guard_tracks_active_scope_and_idle_transition(monkeypatch):
    states = []

    def fake_post(self, path, payload):
        if path == "/v1/acquire":
            return TransportResponse(True, {"lease_id": "l1", "status": "granted"}, False)
        if path == "/v1/heartbeat":
            states.append((payload["active"], payload["state"]))
            return TransportResponse(True, {"status": "active"}, False)
        return TransportResponse(True, {"ok": True}, False)

    monkeypatch.setattr("cudabroker_client.transport.BrokerTransport.post", fake_post)
    mm = ManagedModel("m", loader=lambda: "gpu")
    mm.acquire()
    with mm.inference() as model:
        assert model == "gpu"
        assert mm._active_count == 1
    assert mm._active_count == 0
    assert (True, "inference_active") in states
    assert states[-1] == (False, "loaded_idle")


@pytest.mark.parametrize("lost_status", ["unknown", "dead", "released"])
def test_managed_model_lost_lease_unloads_gpu_and_falls_back(monkeypatch, lost_status):
    events = []
    heartbeats = 0

    def fake_post(self, path, payload):
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

    monkeypatch.setattr("cudabroker_client.transport.BrokerTransport.post", fake_post)
    monkeypatch.setattr(ManagedModel, "_empty_cuda_cache", staticmethod(lambda: events.append("empty")))
    mm = ManagedModel("m", loader=lambda: "gpu", cpu_fallback=lambda: "cpu")
    assert mm.acquire() == "gpu"

    assert mm.touch() == lost_status
    assert mm.model == "cpu"
    assert mm.current_device == "cpu"
    assert events[:2] == ["empty", "ack"]

def test_managed_model_context_manager_releases(monkeypatch):
    def fake_post(self, path, payload):
        if path == "/v1/acquire":
            return TransportResponse(True, {"lease_id": "l1", "status": "granted"}, False)
        if path == "/v1/heartbeat":
            return TransportResponse(True, {"status": "active"}, False)
        return TransportResponse(True, {"ok": True}, False)

    monkeypatch.setattr("cudabroker_client.transport.BrokerTransport.post", fake_post)
    mm = ManagedModel("m", loader=lambda: "gpu")
    with mm as model:
        assert model == "gpu"
        assert mm.current_device == "cuda"
    assert mm.current_device == "unloaded"
