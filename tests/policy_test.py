import time

from cudabroker.config import BrokerConfig
from cudabroker.policy import Policy
from cudabroker.store import BrokerStore, ModelInfo


def mk(strict=False):
    cfg = BrokerConfig(headroom_mb=100, strict=strict, fits_concurrent=True, stat_mode="p98")
    s = BrokerStore()
    s.set_gpu(total_mb=10_000, free_mb=10_000)
    return s, Policy(s, cfg)


def test_concurrent_when_fits():
    s, p = mk()
    s.register(ModelInfo("a", "c", declared_vram_mb=1000))
    s.register(ModelInfo("b", "c", declared_vram_mb=1000))
    assert p.acquire("a", "c").status == "granted"
    s.active_leases()[0].state = "loaded_idle"
    assert p.acquire("b", "c").status == "granted"
    assert len(s.active_leases()) == 2


def test_strict_one_at_a_time():
    s, p = mk(strict=True)
    s.register(ModelInfo("a", "c", declared_vram_mb=1000))
    s.register(ModelInfo("b", "c", declared_vram_mb=1000))
    assert p.acquire("a", "c").status == "granted"
    assert p.acquire("b", "c").status == "queued"


def test_evict_low_priority_cpu_capable_goes_evicting_not_free():
    s, p = mk()
    s.set_gpu(total_mb=5000, free_mb=5000)
    s.register(ModelInfo("low", "c", declared_vram_mb=4500, gpu_priority=1, cpu_capable=True, ttl_seconds=600))
    s.register(ModelInfo("new", "c", declared_vram_mb=4000, gpu_priority=9, cpu_capable=False, ttl_seconds=600))
    assert p.acquire("low", "c").status == "granted"
    low = s.active_leases()[0]
    low.state = "loaded_idle"
    r = p.acquire("new", "c")
    assert r.status == "queued"
    assert r.evicted
    assert s.leases[low.lease_id].status == "evicting"
    assert len(s.reserving_leases()) == 1


def test_release_after_evict_allows_promotion():
    s, p = mk()
    s.set_gpu(total_mb=5000, free_mb=5000)
    s.register(ModelInfo("low", "c", declared_vram_mb=4500, gpu_priority=1, cpu_capable=True, ttl_seconds=600))
    s.register(ModelInfo("new", "c", declared_vram_mb=4000, gpu_priority=9, cpu_capable=False, ttl_seconds=600))
    p.acquire("low", "c")
    low = s.active_leases()[0]
    low.state = "loaded_idle"
    r = p.acquire("new", "c")
    assert r.status == "queued"
    s.release(low.lease_id)
    p.try_promote_queue()
    assert s.leases[r.lease.lease_id].status == "active"


def test_ttl_evict_only_idle():
    s, p = mk()
    s.register(ModelInfo("a", "c", declared_vram_mb=1000, cpu_capable=True, ttl_seconds=0.01))
    p.acquire("a", "c")
    l = s.active_leases()[0]
    l.last_active = time.time() - 1
    l.state = "inference_active"
    s.register(ModelInfo("b", "c", declared_vram_mb=1000))
    p.acquire("b", "c")
    assert l.status == "active"
    l.state = "loaded_idle"
    p.acquire("c", "c")
    assert l.status == "evicting"


def test_acquire_idempotent_for_same_request_id():
    s, p = mk(strict=True)
    s.register(ModelInfo("a", "c", declared_vram_mb=1000))
    s.register(ModelInfo("b", "c", declared_vram_mb=1000))
    assert p.acquire("a", "c").status == "granted"
    r1 = p.acquire("b", "c", worker_id="w1", request_id="r1")
    r2 = p.acquire("b", "c", worker_id="w1", request_id="r1")
    assert r1.status == "queued"
    assert r2.status == "queued"
    assert r1.lease.lease_id == r2.lease.lease_id
    assert len(s.queued_leases()) == 1
