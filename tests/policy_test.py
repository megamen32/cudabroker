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


def test_lower_priority_request_cannot_evict_higher_priority_idle_model():
    s, p = mk()
    s.set_gpu(total_mb=5000, free_mb=5000)
    s.register(ModelInfo("whisper", "asr", declared_vram_mb=4500, gpu_priority=8, cpu_capable=True))
    s.register(ModelInfo("embedding", "memory", declared_vram_mb=4000, gpu_priority=0, cpu_capable=True))

    assert p.acquire("whisper", "asr").status == "granted"
    whisper = s.active_leases()[0]
    whisper.state = "loaded_idle"

    result = p.acquire("embedding", "memory")

    assert result.status == "queued"
    assert result.evicted == []
    assert whisper.status == "active"


def test_queued_high_priority_request_evicts_low_priority_once_inference_is_idle():
    s, p = mk()
    s.set_gpu(total_mb=5000, free_mb=5000)
    s.register(ModelInfo("embedding", "memory", declared_vram_mb=4500, gpu_priority=0, cpu_capable=True))
    s.register(ModelInfo("whisper", "asr", declared_vram_mb=4000, gpu_priority=8, cpu_capable=False))

    assert p.acquire("embedding", "memory").status == "granted"
    embedding = s.active_leases()[0]
    embedding.state = "inference_active"
    queued = p.acquire("whisper", "asr")
    assert queued.status == "queued"
    assert embedding.status == "active"

    embedding.state = "loaded_idle"
    p.try_promote_queue()

    assert embedding.status == "evicting"
    assert queued.lease.status == "queued"
    s.release(embedding.lease_id)
    p.try_promote_queue()
    assert queued.lease.status == "active"


def test_queue_promotes_highest_priority_before_fifo_order():
    s, p = mk()
    s.set_gpu(total_mb=5000, free_mb=5000)
    s.register(ModelInfo("blocker", "c", declared_vram_mb=4500, gpu_priority=9))
    s.register(ModelInfo("low", "c", declared_vram_mb=3000, gpu_priority=0))
    s.register(ModelInfo("high", "c", declared_vram_mb=3000, gpu_priority=8))

    blocker = p.acquire("blocker", "c").lease
    low = p.acquire("low", "c").lease
    high = p.acquire("high", "c").lease
    assert low.status == "queued"
    assert high.status == "queued"

    s.release(blocker.lease_id)
    p.try_promote_queue()

    assert high.status == "active"
    assert low.status == "queued"


def test_eviction_marks_only_the_minimum_lower_priority_footprint_needed():
    s, p = mk()
    s.set_gpu(total_mb=5000, free_mb=2000)
    s.register(ModelInfo("low-a", "c", declared_vram_mb=1500, gpu_priority=0, cpu_capable=True))
    s.register(ModelInfo("low-b", "c", declared_vram_mb=1500, gpu_priority=0, cpu_capable=True))
    s.register(ModelInfo("high", "c", declared_vram_mb=2500, gpu_priority=8))

    first = s.new_lease("low-a", "c", "active")
    second = s.new_lease("low-b", "c", "active")
    first.state = "loaded_idle"
    second.state = "loaded_idle"

    result = p.acquire("high", "c")

    assert result.status == "queued"
    assert len(result.evicted) == 1
    assert sum(lease.status == "evicting" for lease in (first, second)) == 1


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
    r1.lease.last_heartbeat = time.time() - 100
    r2 = p.acquire("b", "c", worker_id="w1", request_id="r1")
    assert r1.status == "queued"
    assert r2.status == "queued"
    assert r1.lease.lease_id == r2.lease.lease_id
    assert r2.lease.last_heartbeat > time.time() - 1
    assert len(s.queued_leases()) == 1
