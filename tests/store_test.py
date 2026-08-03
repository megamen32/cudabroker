from cudabroker.store import BrokerStore, ModelInfo


def test_footprint_uses_declared_until_enough_samples():
    s = BrokerStore(stats_limit=5)
    s.register(ModelInfo("m", "c", declared_vram_mb=100))
    s.add_sample("m", 50)
    assert s.footprint("m", "p98") == 100


def test_footprint_modes():
    s = BrokerStore(stats_limit=10)
    s.register(ModelInfo("m", "c", declared_vram_mb=0))
    for x in [10, 20, 30, 40, 100]:
        s.add_sample("m", x)
    assert s.footprint("m", "mean") == 40
    assert s.footprint("m", "max") == 100
    assert s.footprint("m", "p98") >= 40


def test_ensure_model_refreshes_persisted_runtime_metadata():
    store = BrokerStore()
    store.register(
        ModelInfo(
            "whisper-large-v3",
            "whisper",
            declared_vram_mb=5200,
            gpu_priority=8,
            cpu_capable=False,
            ttl_seconds=900,
        )
    )

    refreshed = store.ensure_model(
        "whisper-large-v3",
        "whisper",
        declared_vram_mb=None,
        gpu_priority=8,
        cpu_capable=False,
        ttl_seconds=86400,
    )

    assert refreshed.declared_vram_mb == 5200
    assert refreshed.ttl_seconds == 86400


def test_state_round_trip_preserves_live_leases_models_and_samples(tmp_path):
    path = tmp_path / "state.json"
    source = BrokerStore(stats_limit=5)
    source.register(ModelInfo("embedding", "memory", declared_vram_mb=4800, gpu_priority=0, cpu_capable=True))
    lease = source.new_lease("embedding", "memory", "active", worker_id="w1", request_id="r1")
    lease.state = "loaded_idle"
    source.add_sample("embedding", 4500)
    source.add_sample("embedding", 4600)
    assert source.save(path) is True

    restored = BrokerStore(stats_limit=5)
    assert restored.load(path) is True

    assert restored.models["embedding"].gpu_priority == 0
    assert restored.models["embedding"].declared_vram_mb == 4800
    assert restored.leases[lease.lease_id].state == "loaded_idle"
    assert restored.leases[lease.lease_id].request_id == "r1"
    assert list(restored.stats["embedding"]) == [4500.0, 4600.0]


def test_state_snapshot_drops_terminal_leases(tmp_path):
    path = tmp_path / "state.json"
    source = BrokerStore()
    source.register(ModelInfo("m", "c", declared_vram_mb=100))
    live = source.new_lease("m", "c", "active")
    terminal = source.new_lease("m", "c", "active")
    source.release(terminal.lease_id)
    source.save(path)

    restored = BrokerStore()
    assert restored.load(path) is True
    assert set(restored.leases) == {live.lease_id}


def test_corrupt_state_is_ignored_without_destroying_empty_store(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("not json", encoding="utf-8")
    restored = BrokerStore()
    assert restored.load(path) is False
    assert restored.models == {}
    assert restored.leases == {}
