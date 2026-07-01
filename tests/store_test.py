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
