from cudabroker.gpu import read_cuda_mem_mb


def test_sampler_none_returns_zero():
    assert read_cuda_mem_mb("none") == (0.0, 0.0)


def test_unknown_sampler_falls_back(monkeypatch):
    monkeypatch.setattr("cudabroker.gpu._read_torch_cuda_mem_mb", lambda: None)
    monkeypatch.setattr("cudabroker.gpu._read_nvidia_smi_mem_mb", lambda: (10.0, 4.0))
    assert read_cuda_mem_mb("bad") == (10.0, 4.0)


def test_torch_mode_does_not_fallback_to_smi(monkeypatch):
    monkeypatch.setattr("cudabroker.gpu._read_torch_cuda_mem_mb", lambda: None)
    monkeypatch.setattr("cudabroker.gpu._read_nvidia_smi_mem_mb", lambda: (10.0, 4.0))
    assert read_cuda_mem_mb("torch") == (0.0, 0.0)


def test_nvidia_smi_mode(monkeypatch):
    monkeypatch.setattr("cudabroker.gpu._read_torch_cuda_mem_mb", lambda: (20.0, 9.0))
    monkeypatch.setattr("cudabroker.gpu._read_nvidia_smi_mem_mb", lambda: (10.0, 4.0))
    assert read_cuda_mem_mb("nvidia-smi") == (10.0, 4.0)


def test_auto_prefers_torch(monkeypatch):
    monkeypatch.setattr("cudabroker.gpu._read_torch_cuda_mem_mb", lambda: (20.0, 9.0))
    monkeypatch.setattr("cudabroker.gpu._read_nvidia_smi_mem_mb", lambda: (10.0, 4.0))
    assert read_cuda_mem_mb("auto") == (20.0, 9.0)


def test_auto_falls_back_to_nvidia_smi(monkeypatch):
    monkeypatch.setattr("cudabroker.gpu._read_torch_cuda_mem_mb", lambda: None)
    monkeypatch.setattr("cudabroker.gpu._read_nvidia_smi_mem_mb", lambda: (10.0, 4.0))
    assert read_cuda_mem_mb("auto") == (10.0, 4.0)
