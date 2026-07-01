from __future__ import annotations

import logging
import subprocess
import threading

from .store import BrokerStore

log = logging.getLogger(__name__)

VALID_SAMPLERS = {"auto", "torch", "nvidia-smi", "nvidia_smi", "smi", "none"}


def _read_torch_cuda_mem_mb() -> tuple[float, float] | None:
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            return None
        free_b, total_b = torch.cuda.mem_get_info()
        return total_b / 1024 / 1024, free_b / 1024 / 1024
    except Exception as e:
        log.debug("torch cuda mem_get_info unavailable: %s", e)
        return None


def _read_nvidia_smi_mem_mb() -> tuple[float, float] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        total = 0.0
        free = 0.0
        for line in out.splitlines():
            if not line.strip():
                continue
            t, f = [float(x.strip()) for x in line.split(",")[:2]]
            total += t
            free += f
        if total > 0:
            return total, free
    except Exception as e:
        log.debug("nvidia-smi memory query unavailable: %s", e)
    return None


def read_cuda_mem_mb(sampler: str = "auto") -> tuple[float, float]:
    sampler = (sampler or "auto").strip().lower()
    if sampler not in VALID_SAMPLERS:
        log.warning("Unknown CUDABROKER_GPU_SAMPLER=%r, falling back to auto", sampler)
        sampler = "auto"

    if sampler == "none":
        return 0.0, 0.0
    if sampler == "torch":
        return _read_torch_cuda_mem_mb() or (0.0, 0.0)
    if sampler in {"nvidia-smi", "nvidia_smi", "smi"}:
        return _read_nvidia_smi_mem_mb() or (0.0, 0.0)

    # auto: prefer torch when it is installed in the broker process and CUDA is visible,
    # then fall back to nvidia-smi. This lets cudabroker run without torch.
    return _read_torch_cuda_mem_mb() or _read_nvidia_smi_mem_mb() or (0.0, 0.0)


class GpuSampler:
    def __init__(self, store: BrokerStore, interval: float = 1.5, sampler: str = "auto"):
        self.store = store
        self.interval = interval
        self.sampler = sampler
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="cudabroker-gpu-sampler", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            total, free = read_cuda_mem_mb(self.sampler)
            self.store.set_gpu(total, free)
            self._stop.wait(self.interval)
