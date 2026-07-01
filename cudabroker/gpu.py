from __future__ import annotations

import logging
import threading
import time

from .store import BrokerStore

log = logging.getLogger(__name__)


def read_cuda_mem_mb() -> tuple[float, float]:
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            return 0.0, 0.0
        free_b, total_b = torch.cuda.mem_get_info()
        return total_b / 1024 / 1024, free_b / 1024 / 1024
    except Exception as e:  # torch absent is valid for tests/dev
        log.debug("cuda mem_get_info unavailable: %s", e)
        return 0.0, 0.0


class GpuSampler:
    def __init__(self, store: BrokerStore, interval: float = 1.5):
        self.store = store
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="cudabroker-gpu-sampler", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            total, free = read_cuda_mem_mb()
            self.store.set_gpu(total, free)
            self._stop.wait(self.interval)
