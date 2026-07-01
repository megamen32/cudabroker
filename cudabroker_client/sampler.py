from __future__ import annotations


def memory_allocated_mb() -> float | None:
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.memory_allocated()) / 1024 / 1024
    except Exception:
        return None
