from __future__ import annotations

import os
from dataclasses import dataclass


def _bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class BrokerConfig:
    host: str = os.getenv("CUDABROKER_HOST", "127.0.0.1")
    port: int = int(os.getenv("CUDABROKER_PORT", "7777"))
    headroom_mb: float = float(os.getenv("CUDABROKER_HEADROOM_MB", "512"))
    stat_mode: str = os.getenv("CUDABROKER_STAT_MODE", "p98").lower()
    strict: bool = _bool("CUDABROKER_STRICT", False)
    fits_concurrent: bool = _bool("CUDABROKER_FITS_CONCURRENT", True)
    sample_interval: float = float(os.getenv("CUDABROKER_SAMPLE_INTERVAL", "1.5"))
    gpu_sampler: str = os.getenv("CUDABROKER_GPU_SAMPLER", "auto").strip().lower()
    lease_timeout: float = float(os.getenv("CUDABROKER_LEASE_TIMEOUT", "20"))
    default_ttl_seconds: float = float(os.getenv("CUDABROKER_DEFAULT_TTL_SECONDS", "600"))
    stats_limit: int = int(os.getenv("CUDABROKER_STATS_LIMIT", "200"))


def get_config() -> BrokerConfig:
    return BrokerConfig()
