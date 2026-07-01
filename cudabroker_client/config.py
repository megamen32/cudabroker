from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


@dataclass(frozen=True)
class ClientConfig:
    url: str = field(default_factory=lambda: _env("CUDABROKER_URL", "http://127.0.0.1:7777"))
    client_id: str = field(default_factory=lambda: _env("CUDABROKER_CLIENT_ID", "default"))
    worker_id: str = field(default_factory=lambda: _env("CUDABROKER_WORKER_ID", _default_worker_id()))
    timeout: float = field(default_factory=lambda: _env_float("CUDABROKER_TIMEOUT", "5"))
    heartbeat_interval: float = field(default_factory=lambda: _env_float("CUDABROKER_HEARTBEAT_INTERVAL", "5"))
    active_grace_seconds: float = field(default_factory=lambda: _env_float("CUDABROKER_ACTIVE_GRACE_SECONDS", "30"))
    required: bool = field(default_factory=lambda: _env_bool("CUDABROKER_REQUIRED", "0"))
