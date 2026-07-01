from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ClientConfig:
    url: str = os.getenv("CUDABROKER_URL", "http://127.0.0.1:7777")
    client_id: str = os.getenv("CUDABROKER_CLIENT_ID", "default")
    timeout: float = float(os.getenv("CUDABROKER_TIMEOUT", "5"))
    heartbeat_interval: float = float(os.getenv("CUDABROKER_HEARTBEAT_INTERVAL", "0"))
