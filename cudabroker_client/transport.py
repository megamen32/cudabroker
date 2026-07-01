from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .config import ClientConfig

log = logging.getLogger(__name__)


@dataclass
class TransportResponse:
    ok: bool
    data: dict[str, Any]
    local_mode: bool = False


class BrokerTransport:
    def __init__(self, config: ClientConfig | None = None):
        self.config = config or ClientConfig()
        self.local_mode = False
        self._last_warning = 0.0

    def _warn(self, exc: Exception) -> None:
        now = time.time()
        if now - self._last_warning > 60:
            log.warning("cudabroker unavailable, switching to local_mode: %s", exc)
            self._last_warning = now

    def post(self, path: str, payload: dict[str, Any]) -> TransportResponse:
        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                r = client.post(self.config.url.rstrip("/") + path, json=payload)
                r.raise_for_status()
                self.local_mode = False
                return TransportResponse(True, r.json(), False)
        except Exception as e:
            self.local_mode = True
            self._warn(e)
            return TransportResponse(False, {}, True)
