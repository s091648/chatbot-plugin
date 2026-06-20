"""JSON stdout logging + optional Loki shipping.

Call configure_logging() once at process startup (before app creation).
Extra fields passed via extra={"key": "val"} in logging calls are included
as top-level JSON keys, matching the scraper's structlog format.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

_STANDARD_RECORD_KEYS: frozenset[str] = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None))
)


class _JsonFormatter(logging.Formatter):
    def __init__(self, service: str) -> None:
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "event": record.getMessage(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "service": self._service,
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
        }
        for key, val in vars(record).items():
            if key not in _STANDARD_RECORD_KEYS and not key.startswith("_"):
                payload[key] = val
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(
    service: str,
    loki_url: str = "",
    loki_user: str = "",
    loki_api_key: str = "",
    app_env: str = "local",
) -> None:
    """Attach JSON stdout handler to root logger + optional Loki sink.

    Also routes chatbot_plugin_sdk stdlib logs through the same formatter
    so SDK records appear in consistent JSON (not plain text).
    """
    fmt = _JsonFormatter(service)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.INFO)
    stdout.setFormatter(fmt)
    root.addHandler(stdout)

    loki_handler: logging.Handler | None = None
    if all([loki_url, loki_user, loki_api_key]):
        try:
            from logging_loki import LokiHandler  # type: ignore[import]
            loki_handler = LokiHandler(
                url=f"{loki_url.rstrip('/')}/push",
                auth=(loki_user, loki_api_key),
                tags={"app": service, "env": app_env},
                version="1",
            )
            loki_handler.setLevel(logging.INFO)
            loki_handler.setFormatter(fmt)
            root.addHandler(loki_handler)
        except Exception as exc:
            print(f"Loki handler setup failed: {exc}", file=sys.stdout)

    # Route SDK logs through the same JSON formatter, suppress plain-text duplicate
    sdk_logger = logging.getLogger("chatbot_plugin_sdk")
    sdk_logger.setLevel(logging.DEBUG)
    sdk_logger.addHandler(stdout)
    if loki_handler is not None:
        sdk_logger.addHandler(loki_handler)
    sdk_logger.propagate = False
