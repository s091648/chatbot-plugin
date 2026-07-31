"""JSON stdout logging + optional Loki shipping.

Call configure_logging() once at process startup (before app creation).
Extra fields passed via extra={"key": "val"} in logging calls are included
as top-level JSON keys, matching the scraper's structlog format.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone

_STANDARD_RECORD_KEYS: frozenset[str] = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None))
)

# In-app prefixes for traceback filtering: this package's own source dir plus
# any installed first-party package whose internals are worth seeing in full.
_IN_APP_PACKAGES = ["chatbot_plugin_sdk"]


def _resolve_in_app_prefixes() -> list[str]:
    prefixes = [os.path.dirname(os.path.abspath(__file__))]
    for name in _IN_APP_PACKAGES:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            continue
        if spec and spec.origin:
            prefixes.append(os.path.dirname(spec.origin))
    return prefixes


_IN_APP_PREFIXES = _resolve_in_app_prefixes()


def _format_single(exc_type, exc, tb) -> str:
    """Plain-text traceback keeping only frames under _IN_APP_PREFIXES; falls
    back to the full traceback if that would discard every frame."""
    frames = traceback.extract_tb(tb)
    selected = [f for f in frames if any(f.filename.startswith(p) for p in _IN_APP_PREFIXES)]
    if not selected:
        selected = list(frames)
    lines = ["Traceback (most recent call last):\n"]
    omitted = len(frames) - len(selected)
    if omitted > 0:
        lines.append(f"  ... {omitted} frame(s) outside this project/whitelisted packages omitted ...\n")
    lines += traceback.format_list(selected)
    lines += traceback.format_exception_only(exc_type, exc)
    return "".join(lines)


def _format_filtered_exception(exc_info) -> str:
    """Render exc_info as plain text, keeping only in-app frames at every
    level of the __cause__/__context__ chain (mirrors how
    traceback.format_exception() walks chained exceptions)."""
    _, top_exc, top_tb = exc_info

    chain = []
    current = top_exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        if current.__cause__ is not None:
            current = current.__cause__
        elif current.__context__ is not None and not current.__suppress_context__:
            current = current.__context__
        else:
            current = None
    chain.reverse()

    parts = []
    for i, exc in enumerate(chain):
        if i > 0:
            prev = chain[i - 1]
            connector = (
                "\nThe above exception was the direct cause of the following exception:\n\n"
                if exc.__cause__ is prev
                else "\nDuring handling of the above exception, another exception occurred:\n\n"
            )
            parts.append(connector)
        tb = top_tb if exc is top_exc else exc.__traceback__
        parts.append(_format_single(type(exc), exc, tb))
    return "".join(parts)


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
            payload["exc_info"] = _format_filtered_exception(record.exc_info)
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
