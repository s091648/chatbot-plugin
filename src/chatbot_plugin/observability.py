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


def _add_otel_context(record: logging.LogRecord) -> bool:
    """logging.Filter that injects the current OTel trace_id/span_id onto the
    record so Loki log lines correlate with Tempo traces by trace_id — mirrors
    backend/observability.py's _add_otel_context structlog processor. Always
    returns True (a no-op filter never drops records); safe when no
    TracerProvider is configured, since the default tracer's span context is
    simply invalid."""
    try:
        from opentelemetry import trace as _otel_trace
        ctx = _otel_trace.get_current_span().get_span_context()
        if ctx.is_valid:
            record.trace_id = format(ctx.trace_id, "032x")
            record.span_id = format(ctx.span_id, "016x")
    except Exception:
        pass
    return True


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
            payload["exception"] = _format_filtered_exception(record.exc_info)
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
    stdout.addFilter(_add_otel_context)
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
            loki_handler.addFilter(_add_otel_context)
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


def setup_tracing(app_env: str, otlp_endpoint: str, otlp_user: str, api_key: str):
    """Initialize OTel tracing with a Grafana Cloud OTLP exporter and return the
    TracerProvider. Returns None (no-op tracer) if any of otlp_endpoint/
    otlp_user/api_key are absent.

    Mirrors backend/observability.py's setup_tracing() — same Grafana Cloud
    tenant/credentials, separate service.name so traces from this service are
    distinguishable from backend's/the scraper's. No Redis instrumentation
    (unlike backend): this service has no Redis dependency. AsyncPgBackend
    (chatbot_plugin_sdk) uses a real SQLAlchemy AsyncEngine under the
    postgresql+asyncpg dialect, so SQLAlchemyInstrumentor still applies —
    called with no `engine=` kwarg here since AsyncPgBackend doesn't expose
    its engine publicly; the instrumentor's global hook still picks it up.
    """
    if not all([otlp_endpoint, otlp_user, api_key]):
        missing = [
            k
            for k, v in {
                "GRAFANA_OTLP_ENDPOINT": otlp_endpoint,
                "GRAFANA_OTLP_USER": otlp_user,
                "GRAFANA_API_KEY": api_key,
            }.items()
            if not v
        ]
        print(f"[tracing] Skipping OTLP setup, missing env vars: {missing}", file=sys.stdout)
        return None

    try:
        import base64
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        auth_str = f"{otlp_user}:{api_key}"
        encoded_auth = base64.b64encode(auth_str.encode()).decode()

        resource = Resource.create({
            "service.name": "chatbot-plugin",
            "deployment.environment": app_env,
        })
        exporter = OTLPSpanExporter(
            endpoint=f"{otlp_endpoint.rstrip('/')}/v1/traces",
            headers={"Authorization": f"Basic {encoded_auth}"},
            timeout=15,
        )
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        # Without this, FastAPIInstrumentor below only ever produces one flat span per
        # request — no visibility into whether a slow request was slow because of a DB
        # query (pgvector search) vs. the LLM call itself. Wrapped in try/except so a
        # failure here doesn't take down request-level tracing, which already succeeded.
        try:
            from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
            SQLAlchemyInstrumentor().instrument(tracer_provider=provider)
        except Exception as e:
            print(f"[tracing] SQLAlchemy instrumentation failed: {e}", file=sys.stdout)

        # Two purposes: (1) continues the caller's trace when backend proxies in with a
        # traceparent header (see backend/observability.py's matching HTTPXClientInstrumentor
        # — without it on *both* sides, backend's and this service's traces are disconnected,
        # each starting its own trace_id), and (2) gives outgoing calls to Gemini/the
        # embedding endpoint/OpenRouter their own spans, finer-grained than the manual
        # chat.retrieve/chat.llm_stream spans in chat_service.py.
        try:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
            HTTPXClientInstrumentor().instrument(tracer_provider=provider)
        except Exception as e:
            print(f"[tracing] httpx client instrumentation failed: {e}", file=sys.stdout)

        print("[tracing] OTLP setup successful", file=sys.stdout)
        return provider
    except Exception as e:
        print(f"[tracing] OTLP setup failed: {e}", file=sys.stdout)
        return None
