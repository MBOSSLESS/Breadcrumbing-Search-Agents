"""Opt-in hierarchical runtime telemetry for latency experiments.

The recorder is deliberately independent from existing ``metrics`` and
``attack_trace`` schemas.  When no session is active, ``trace_span`` is a
near-no-op so normal experiments retain their current behavior and output.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
import itertools
import time
from typing import Any, Dict, Iterator, Optional


TRACE_VERSION = "runtime_trace_v1"
_CURRENT_RECORDER: ContextVar[Optional["RuntimeTraceRecorder"]] = ContextVar(
    "runtime_trace_recorder", default=None
)
_SPAN_STACK: ContextVar[tuple[str, ...]] = ContextVar(
    "runtime_trace_span_stack", default=()
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SpanHandle:
    record: Optional[Dict[str, Any]] = None

    def update(self, **values: Any) -> None:
        if self.record is not None:
            self.record.update(values)


@dataclass
class RuntimeTraceRecorder:
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at_utc: str = field(default_factory=_utc_now)
    started_ns: int = field(default_factory=time.perf_counter_ns)
    spans: list[Dict[str, Any]] = field(default_factory=list)
    _ids: Any = field(default_factory=lambda: itertools.count(1), repr=False)

    @contextmanager
    def span(self, kind: str, **attributes: Any) -> Iterator[SpanHandle]:
        span_id = f"s{next(self._ids):06d}"
        stack = _SPAN_STACK.get()
        start_ns = time.perf_counter_ns()
        record: Dict[str, Any] = {
            "span_id": span_id,
            "parent_span_id": stack[-1] if stack else None,
            "kind": kind,
            "started_at_utc": _utc_now(),
            "start_offset_ms": round((start_ns - self.started_ns) / 1_000_000, 3),
            **attributes,
        }
        token = _SPAN_STACK.set((*stack, span_id))
        handle = SpanHandle(record)
        try:
            yield handle
        except BaseException as exc:
            record.setdefault("success", False)
            record.setdefault("error_type", type(exc).__name__)
            raise
        finally:
            end_ns = time.perf_counter_ns()
            record["duration_ms"] = round((end_ns - start_ns) / 1_000_000, 3)
            record.setdefault("success", True)
            self.spans.append(record)
            _SPAN_STACK.reset(token)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "version": TRACE_VERSION,
            "started_at_utc": self.started_at_utc,
            "elapsed_ms": round(
                (time.perf_counter_ns() - self.started_ns) / 1_000_000, 3
            ),
            "metadata": dict(self.metadata),
            "spans": list(self.spans),
        }


@contextmanager
def runtime_trace_session(
    enabled: bool,
    metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[Optional[RuntimeTraceRecorder]]:
    if not enabled:
        yield None
        return
    recorder = RuntimeTraceRecorder(metadata=dict(metadata or {}))
    recorder_token = _CURRENT_RECORDER.set(recorder)
    stack_token = _SPAN_STACK.set(())
    try:
        yield recorder
    finally:
        _SPAN_STACK.reset(stack_token)
        _CURRENT_RECORDER.reset(recorder_token)


@contextmanager
def trace_span(kind: str, **attributes: Any) -> Iterator[SpanHandle]:
    recorder = _CURRENT_RECORDER.get()
    if recorder is None:
        yield SpanHandle()
        return
    with recorder.span(kind, **attributes) as handle:
        yield handle


def runtime_trace_enabled() -> bool:
    return _CURRENT_RECORDER.get() is not None
