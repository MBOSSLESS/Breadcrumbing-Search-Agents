"""Structured Planner and per-stage Attacker calls for IH Search."""

from __future__ import annotations

import time
from typing import Any

from breadcrumbing.models import unified_client
from breadcrumbing.runtime_trace import trace_span

from .prompts import (
    attacker_messages,
    planner_messages,
)
from .schema import FixedStrategy, GeneratedAsset, MatchedCaseContext


def _usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if isinstance(usage, dict):
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
    else:
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        completion = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    return {"prompt": prompt, "completion": completion, "total": prompt + completion}


class IntentHijackingSearchPolicy:
    """Minimal policy using schema-constrained Planner and Attacker calls."""

    def __init__(
        self,
        *,
        model_name: str,
        temperature: float = 0.5,
        max_planner_retries: int = 2,
        max_planner_tokens: int = 2500,
        max_generator_tokens: int = 10000,
        client: Any = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_planner_retries = max(0, int(max_planner_retries))
        self.max_planner_tokens = max_planner_tokens
        self.max_generator_tokens = max_generator_tokens
        self.client = client or unified_client

    def _call(
        self,
        *,
        phase: str,
        messages: list[dict[str, str]],
        response_format: type,
        max_tokens: int,
        call_log: list[dict[str, Any]],
        attempt: int,
    ) -> Any:
        started = time.perf_counter()
        record: dict[str, Any] = {
            "phase": phase,
            "model": self.model_name,
            "attempt": attempt,
            "success": False,
            "tokens": {"prompt": 0, "completion": 0, "total": 0},
        }
        try:
            with trace_span(
                "llm.logical",
                role="attacker",
                method="intent_hijacking",
                phase=phase,
                format_attempt=attempt,
            ) as logical_span:
                response = self.client.beta_chat_completions_parse(
                    model_name=self.model_name,
                    messages=messages,
                    response_format=response_format,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )
                logical_span.update(**_usage(response))
            record["tokens"] = _usage(response)
            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise ValueError("model returned no parsed response")
            if not isinstance(parsed, response_format):
                parsed = response_format.model_validate(parsed)
            record["success"] = True
            return parsed
        except Exception as exc:
            record["error_type"] = type(exc).__name__
            record["error_message"] = str(exc)[:500]
            raise
        finally:
            record["duration_ms"] = round(
                (time.perf_counter() - started) * 1000,
                1,
            )
            call_log.append(record)

    def plan(
        self,
        *,
        case: MatchedCaseContext,
        call_log: list[dict[str, Any]],
    ) -> FixedStrategy:
        last_error: Exception | None = None
        for attempt in range(1, self.max_planner_retries + 2):
            try:
                return self._call(
                    phase="planner",
                    messages=planner_messages(case),
                    response_format=FixedStrategy,
                    max_tokens=self.max_planner_tokens,
                    call_log=call_log,
                    attempt=attempt,
                )
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def generate(
        self,
        *,
        case: MatchedCaseContext,
        strategy: FixedStrategy,
        stage: int,
        public_history: list[dict[str, Any]],
        latest_delta: list[dict[str, Any]],
        query: str,
        organic_results_text: str,
        fixed_source_name: str,
        fixed_source_host: str,
        call_log: list[dict[str, Any]],
        retry_feedback: str = "",
        attempt: int = 1,
    ) -> GeneratedAsset:
        return self._call(
            phase=f"attacker_turn_{stage}",
            messages=attacker_messages(
                case=case,
                strategy=strategy,
                stage=stage,
                public_history=public_history,
                latest_delta=latest_delta,
                query=query,
                organic_results_text=organic_results_text,
                fixed_source_name=fixed_source_name,
                fixed_source_host=fixed_source_host,
                retry_feedback=retry_feedback,
            ),
            response_format=GeneratedAsset,
            max_tokens=self.max_generator_tokens,
            call_log=call_log,
            attempt=attempt,
        )
