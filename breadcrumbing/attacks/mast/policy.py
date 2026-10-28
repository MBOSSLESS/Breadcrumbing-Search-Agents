"""Structured Planner/Generator calls for MAST-Direct-Search."""

from __future__ import annotations

import time
from typing import Any, Optional

from breadcrumbing.models import unified_client
from breadcrumbing.runtime_trace import trace_span

from .prompts import generator_messages, planner_messages
from .schema import GeneratedAsset, MatchedCaseContext, PlannerDecision


def _usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if isinstance(usage, dict):
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
    else:
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        completion = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    return {"prompt": prompt, "completion": completion, "total": prompt + completion}


class MASTDirectPolicy:
    """Narrow two-stage policy, intentionally independent of AgentAttacker."""

    def __init__(
        self,
        *,
        model_name: str,
        temperature: float,
        max_format_retries: int = 1,
        max_planner_tokens: int = 2000,
        max_generator_tokens: int = 10000,
        client: Any = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_format_retries = max(0, int(max_format_retries))
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
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.max_format_retries + 2):
            started = time.perf_counter()
            record = {
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
                    method="mast",
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
                last_error = exc
                record["error_type"] = type(exc).__name__
            finally:
                record["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)
                call_log.append(record)
        assert last_error is not None
        raise last_error

    def plan(
        self,
        *,
        case: MatchedCaseContext,
        query: str,
        organic_results: list[dict[str, Any]],
        partial_attack_sequence: list[dict[str, Any]],
        previous_assets: list[dict[str, Any]],
        remaining_injections: Optional[int],
        call_log: list[dict[str, Any]],
    ) -> PlannerDecision:
        return self._call(
            phase="planner",
            messages=planner_messages(
                case=case,
                query=query,
                organic_results=organic_results,
                partial_attack_sequence=partial_attack_sequence,
                previous_assets=previous_assets,
                remaining_injections=remaining_injections,
            ),
            response_format=PlannerDecision,
            max_tokens=self.max_planner_tokens,
            call_log=call_log,
        )

    def generate(
        self,
        *,
        case: MatchedCaseContext,
        query: str,
        organic_results: list[dict[str, Any]],
        decision: PlannerDecision,
        previous_assets: list[dict[str, Any]],
        call_log: list[dict[str, Any]],
    ) -> GeneratedAsset:
        return self._call(
            phase="generator",
            messages=generator_messages(
                case=case,
                query=query,
                organic_results=organic_results,
                decision=decision,
                previous_assets=previous_assets,
            ),
            response_format=GeneratedAsset,
            max_tokens=self.max_generator_tokens,
            call_log=call_log,
        )
