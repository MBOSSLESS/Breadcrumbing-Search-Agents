"""Event-local structured policy loop for the Evo-Attacker attacker."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from breadcrumbing.models import unified_client
from breadcrumbing.runtime_trace import trace_span

from .prompts import (
    FORMAT_REPAIR,
    decision_instruction,
    retrieval_messages,
    retrieved_information,
)
from .schema import (
    EventDecision,
    FinalEventDecision,
    MatchedCaseContext,
    RetrievalRequest,
)


def _usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if isinstance(usage, dict):
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
    else:
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        completion = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    return {"prompt": prompt, "completion": completion, "total": prompt + completion}


@dataclass(frozen=True)
class PolicyOutcome:
    action: str
    reflection_summary: str
    modification_instruction: str
    retrieval_ids: list[str]
    continue_used: bool


class EvoPolicy:
    """Query/retrieve/reflect policy with strict event-local JSON."""

    def __init__(
        self,
        *,
        model_name: str = "qwen3",
        temperature: float = 0.5,
        max_format_retries: int = 1,
        max_query_tokens: int = 800,
        max_decision_tokens: int = 1800,
        client: Any = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_format_retries = max(0, int(max_format_retries))
        self.max_query_tokens = max_query_tokens
        self.max_decision_tokens = max_decision_tokens
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
        attempt_messages = list(messages)
        for attempt in range(1, self.max_format_retries + 2):
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
                    method="evo_attacker",
                    phase=phase,
                    format_attempt=attempt,
                ) as logical_span:
                    response = self.client.beta_chat_completions_parse(
                        model_name=self.model_name,
                        messages=attempt_messages,
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
                message = response.choices[0].message
                content = str(getattr(message, "content", "") or "")
                record.update(
                    {
                        "parsed_schema": response_format.__name__,
                        "finish_reason": getattr(
                            response.choices[0], "finish_reason", None
                        ),
                        "content_chars": len(content),
                        "reasoning_present": bool(getattr(message, "reasoning", None)),
                    }
                )
                record["success"] = True
                return parsed
            except Exception as exc:
                last_error = exc
                record["error_type"] = type(exc).__name__
                if attempt <= self.max_format_retries:
                    attempt_messages = [
                        *messages,
                        {"role": "user", "content": FORMAT_REPAIR},
                    ]
            finally:
                record["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)
                call_log.append(record)
        assert last_error is not None
        raise last_error

    def run_event(
        self,
        *,
        case: MatchedCaseContext,
        event: dict[str, Any],
        safe_history: list[dict[str, Any]],
        retriever: Any,
        retrieval_k: int,
        call_log: list[dict[str, Any]],
    ) -> PolicyOutcome:
        """Run one event-local session; CONTINUE can occur at most once."""
        messages = retrieval_messages(case=case, event=event, safe_history=safe_history)
        request = self._call(
            phase="retrieval_query",
            messages=messages,
            response_format=RetrievalRequest,
            max_tokens=self.max_query_tokens,
            call_log=call_log,
        )
        memories = retriever.retrieve(request.query, k=retrieval_k)
        retrieval_ids = [item.entry_id for item in memories]
        messages.extend(
            [
                {"role": "assistant", "content": request.model_dump_json()},
                {"role": "user", "content": retrieved_information(memories)},
                {"role": "user", "content": decision_instruction(allow_continue=True)},
            ]
        )
        decision = self._call(
            phase="decision",
            messages=messages,
            response_format=EventDecision,
            max_tokens=self.max_decision_tokens,
            call_log=call_log,
        )
        if decision.action != "CONTINUE":
            return PolicyOutcome(
                action=decision.action,
                reflection_summary=decision.reflection_summary,
                modification_instruction=decision.modification_instruction,
                retrieval_ids=retrieval_ids,
                continue_used=False,
            )

        second = retriever.retrieve(decision.next_retrieval_query, k=retrieval_k)
        retrieval_ids.extend(item.entry_id for item in second)
        messages.extend(
            [
                {"role": "assistant", "content": decision.model_dump_json()},
                {"role": "user", "content": retrieved_information(second)},
                {"role": "user", "content": decision_instruction(allow_continue=False)},
            ]
        )
        final = self._call(
            phase="decision_after_continue",
            messages=messages,
            response_format=FinalEventDecision,
            max_tokens=self.max_decision_tokens,
            call_log=call_log,
        )
        return PolicyOutcome(
            action=final.action,
            reflection_summary=final.reflection_summary,
            modification_instruction=final.modification_instruction,
            retrieval_ids=retrieval_ids,
            continue_used=True,
        )
