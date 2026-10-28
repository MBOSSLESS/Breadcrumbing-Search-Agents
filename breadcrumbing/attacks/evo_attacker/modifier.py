"""Independent structured Modifier for Search cards and Visit pages."""

from __future__ import annotations

import time
from typing import Any, Callable

from breadcrumbing.models import unified_client
from breadcrumbing.runtime_trace import trace_span

from .policy import _usage
from .prompts import FORMAT_REPAIR, page_modifier_messages, search_modifier_messages
from .schema import LandingPage, MatchedCaseContext, SearchCard


class EvoModifier:
    """Independent Modifier with one bounded format retry."""

    def __init__(
        self,
        *,
        model_name: str = "qwen3",
        temperature: float = 0.5,
        max_format_retries: int = 1,
        max_search_tokens: int = 1800,
        max_page_tokens: int = 8000,
        client: Any = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_format_retries = max(0, int(max_format_retries))
        self.max_search_tokens = max_search_tokens
        self.max_page_tokens = max_page_tokens
        self.client = client or unified_client

    def _call(
        self,
        *,
        phase: str,
        messages: list[dict[str, str]],
        response_format: type,
        max_tokens: int,
        call_log: list[dict[str, Any]],
        validator: Callable[[Any], Any] | None = None,
        repair_instruction: str = FORMAT_REPAIR,
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
                if validator is not None:
                    parsed = validator(parsed)
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
                        {"role": "user", "content": repair_instruction},
                    ]
            finally:
                record["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)
                call_log.append(record)
        assert last_error is not None
        raise last_error

    def render_search(
        self,
        *,
        case: MatchedCaseContext,
        event: dict[str, Any],
        instruction: str,
        call_log: list[dict[str, Any]],
        validator: Callable[[SearchCard], SearchCard] | None = None,
    ) -> SearchCard:
        return self._call(
            phase="search_modifier",
            messages=search_modifier_messages(
                case=case, event=event, instruction=instruction
            ),
            response_format=SearchCard,
            max_tokens=self.max_search_tokens,
            call_log=call_log,
            validator=validator,
            repair_instruction=(
                FORMAT_REPAIR
                + " The link must be a newly generated unique HTTPS URL on a controlled "
                "host, with no credentials or fragment, and must not reuse or imitate "
                "any organic or previously registered domain."
            ),
        )

    def render_page(
        self,
        *,
        case: MatchedCaseContext,
        event: dict[str, Any],
        instruction: str,
        call_log: list[dict[str, Any]],
    ) -> LandingPage:
        return self._call(
            phase="page_modifier",
            messages=page_modifier_messages(
                case=case, event=event, instruction=instruction
            ),
            response_format=LandingPage,
            max_tokens=self.max_page_tokens,
            call_log=call_log,
        )
