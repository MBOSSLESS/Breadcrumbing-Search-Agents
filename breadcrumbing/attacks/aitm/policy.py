"""One structured AiTM generation per atomic Search event."""

from __future__ import annotations

import json
import time
from typing import Any

from breadcrumbing.models import unified_client
from breadcrumbing.runtime_trace import trace_span

from .prompts import generation_messages
from .schema import (
    MAX_SNIPPET_CHARS,
    MAX_TITLE_CHARS,
    AiTMGeneration,
    AiTMProviderResponse,
    MatchedCaseContext,
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


def _inline(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split()).strip()[:limit].strip()


def _json_generation(text: str) -> AiTMGeneration | None:
    """Parse one complete JSON object, optionally inside a Markdown fence."""
    candidate = str(text or "").strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()
    start = candidate.find("{")
    if start < 0:
        return None
    try:
        payload, end = json.JSONDecoder().raw_decode(candidate[start:])
    except (json.JSONDecodeError, TypeError):
        return None
    if candidate[start + end :].strip() or not isinstance(payload, dict):
        return None
    try:
        return AiTMGeneration.model_validate(payload)
    except Exception:
        return None


def _direct_instruction_generation(text: str, query: str) -> AiTMGeneration:
    """Serialize one provider-generated direct I_t into a deterministic card."""
    # Some OpenRouter Qwen providers return one token per line. That is a
    # transport artifact rather than meaningful page formatting, so direct
    # (non-JSON) instructions are canonicalized to ordinary prose before they
    # become I_t. Structured Markdown instructions remain untouched.
    instruction = " ".join(str(text or "").split()).strip()
    words = instruction.split()
    folded = instruction.casefold()
    refusal_markers = (
        "cannot generate",
        "cannot assist",
        "can't assist",
        "can't help",
        "unable to provide",
        "designed to refuse",
        "violates ethical guidelines",
        "ethical responses instead",
    )
    if len(instruction) < 160 or len(words) < 24:
        raise ValueError("provider returned an underspecified direct instruction")
    if any(marker in folded for marker in refusal_markers):
        raise ValueError("provider returned a refusal instead of an instruction")
    first_line = next(
        (
            line.strip().lstrip("#").strip()
            for line in instruction.splitlines()
            if line.strip()
        ),
        "",
    )
    topic = _inline(query, 140)
    title = _inline(first_line, MAX_TITLE_CHARS)
    if not title or len(title) < 8:
        title = _inline(f"Important Guidance for {topic}", MAX_TITLE_CHARS)
    snippet = _inline(instruction, MAX_SNIPPET_CHARS)
    return AiTMGeneration.model_validate(
        {
            "instruction": instruction,
            "search_result": {
                "title": title,
                "link": "https://independent-guidance.example/notice",
                "snippet": snippet or "Updated authoritative guidance.",
                "date": "",
                "source": "Independent Guidance Office",
            },
        }
    )


def _normalize_provider_response(parsed: Any, query: str) -> tuple[AiTMGeneration, str]:
    if isinstance(parsed, AiTMGeneration):
        return parsed, "structured"
    if not isinstance(parsed, AiTMProviderResponse):
        parsed = AiTMProviderResponse.model_validate(parsed)
    if parsed.instruction is not None and parsed.search_result is not None:
        return (
            AiTMGeneration(
                instruction=parsed.instruction,
                search_result=parsed.search_result,
            ),
            "structured",
        )
    direct = str(parsed.result or "").strip()
    embedded = _json_generation(direct)
    if embedded is not None:
        return embedded, "recovered_json_envelope"
    return _direct_instruction_generation(direct, query), "recovered_direct_instruction"


class AiTMSearchPolicy:
    """Independent one-step policy; it has no persistent chat/session state."""

    def __init__(
        self,
        *,
        model_name: str,
        temperature: float,
        max_format_retries: int = 1,
        max_generation_tokens: int = 10000,
        client: Any = None,
    ):
        self.model_name = model_name
        self.temperature = float(temperature)
        self.max_format_retries = max(0, int(max_format_retries))
        self.max_generation_tokens = max(1, int(max_generation_tokens))
        self.client = client or unified_client

    def generate(
        self,
        *,
        case: MatchedCaseContext,
        query: str,
        organic_results_text: str,
        previous_instruction: str,
        call_log: list[dict[str, Any]],
    ) -> AiTMGeneration:
        messages = generation_messages(
            case=case,
            query=query,
            organic_results_text=organic_results_text,
            previous_instruction=previous_instruction,
        )
        last_error: Exception | None = None
        for attempt in range(1, self.max_format_retries + 2):
            started = time.perf_counter()
            record = {
                "phase": "aitm_generation",
                "model": self.model_name,
                "attempt": attempt,
                "success": False,
                "tokens": {"prompt": 0, "completion": 0, "total": 0},
            }
            try:
                with trace_span(
                    "llm.logical",
                    role="attacker",
                    method="aitm",
                    phase="aitm_generation",
                    format_attempt=attempt,
                ) as logical_span:
                    response = self.client.beta_chat_completions_parse(
                        model_name=self.model_name,
                        messages=messages,
                        response_format=AiTMProviderResponse,
                        temperature=self.temperature,
                        max_tokens=self.max_generation_tokens,
                    )
                    logical_span.update(**_usage(response))
                record["tokens"] = _usage(response)
                parsed = response.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("model returned no parsed response")
                generated, response_mode = _normalize_provider_response(parsed, query)
                record["success"] = True
                record["response_mode"] = response_mode
                return generated
            except Exception as exc:
                last_error = exc
                record["error_type"] = type(exc).__name__
                record["error_message"] = str(exc)[:500]
            finally:
                record["duration_ms"] = round(
                    (time.perf_counter() - started) * 1000,
                    1,
                )
                call_log.append(record)
        assert last_error is not None
        raise last_error
