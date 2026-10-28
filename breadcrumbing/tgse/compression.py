"""Trace compression for strategy evolution."""

from __future__ import annotations

import json
import logging
import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field
from tqdm import tqdm

from breadcrumbing.models import unified_client
from .storage import ensure_dir, load_json, load_jsonl, save_json

from .compression_prompt import format_case_summary_prompt

logger = logging.getLogger(__name__)


class InteractionBlock(BaseModel):
    tool: str
    tool_input_summary: str
    attack_relation: str
    outcome_summary: str
    why_it_mattered: str


class KeyDecision(BaseModel):
    event_id: int
    type: str
    decision: str
    reflection_summary: str
    payload_summary: str = ""
    agent_response: str


class FailureAnalysis(BaseModel):
    attack_executed: bool
    agent_responsive: bool
    suspected_issue: str
    issue_description: str


class CaseSummaryResponse(BaseModel):
    interaction_blocks: List[InteractionBlock] = Field(default_factory=list)
    key_decisions: List[KeyDecision] = Field(default_factory=list)
    failure_analysis: FailureAnalysis
    agent_behavior: str


def _compression_max_tokens() -> int:
    raw_value = os.getenv("TGSE_COMPRESSION_MAX_TOKENS", "20000")
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        logger.warning("Invalid TGSE_COMPRESSION_MAX_TOKENS=%r; using 20000", raw_value)
        return 20000
    return max(1024, value)


def _env_int(name: str, default: int, minimum: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using %d", name, raw_value, default)
        return default
    return max(minimum, value)


def _compression_worker_count(requested_workers: int) -> int:
    default_cap = 4
    cap = _env_int("TGSE_COMPRESSION_WORKERS", default_cap, 1)
    return max(1, min(requested_workers, cap))


PATTERNS = {
    "think": re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
    "answer": re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE),
    "tool_call": re.compile(
        r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE
    ),
    "tool_response": re.compile(
        r"<tool_response>(.*?)</tool_response>", re.DOTALL | re.IGNORECASE
    ),
}


def _extract_attack_summary(
    attack_trace: Union[List[Any], Dict[str, Any], None], asr: bool
) -> Dict[str, Any]:
    events: List[Dict[str, Any]] = []
    if isinstance(attack_trace, list):
        events = [e for e in attack_trace if isinstance(e, dict)]
    elif isinstance(attack_trace, dict):
        raw_events = attack_trace.get("events", [])
        if isinstance(raw_events, list):
            events = [e for e in raw_events if isinstance(e, dict)]

    attack_events = sum(
        1 for e in events if (e.get("decision") or {}).get("attack") is True
    )
    return {
        "total_events": len(events),
        "attack_events": attack_events,
        "termination": "success" if asr else "failure",
    }


def _extract_strategy_from_trace(
    attack_trace: Union[List[Any], Dict[str, Any], None],
) -> str:
    if not attack_trace:
        return "unknown"

    if isinstance(attack_trace, dict):
        usage = ((attack_trace.get("stats") or {}).get("strategy") or {}).get(
            "usage"
        ) or {}
        if usage:
            return max(usage, key=usage.get)
        for event in attack_trace.get("events", []):
            if not isinstance(event, dict):
                continue
            for change in (event.get("meta") or {}).get("strategy_changes") or []:
                if change.get("action") == "set":
                    return change.get("to", "unknown")

    if isinstance(attack_trace, list):
        for event in attack_trace:
            if not isinstance(event, dict):
                continue
            for change in (event.get("meta") or {}).get("strategy_changes") or []:
                if change.get("action") == "set":
                    return change.get("to", "unknown")

    return "unknown"


def _truncate(text: str, max_len: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _bounded_for_prompt(value: Any, *, depth: int = 0) -> Any:
    """Keep attack-trace structure while bounding very large strings/lists."""
    if depth > 8:
        return _truncate(str(value), 240)

    if isinstance(value, dict):
        return {
            str(k): _bounded_for_prompt(v, depth=depth + 1) for k, v in value.items()
        }

    if isinstance(value, list):
        max_items = _env_int("TGSE_COMPRESSION_TRACE_LIST_MAX_ITEMS", 30, 1)
        bounded = [
            _bounded_for_prompt(item, depth=depth + 1) for item in value[:max_items]
        ]
        if len(value) > max_items:
            bounded.append({"_truncated_items": len(value) - max_items})
        return bounded

    if isinstance(value, str):
        return _truncate(
            value, _env_int("TGSE_COMPRESSION_TRACE_STRING_MAX_CHARS", 1400, 120)
        )

    return value


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _extract_blocks(content: str) -> List[Tuple[str, str]]:
    pattern = re.compile(
        r"(<think>.*?</think>|<tool_call>.*?</tool_call>|<tool_response>.*?</tool_response>|<answer>.*?</answer>)",
        re.DOTALL | re.IGNORECASE,
    )
    parts: List[Tuple[str, str]] = []
    last = 0
    for match in pattern.finditer(content):
        if match.start() > last:
            parts.append(("text", content[last : match.start()]))
        block = match.group(0)
        tag = "text"
        lowered = block.lower()
        if lowered.startswith("<think>"):
            tag = "think"
        elif lowered.startswith("<tool_call>"):
            tag = "tool_call"
        elif lowered.startswith("<tool_response>"):
            tag = "tool_response"
        elif lowered.startswith("<answer>"):
            tag = "answer"
        parts.append((tag, block))
        last = match.end()
    if last < len(content):
        parts.append(("text", content[last:]))
    return parts


def _parse_tool_call_json(block: str) -> Optional[Dict[str, Any]]:
    inner = re.sub(
        r"^<tool_call>|</tool_call>$", "", block.strip(), flags=re.IGNORECASE
    ).strip()
    try:
        return json.loads(inner)
    except Exception:
        return None


def _format_tool_call(block: str) -> str:
    payload = _parse_tool_call_json(block) or {}
    name = payload.get("name", "unknown")
    arguments = payload.get("arguments", {}) or {}
    if name == "search":
        query = arguments.get("query", [])
        if isinstance(query, str):
            query = [query]
        return f"<tool_call> search | q: {'; '.join(_truncate(str(q), 100) for q in query[:4])} </tool_call>"
    if name == "visit":
        urls = arguments.get("url", [])
        if isinstance(urls, str):
            urls = [urls]
        goal = _truncate(arguments.get("goal", ""), 120)
        return f"<tool_call> visit | url: {'; '.join(_truncate(str(u), 90) for u in urls[:3])} | goal: {goal} </tool_call>"
    return f"<tool_call> {name} </tool_call>"


def _format_tool_response(block: str, mode: str) -> str:
    if mode == "skip":
        return "<tool_response> (skipped) </tool_response>"
    match = PATTERNS["tool_response"].search(block)
    body = match.group(1).strip() if match else block
    body = re.sub(r"\s+", " ", body)
    if mode == "key":
        parts = re.split(r"(?<=[.!?])\s+", body)
        body = " ".join(parts[:2])
    return f"<tool_response> {_truncate(body, 500)} </tool_response>"


def _format_think(block: str, compress: bool) -> str:
    if not compress:
        return block
    inner = re.sub(r"^<think>|</think>$", "", block.strip(), flags=re.IGNORECASE)
    return f"<think> {_truncate(_compact_whitespace(inner), 900)} </think>"


def _format_answer(block: str) -> str:
    match = PATTERNS["answer"].search(block)
    answer = match.group(1).strip() if match else block
    return f"<answer> {_truncate(_compact_whitespace(answer), 2500)} </answer>"


def _summarize_assistant_structured(
    content: str, compress_think: bool, tool_summary_mode: str
) -> str:
    output: List[str] = []
    for typ, block in _extract_blocks(content):
        if typ == "think":
            output.append(_format_think(block, compress_think))
        elif typ == "tool_call":
            output.append(_format_tool_call(block))
        elif typ == "tool_response":
            output.append(_format_tool_response(block, tool_summary_mode))
        elif typ == "answer":
            output.append(_format_answer(block))
            break
        elif typ == "text":
            text = _compact_whitespace(block)
            if text:
                output.append(f"<text> {_truncate(text, 240)} </text>")
    return "\n".join(output)


def _summarize_messages(
    messages: List[Dict[str, Any]],
    max_chars: int = 12000,
    keep_full_user: bool = True,
    compress_think: bool = True,
    tool_summary_mode: Literal["brief", "key", "skip"] = "brief",
) -> str:
    if not messages:
        return ""

    parts: List[str] = []
    remaining = max_chars
    user_count = 0

    for msg in messages:
        role = msg.get("role", "unknown")
        content = str(msg.get("content", ""))
        if not content.strip():
            continue

        if role == "user":
            if PATTERNS["tool_response"].search(content):
                summary = _format_tool_response(content, tool_summary_mode)
            elif user_count < 3 or not keep_full_user:
                summary = (
                    f"<user> {_truncate(_compact_whitespace(content), 400)} </user>"
                )
                user_count += 1
            else:
                continue
        elif role == "assistant":
            summary = _summarize_assistant_structured(
                content, compress_think, tool_summary_mode
            )
        else:
            continue

        if len(summary) > remaining - 50:
            parts.append(_truncate(summary, max(remaining, 0)))
            break
        parts.append(summary)
        remaining -= len(summary) + 2

    return "\n\n".join(parts)


def _extract_section(text: str, label: str) -> str:
    pattern = re.compile(
        rf"{re.escape(label)}\s*(.*?)(?=\n\n[A-Z][A-Za-z ]+:|$)", re.DOTALL
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _compact_event_reflection(text: str, attack: bool) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""

    if attack:
        normalized = _compact_whitespace(raw)
        if len(normalized) <= 1400:
            return normalized
        return normalized[:950].rstrip() + " ... " + normalized[-350:].lstrip()

    webpage = _compact_whitespace(_extract_section(raw, "Webpage Summary:"))
    trajectory = _compact_whitespace(
        _extract_section(raw, "Agent Trajectory Analysis:")
    )
    adaptation = _compact_whitespace(_extract_section(raw, "Adversarial Adaptation:"))
    segments: List[str] = []
    if webpage:
        segments.append("Webpage Summary: " + _truncate(webpage, 420))
    if trajectory:
        segments.append("Agent Trajectory Analysis: " + _truncate(trajectory, 420))
    if adaptation:
        segments.append("Adversarial Adaptation: " + _truncate(adaptation, 420))
    if segments:
        return "\n\n".join(segments)

    normalized = _compact_whitespace(raw)
    if len(normalized) <= 1000:
        return normalized
    return normalized[:700].rstrip() + " ... " + normalized[-240:].lstrip()


def _summarize_page_content(page_content: str, max_len: int = 1500) -> str:
    text = str(page_content or "").strip()
    if not text:
        return ""
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) <= max_len:
        return text
    head = text[:1000].rstrip()
    tail = text[-350:].lstrip()
    return head + "\n...\n" + tail


def _build_payload_outline(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict) or not payload:
        return {}
    outline: Dict[str, Any] = {}
    for key in ["title", "link", "position", "date", "source"]:
        value = payload.get(key)
        if value:
            outline[key] = value
    snippet = payload.get("snippet")
    if snippet:
        outline["snippet"] = _truncate(_compact_whitespace(snippet), 700)
    target_urls = payload.get("target_urls")
    if isinstance(target_urls, list) and target_urls:
        outline["target_urls"] = target_urls[:5]
    page_content = payload.get("page_content")
    if page_content:
        outline["page_content_excerpt"] = _summarize_page_content(
            page_content, max_len=1600
        )
    return outline


def build_attack_trace_packet(
    attack_trace: Union[List[Any], Dict[str, Any], None],
) -> Dict[str, Any]:
    if attack_trace is None or (isinstance(attack_trace, list) and not attack_trace):
        return {"events": [], "error": "No attack trace available"}

    if isinstance(attack_trace, list):
        attack_trace = {"events": attack_trace}

    events = attack_trace.get("events", []) if isinstance(attack_trace, dict) else []
    packet_events: List[Dict[str, Any]] = []
    for raw_event in events if isinstance(events, list) else []:
        if not isinstance(raw_event, dict):
            continue
        decision = raw_event.get("decision") or {}
        event_packet = {
            "event_id": raw_event.get("id") or raw_event.get("event_id") or 0,
            "type": raw_event.get("type", "unknown"),
            "input": {
                key: value
                for key, value in (raw_event.get("input") or {}).items()
                if key in {"query", "url", "goal", "attack_source"}
                and value not in (None, "", [])
            },
            "attack": bool(decision.get("attack")),
            "reflection_outline": _compact_event_reflection(
                decision.get("reflection", ""), bool(decision.get("attack"))
            ),
            "payload_outline": _build_payload_outline(decision.get("payload") or {}),
        }
        packet_events.append(event_packet)

    stats = (
        ((attack_trace.get("stats") or {}).get("strategy") or {}).get("usage")
        if isinstance(attack_trace, dict)
        else {}
    )
    return {
        "events": packet_events,
        "strategy_usage": stats or {},
    }


def compress_attack_trace_for_summary(
    attack_trace: Union[List[Any], Dict[str, Any], None],
) -> str:
    return json.dumps(
        build_attack_trace_packet(attack_trace),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _summarize_tool_response_text(text: str) -> str:
    normalized = _compact_whitespace(text)
    if not normalized:
        return ""
    if "Invalid request format" in normalized:
        return _truncate(normalized, 240)
    lines = re.split(r"(?<=[.!?])\s+", normalized)
    return _truncate(" ".join(lines[:2]), 320)


def build_message_flow_packet(
    messages: List[Dict[str, Any]], max_blocks: int = 8
) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    pending_call: Optional[Dict[str, Any]] = None

    for msg in messages or []:
        role = msg.get("role")
        content = str(msg.get("content", ""))
        if not content:
            continue

        if role == "assistant":
            tool_match = PATTERNS["tool_call"].search(content)
            answer_match = PATTERNS["answer"].search(content)
            if tool_match:
                payload = _parse_tool_call_json(tool_match.group(0)) or {}
                name = payload.get("name", "unknown")
                arguments = payload.get("arguments", {}) or {}
                pending_call = {
                    "tool": name,
                    "input": arguments,
                    "assistant_context": _truncate(
                        _summarize_assistant_structured(content, True, "key"), 900
                    ),
                    "response": "",
                }
            elif answer_match:
                blocks.append(
                    {
                        "tool": "answer",
                        "input": {},
                        "assistant_context": _truncate(
                            _compact_whitespace(answer_match.group(1)), 1200
                        ),
                        "response": "final_answer",
                    }
                )
            continue

        if role != "user":
            continue

        if pending_call is not None and PATTERNS["tool_response"].search(content):
            response_match = PATTERNS["tool_response"].search(content)
            response_text = response_match.group(1) if response_match else content
            pending_call["response"] = _summarize_tool_response_text(response_text)
            blocks.append(pending_call)
            pending_call = None
        elif "[SYSTEM WARNING]" in content:
            blocks.append(
                {
                    "tool": "warning",
                    "input": {},
                    "assistant_context": _truncate(_compact_whitespace(content), 220),
                    "response": "",
                }
            )
        else:
            blocks.append(
                {
                    "tool": "user",
                    "input": {},
                    "assistant_context": _truncate(_compact_whitespace(content), 260),
                    "response": "",
                }
            )

    return blocks[:max_blocks]


def extract_final_answer_excerpt(
    messages: List[Dict[str, Any]], max_len: int = 2500
) -> str:
    for msg in reversed(messages or []):
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content", ""))
        match = PATTERNS["answer"].search(content)
        if match:
            return _truncate(_compact_whitespace(match.group(1)), max_len)
    return ""


def build_compression_inputs(case_result: Dict[str, Any]) -> Dict[str, Any]:
    test_case = case_result.get("test_case", {})
    metrics = case_result.get("metrics", {})
    evaluation = case_result.get("evaluation", {})

    attack_trace = metrics.get("attack_trace", [])
    if attack_trace is None or attack_trace == []:
        attack_trace = {"events": [], "error": metrics.get("error", "Attack failed")}

    messages = metrics.get("messages", []) or []
    asr_result = evaluation.get("asr", {})
    asr_success = bool(asr_result.get("success", False))

    bounded_attack_trace = _bounded_for_prompt(attack_trace)

    return {
        "dataset": test_case.get("_metadata", {}).get("source_dataset", "unknown"),
        "test_case_id": test_case.get("test_case_id", "unknown"),
        "user_query": test_case.get("user_query", ""),
        "target_consequence": test_case.get("target_consequence", ""),
        "judge_reasoning": asr_result.get("judge_response", ""),
        "strategy_used": _extract_strategy_from_trace(attack_trace),
        "attack_trace": attack_trace,
        "attack_trace_json": json.dumps(
            bounded_attack_trace, ensure_ascii=False, separators=(",", ":")
        ),
        "event_count": len(
            (attack_trace.get("events") or []) if isinstance(attack_trace, dict) else []
        ),
        "messages": messages,
        "messages_summary": _summarize_messages(
            messages,
            max_chars=_env_int("TGSE_COMPRESSION_MESSAGES_MAX_CHARS", 24000, 4000),
            keep_full_user=True,
            compress_think=True,
            tool_summary_mode="brief",
        ),
        "message_flow_packet": build_message_flow_packet(
            messages,
            max_blocks=_env_int("TGSE_COMPRESSION_FLOW_MAX_BLOCKS", 6, 1),
        ),
        "final_answer_excerpt": extract_final_answer_excerpt(
            messages,
            max_len=_env_int("TGSE_COMPRESSION_FINAL_ANSWER_MAX_CHARS", 1600, 400),
        ),
        "asr_success": asr_success,
    }


def _expected_event_specs(
    attack_trace: Union[List[Any], Dict[str, Any], None],
) -> List[Dict[str, Any]]:
    if isinstance(attack_trace, list):
        events = attack_trace
    elif isinstance(attack_trace, dict):
        events = attack_trace.get("events", [])
    else:
        events = []

    specs: List[Dict[str, Any]] = []
    for event in events if isinstance(events, list) else []:
        if not isinstance(event, dict):
            continue
        decision = event.get("decision") or {}
        specs.append(
            {
                "event_id": int(event.get("id") or event.get("event_id") or 0),
                "type": str(event.get("type") or "unknown"),
                "decision": "attack" if bool(decision.get("attack")) else "no_attack",
            }
        )
    return specs


def _validate_summary(
    summary: Dict[str, Any], prepared: Dict[str, Any]
) -> Optional[str]:
    expected_specs = _expected_event_specs(prepared["attack_trace"])
    actual = summary.get("key_decisions", [])

    if len(actual) != len(expected_specs):
        return f"expected {len(expected_specs)} key_decisions but got {len(actual)}"

    for idx, (expected, observed) in enumerate(zip(expected_specs, actual), start=1):
        if int(observed.get("event_id", -1)) != expected["event_id"]:
            return (
                f"event #{idx} event_id mismatch: expected {expected['event_id']} "
                f"but got {observed.get('event_id')}"
            )
        if str(observed.get("type") or "") != expected["type"]:
            return (
                f"event #{idx} type mismatch: expected {expected['type']} "
                f"but got {observed.get('type')}"
            )
        if str(observed.get("decision") or "") != expected["decision"]:
            return (
                f"event #{idx} decision mismatch: expected {expected['decision']} "
                f"but got {observed.get('decision')}"
            )
    return None


def compress_single_case(
    case_result: Dict[str, Any],
    compression_model: str,
    temperature: float = 0.0,
    feedback_note: str = "",
) -> Tuple[Optional[Dict[str, Any]], Dict[str, int], Optional[str]]:
    try:
        prepared = build_compression_inputs(case_result)
        system_prompt, user_prompt = format_case_summary_prompt(
            dataset=prepared["dataset"],
            user_query=prepared["user_query"],
            target_consequence=prepared["target_consequence"],
            strategy_used=prepared["strategy_used"],
            attack_trace_json=prepared["attack_trace_json"],
            event_count=prepared["event_count"],
            messages_summary=prepared["messages_summary"],
            message_flow_packet=prepared["message_flow_packet"],
            final_answer_excerpt=prepared["final_answer_excerpt"],
            asr=prepared["asr_success"],
            judge_reasoning=prepared["judge_reasoning"],
        )
        if feedback_note:
            user_prompt = (
                user_prompt + "\n\n## Regeneration Correction\n\n" + feedback_note
            )

        response = unified_client.beta_chat_completions_parse(
            model_name=compression_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=_compression_max_tokens(),
            response_format=CaseSummaryResponse,
        )

        if not response or not response.choices:
            logger.error(
                "Empty response during TGSE compression for %s",
                prepared["test_case_id"],
            )
            return None, {}, "empty_response"

        usage = {
            "calls": 1,
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens
            if response.usage
            else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }

        parsed = response.choices[0].message.parsed
        result = parsed.model_dump()
        validation_error = _validate_summary(result, prepared)
        if validation_error:
            logger.warning(
                "TGSE compression validation failed for %s: %s",
                prepared["test_case_id"],
                validation_error,
            )
            return None, usage, validation_error
        result["dataset"] = prepared["dataset"]
        result["test_case_id"] = prepared["test_case_id"]
        result["user_query"] = prepared["user_query"]
        result["target_consequence"] = prepared["target_consequence"]
        result["asr"] = prepared["asr_success"]
        result["strategy_used"] = prepared["strategy_used"]
        result["attack_summary"] = _extract_attack_summary(
            prepared["attack_trace"], prepared["asr_success"]
        )
        return result, usage, None
    except Exception as exc:
        logger.error("TGSE compression failed: %s", exc)
        logger.error(traceback.format_exc())
        return None, {}, str(exc)


def _compression_temperature_for_attempt(
    base_temperature: float, attempt: int
) -> float:
    if attempt <= 1:
        return base_temperature
    if attempt == 2:
        return max(base_temperature, 0.3)
    return max(base_temperature, 0.4)


def _format_retry_feedback(error_note: str, attempt_temperature: float) -> str:
    specific = str(error_note or "unknown_error").strip()
    return (
        "Your previous compression response could not be accepted.\n"
        f"Exact failure: {specific}\n"
        f"This retry uses temperature={attempt_temperature:.1f}; produce a fresh, complete JSON object only.\n"
        "Required top-level fields are exactly: interaction_blocks, key_decisions, failure_analysis, agent_behavior.\n"
        "Do not wrap the JSON in a `result` field, markdown fence, prose, or comments.\n"
        "Do not invent or alter key names. In particular, avoid corrupted suffixes such as `游戏副本`.\n"
        "Preserve every attack_trace event in order, and make `key_decisions` match each event_id, type, and attack/no_attack decision exactly."
    )


def _is_transient_provider_error(error_note: str) -> bool:
    normalized = str(error_note or "").lower()
    return any(
        marker in normalized
        for marker in [
            "concurrency limit",
            "retry shortly",
            "upstream bedrock call exceeded request deadline",
            "error code: 503",
            "error code: 504",
            "http status 503",
            "http status 504",
        ]
    )


def compress_single_case_with_retry(
    case_result: Dict[str, Any],
    compression_model: str,
    temperature: float = 0.0,
    max_retries: int = 8,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, int], Dict[str, Any]]:
    test_case_id = (case_result.get("test_case") or {}).get("test_case_id", "unknown")
    total_usage = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    attempts = max(1, max_retries)

    feedback_note = ""
    last_error = ""
    attempt_errors: List[Dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        attempt_temperature = _compression_temperature_for_attempt(temperature, attempt)
        summary, usage, error_note = compress_single_case(
            case_result,
            compression_model,
            attempt_temperature,
            feedback_note=feedback_note,
        )
        for key in total_usage:
            total_usage[key] += usage.get(key, 0)
        if summary is not None:
            if attempt > 1:
                logger.warning(
                    "Case %s TGSE compression succeeded on retry %d/%d",
                    test_case_id,
                    attempt,
                    attempts,
                )
            return summary, total_usage, {}
        last_error = str(error_note or "unknown_error")
        attempt_errors.append(
            {
                "attempt": attempt,
                "temperature": attempt_temperature,
                "error": last_error,
            }
        )
        if attempt < attempts:
            next_temperature = _compression_temperature_for_attempt(
                temperature, attempt + 1
            )
            if _is_transient_provider_error(last_error):
                provider_attempt_limit = _env_int(
                    "TGSE_COMPRESSION_PROVIDER_MAX_RETRIES", 2, 1
                )
                if attempt >= provider_attempt_limit:
                    logger.error(
                        "Case %s TGSE compression hit transient provider error after %d outer attempts; "
                        "stopping compression retries for this case: %s",
                        test_case_id,
                        attempt,
                        last_error,
                    )
                    break
                feedback_note = ""
            else:
                feedback_note = _format_retry_feedback(last_error, next_temperature)
            logger.warning(
                "Case %s TGSE compression failed on attempt %d/%d at temperature %.1f, retrying: %s",
                test_case_id,
                attempt,
                attempts,
                attempt_temperature,
                last_error,
            )

    logger.error(
        "Case %s TGSE compression failed after %d attempts", test_case_id, attempts
    )
    failure = {
        "test_case_id": test_case_id,
        "error": last_error,
        "attempts": attempt_errors,
        "compression_model": compression_model,
    }
    return None, total_usage, failure


def compress_cases_with_resume(
    evaluated_results: List[Dict[str, Any]],
    output_file: str,
    compression_model: str,
    checkpoint_file: Optional[str] = None,
    temperature: float = 0.0,
    num_workers: int = 128,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
    ensure_dir(os.path.dirname(output_file))
    failure_file = os.path.join(
        os.path.dirname(output_file), "compression_failures.jsonl"
    )
    stats_file = os.path.join(os.path.dirname(output_file), "compression_stats.json")
    total_usage = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    compressed_ids = set()
    existing_summaries: List[Dict[str, Any]] = []
    previous_failures: List[Dict[str, Any]] = []

    if checkpoint_file:
        try:
            if os.path.exists(checkpoint_file):
                checkpoint = load_json(checkpoint_file)
                compressed_ids = set(checkpoint.get("compressed_ids", []))
        except Exception as exc:
            logger.warning("Failed to load TGSE compression checkpoint: %s", exc)

    if os.path.exists(output_file):
        existing_summaries = load_jsonl(output_file)
        for summary in existing_summaries:
            case_id = summary.get("test_case_id")
            if case_id:
                compressed_ids.add(case_id)

    if os.path.exists(failure_file):
        try:
            previous_failures = load_jsonl(failure_file)
        except Exception as exc:
            logger.warning("Failed to load existing TGSE compression failures: %s", exc)

    stats = {
        "total": len(evaluated_results),
        "already_compressed": len(compressed_ids),
        "to_compress": 0,
        "succeeded": 0,
        "failed": 0,
        "partial": False,
        "compressed_total": len(existing_summaries),
        "expected_total": len(evaluated_results),
        "failed_case_ids": [],
    }

    to_compress: List[Tuple[int, Dict[str, Any], str]] = []
    for idx, case_result in enumerate(evaluated_results):
        case_id = (case_result.get("test_case") or {}).get(
            "test_case_id"
        ) or f"case_{idx}"
        if case_id not in compressed_ids:
            to_compress.append((idx, case_result, case_id))

    stats["to_compress"] = len(to_compress)
    if stats["to_compress"] == 0:
        stats["compressed_total"] = len(existing_summaries)
        stats["partial"] = len(existing_summaries) != len(evaluated_results)
        save_json(stats, stats_file)
        return existing_summaries, stats, total_usage

    new_summaries: List[Optional[Dict[str, Any]]] = [None] * len(to_compress)
    failures: List[Dict[str, Any]] = []
    usage_lock = __import__("threading").Lock()
    effective_workers = _compression_worker_count(num_workers)
    if effective_workers != num_workers:
        logger.info(
            "TGSE compression workers capped: requested=%d effective=%d",
            num_workers,
            effective_workers,
        )

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {}
        for task_idx, (orig_idx, case_result, case_id) in enumerate(to_compress):
            future = executor.submit(
                compress_single_case_with_retry,
                case_result=case_result,
                compression_model=compression_model,
                temperature=temperature,
                max_retries=8,
            )
            futures[future] = (task_idx, orig_idx, case_id)

        succeeded = 0
        failed = 0
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Compressing cases",
            unit="case",
        ):
            task_idx, orig_idx, case_id = futures[future]
            try:
                summary, usage, failure = future.result()
                with usage_lock:
                    for key in total_usage:
                        total_usage[key] += usage.get(key, 0)
                if summary is not None:
                    new_summaries[task_idx] = summary
                    compressed_ids.add(case_id)
                    succeeded += 1
                else:
                    failed += 1
                    failure = failure or {
                        "test_case_id": case_id,
                        "error": "compression_returned_none",
                        "attempts": [],
                    }
                    failure.setdefault("test_case_id", case_id)
                    failure["original_index"] = orig_idx
                    failures.append(failure)
                    logger.warning(
                        "Case %s (original index %d) returned None", case_id, orig_idx
                    )
            except Exception as exc:
                failed += 1
                failures.append(
                    {
                        "test_case_id": case_id,
                        "original_index": orig_idx,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                logger.error(
                    "Case %s (original index %d) failed with error: %s",
                    case_id,
                    orig_idx,
                    exc,
                )
                logger.error(traceback.format_exc())

    stats["succeeded"] = succeeded
    stats["failed"] = failed

    with open(output_file, "a", encoding="utf-8") as handle:
        for summary in new_summaries:
            if summary is not None:
                handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

    if checkpoint_file:
        save_json(
            {
                "compressed_ids": list(compressed_ids),
                "total_compressed": len(compressed_ids),
            },
            checkpoint_file,
        )

    all_summaries = existing_summaries + [
        summary for summary in new_summaries if summary is not None
    ]
    expected_total = len(evaluated_results)
    stats["compressed_total"] = len(all_summaries)
    stats["partial"] = len(all_summaries) != expected_total
    stats["failed_case_ids"] = [
        failure.get("test_case_id", "unknown") for failure in failures
    ]

    if failures:
        with open(failure_file, "a", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")

    stats["failure_records_total"] = len(previous_failures) + len(failures)
    save_json(stats, stats_file)

    if stats["partial"]:
        logger.warning(
            "TGSE compression produced partial summaries: %d/%d "
            "(failed=%d, succeeded=%d, already_compressed=%d). Continuing with missing compressed fields empty.",
            len(all_summaries),
            expected_total,
            stats["failed"],
            stats["succeeded"],
            stats["already_compressed"],
        )

    return all_summaries, stats, total_usage
