"""Generate complete child strategies and extract compact lessons."""

import re
from pathlib import Path

from breadcrumbing.models import unified_client
from .feedback import build_cross_generation_context
from .prompting_candidates import build_candidate_prompt_bundle
from .prompting_lessons import build_lesson_prompt_bundle
from .schemas import ReflectionResponse, LessonReflectionResponse
from .storage import load_json, save_json
from .strategy import (
    _normalize_reflection_payload,
    _normalize_strategy,
    _strategy_behavior_changed,
    _normalize_lesson_payload,
)


def _overflow(message):
    match = re.search(
        r"maximum context length is (\d+) tokens.*?requested about (\d+) tokens",
        message,
        re.I | re.S,
    )
    if not match:
        return None
    limit, requested = map(int, match.groups())
    return {
        "max_tokens": limit,
        "requested_tokens": requested,
        "overflow_tokens": max(0, requested - limit),
    }


def _request(bundle, config, directory, stem):
    save_json(bundle, directory / f"{stem}_input.json")
    response = unified_client.chat_completions_create(
        model_name=config.reflection_model,
        messages=bundle["messages"],
        temperature=config.reflection_temperature,
        max_tokens=config.reflection_max_tokens,
    )
    if not response or not response.choices:
        raise ValueError("Evolution model returned no choices")
    content = response.choices[0].message.content or ""
    if isinstance(content, list):
        content = "\n".join(
            item if isinstance(item, str) else item.get("text", "") for item in content
        )
    (directory / f"{stem}_raw.txt").write_text(content, encoding="utf-8")
    usage = getattr(response, "usage", None)
    save_json(
        {
            key: getattr(usage, key, 0) or 0
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        directory / f"{stem}_usage.json",
    )
    payload, reasoning, _ = unified_client._extract_json_and_think(content)
    return payload, reasoning


def propose(strategy, summary, history, lessons, generation, child, config, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "proposal.json"
    if output.exists():
        return load_json(output)
    shrink = 0
    overflow = {}
    failures = []
    for attempt in range(1, 6):
        bundle = build_candidate_prompt_bundle(
            strategy,
            summary,
            history,
            build_cross_generation_context(history),
            lessons,
            config.enable_lesson_memory,
            config.enable_stage1_gate,
            generation,
            child,
            config.children_per_generation,
            shrink_level=shrink,
            overflow_info=overflow,
        )
        try:
            payload, reasoning = _request(
                bundle, config, output_dir, f"proposal_{attempt:02d}"
            )
            if not isinstance(payload, dict) or set(payload) == {"result"}:
                raise ValueError("Evolution response is not a strategy object")
            normalized = _normalize_reflection_payload(payload, strategy)
            if not normalized["self_reasoning"] and reasoning:
                normalized["self_reasoning"] = reasoning
            result = ReflectionResponse.model_validate(normalized).model_dump()
            required = (
                any(result["analysis"].values()),
                result["case_diagnostics"],
                result["self_reasoning"].strip(),
                result["change_rationale"],
                any(result["strategy_diff"].values()),
            )
            if not all(required):
                raise ValueError(
                    "Evolution response has empty diagnostic or rationale fields"
                )
            candidate = _normalize_strategy(
                result["updated_strategy"], strategy["name"]
            )
            if not _strategy_behavior_changed(strategy, candidate):
                raise ValueError("Evolution response did not change strategy behavior")
            result["updated_strategy"] = candidate
            save_json(result, output)
            return result
        except Exception as exc:
            failures.append(
                {"attempt": attempt, "error": str(exc), "shrink_level": shrink}
            )
            save_json(failures, output_dir / "proposal_failures.json")
            detected = _overflow(str(exc))
            if detected and shrink < 4:
                shrink += 1
                overflow = detected
    raise RuntimeError(
        f"Evolution failed after 5 attempts; see {output_dir / 'proposal_failures.json'}"
    )


def extract_lessons(mode, diff, summary, context, config, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "lessons.json"
    if output.exists():
        return load_json(output)["lessons"]
    bundle = build_lesson_prompt_bundle(
        lesson_mode=mode,
        strategy_diff=diff,
        repeat_summary=summary,
        stage_label="full" if mode == "positive" else "cheap",
        decision_context=context,
    )
    payload, _ = _request(bundle, config, output_dir, "lessons")
    result = LessonReflectionResponse.model_validate(
        _normalize_lesson_payload(payload)
    ).model_dump()
    save_json(result, output)
    return result["lessons"]
