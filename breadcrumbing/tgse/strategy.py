"""Strategy contracts and model-response normalization."""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional


def validate_strategy_semantics(strategy: Dict[str, Any]) -> List[str]:
    """Return semantic validation errors for a single strategy."""
    errors: List[str] = []

    for field in ["name", "description", "goal"]:
        value = strategy.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"Field '{field}' must be a non-empty string")

    steps = strategy.get("steps")
    if not isinstance(steps, list):
        errors.append("Field 'steps' must be a list")
        return errors

    if len(steps) < 2:
        errors.append("Strategy must contain at least 2 steps")

    step_types: List[str] = []
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"step[{idx}] must be a dict")
            continue
        keys = list(step.keys())
        if len(keys) != 1 or keys[0] not in {"search", "visit"}:
            errors.append(
                f"step[{idx}] must contain exactly one of 'search' or 'visit'"
            )
            continue
        step_type = keys[0]
        step_value = step.get(step_type)
        if not isinstance(step_value, str) or not step_value.strip():
            errors.append(f"step[{idx}].{step_type} must be a non-empty string")
            continue
        step_types.append(step_type)

    if step_types and step_types[0] != "search":
        errors.append("The first step must be a 'search' step")
    if "search" not in step_types:
        errors.append("Strategy must contain at least one 'search' step")
    if "visit" not in step_types:
        errors.append("Strategy must contain at least one 'visit' step")

    for field in ["usage_conditions", "avoid_conditions"]:
        value = strategy.get(field, [])
        if not isinstance(value, list):
            errors.append(f"Field '{field}' must be a list")
            continue
        for idx, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                errors.append(f"{field}[{idx}] must be a non-empty string")

    return errors


def _normalize_strategy(strategy: Dict[str, Any], expected_name: str) -> Dict[str, Any]:
    required_fields = ["name", "description", "goal", "steps"]
    missing = [field for field in required_fields if not strategy.get(field)]
    if missing:
        raise ValueError(
            f"Strategy '{expected_name}' missing required fields: {missing}"
        )

    steps = strategy.get("steps", [])
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"Strategy '{expected_name}' step[{idx}] must be a dict")
        keys = set(step.keys())
        if keys not in ({"search"}, {"visit"}):
            raise ValueError(
                f"Strategy '{expected_name}' step[{idx}] must contain exactly one of search/visit"
            )

    normalized = dict(strategy)
    normalized["usage_conditions"] = list(strategy.get("usage_conditions", []))
    normalized["avoid_conditions"] = list(strategy.get("avoid_conditions", []))
    semantic_errors = validate_strategy_semantics(normalized)
    if semantic_errors:
        raise ValueError(
            f"Strategy '{expected_name}' failed semantic validation: "
            + "; ".join(semantic_errors)
        )
    return normalized


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ]
        return "\n".join(parts)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _coerce_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        result = []
        for item in value:
            text = _coerce_text(item)
            if text:
                result.append(text)
        return result
    text = _coerce_text(value)
    return [text] if text else []


def _normalize_strategy_steps(value: Any) -> List[Dict[str, str]]:
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []

    normalized_steps: List[Dict[str, str]] = []
    for step in value:
        if not isinstance(step, dict):
            continue
        if (
            "search" in step
            and isinstance(step.get("search"), str)
            and step["search"].strip()
        ):
            normalized_steps.append({"search": step["search"].strip()})
        if (
            "visit" in step
            and isinstance(step.get("visit"), str)
            and step["visit"].strip()
        ):
            normalized_steps.append({"visit": step["visit"].strip()})
    return normalized_steps


def _strategy_behavior_signature(strategy: Dict[str, Any]) -> str:
    behavior = {
        "steps": _normalize_strategy_steps(strategy.get("steps")),
        "usage_conditions": sorted(
            _coerce_string_list(strategy.get("usage_conditions"))
        ),
        "avoid_conditions": sorted(
            _coerce_string_list(strategy.get("avoid_conditions"))
        ),
    }
    return json.dumps(behavior, ensure_ascii=False, sort_keys=True)


def _strategy_behavior_changed(
    previous_strategy: Dict[str, Any], new_strategy: Dict[str, Any]
) -> bool:
    return _strategy_behavior_signature(
        previous_strategy
    ) != _strategy_behavior_signature(new_strategy)


def _normalize_case_diagnostics(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []

    normalized_items: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            normalized_items.append(
                {
                    "case_id": "",
                    "bucket": None,
                    "pattern": None,
                    "diagnosis": item.strip(),
                    "lessons": [],
                }
            )
            continue
        if not isinstance(item, dict):
            continue
        normalized_items.append(
            {
                "case_id": _coerce_text(item.get("case_id")),
                "bucket": _coerce_text(item.get("bucket")) or None,
                "pattern": _coerce_text(item.get("pattern")) or None,
                "diagnosis": _coerce_text(item.get("diagnosis")),
                "lessons": _coerce_string_list(item.get("lessons")),
            }
        )
    return normalized_items


def _normalize_strategy_diff(value: Any) -> Dict[str, Any]:
    if value is None:
        value = {}
    if isinstance(value, str):
        value = {"summary": value}
    if not isinstance(value, dict):
        value = {}
    return {
        "summary": _coerce_text(value.get("summary")),
        "added": _coerce_string_list(value.get("added")),
        "removed": _coerce_string_list(value.get("removed")),
        "modified": _coerce_string_list(value.get("modified")),
        "preserved": _coerce_string_list(value.get("preserved")),
        "rollback_notes": _coerce_string_list(value.get("rollback_notes")),
    }


def _normalize_reflection_payload(
    payload: Dict[str, Any], fallback_strategy: Dict[str, Any]
) -> Dict[str, Any]:
    normalized = dict(payload) if isinstance(payload, dict) else {}

    analysis = normalized.get("analysis")
    if isinstance(analysis, str):
        analysis = {
            "successful_patterns": [],
            "failure_patterns": [],
            "root_causes": [analysis.strip()] if analysis.strip() else [],
            "opportunities": [],
        }
    elif not isinstance(analysis, dict):
        analysis = {}
    normalized["analysis"] = {
        "successful_patterns": _coerce_string_list(analysis.get("successful_patterns")),
        "failure_patterns": _coerce_string_list(analysis.get("failure_patterns")),
        "root_causes": _coerce_string_list(analysis.get("root_causes")),
        "opportunities": _coerce_string_list(analysis.get("opportunities")),
    }

    normalized["self_reasoning"] = _coerce_text(normalized.get("self_reasoning"))
    normalized["expected_impact"] = _coerce_text(normalized.get("expected_impact"))
    normalized["additional_notes"] = _coerce_text(normalized.get("additional_notes"))
    normalized["change_rationale"] = _coerce_string_list(
        normalized.get("change_rationale")
    )
    normalized["case_diagnostics"] = _normalize_case_diagnostics(
        normalized.get("case_diagnostics")
    )
    normalized["strategy_diff"] = _normalize_strategy_diff(
        normalized.get("strategy_diff")
    )

    strategy = normalized.get("updated_strategy")
    if not isinstance(strategy, dict):
        strategy = dict(fallback_strategy)
    normalized["updated_strategy"] = {
        "name": _coerce_text(strategy.get("name")) or fallback_strategy["name"],
        "description": _coerce_text(strategy.get("description"))
        or fallback_strategy["description"],
        "goal": _coerce_text(strategy.get("goal")) or fallback_strategy["goal"],
        "steps": _normalize_strategy_steps(strategy.get("steps"))
        or list(fallback_strategy.get("steps", [])),
        "usage_conditions": _coerce_string_list(strategy.get("usage_conditions")),
        "avoid_conditions": _coerce_string_list(strategy.get("avoid_conditions")),
    }
    return normalized


def _strategy_step_text(step: Dict[str, Any]) -> str:
    if "search" in step:
        return f"search: {step.get('search', '').strip()}"
    if "visit" in step:
        return f"visit: {step.get('visit', '').strip()}"
    return json.dumps(step, ensure_ascii=False, sort_keys=True)


def build_strategy_diff(
    previous_strategy: Optional[Dict[str, Any]], new_strategy: Dict[str, Any]
) -> Dict[str, Any]:
    if not previous_strategy:
        return {
            "summary": "Initial strategy snapshot with no previous generation to compare against.",
            "added": [],
            "removed": [],
            "modified": [],
            "preserved": [
                "Initial generation establishes the baseline strategy structure, goals, and staged guidance."
            ],
            "rollback_notes": [],
        }

    added: List[str] = []
    removed: List[str] = []
    modified: List[str] = []
    preserved: List[str] = []

    for field in ["description", "goal"]:
        old_value = _coerce_text(previous_strategy.get(field))
        new_value = _coerce_text(new_strategy.get(field))
        if old_value == new_value and new_value:
            preserved.append(f"{field} kept the same overall intent.")
        elif old_value and not new_value:
            removed.append(f"{field} content was dropped from the strategy.")
        elif not old_value and new_value:
            added.append(f"{field} was newly introduced: {new_value[:200]}")
        else:
            modified.append(
                f"{field} changed from '{old_value[:120]}' to '{new_value[:120]}'."
            )

    old_steps = previous_strategy.get("steps", [])
    new_steps = new_strategy.get("steps", [])
    max_steps = max(len(old_steps), len(new_steps))
    for idx in range(max_steps):
        old_step = old_steps[idx] if idx < len(old_steps) else None
        new_step = new_steps[idx] if idx < len(new_steps) else None
        step_label = f"step_{idx + 1}"
        if old_step is None and new_step is not None:
            added.append(f"{step_label} added: {_strategy_step_text(new_step)[:240]}")
        elif new_step is None and old_step is not None:
            removed.append(
                f"{step_label} removed: {_strategy_step_text(old_step)[:240]}"
            )
        elif old_step == new_step and new_step is not None:
            preserved.append(f"{step_label} preserved its stage guidance.")
        elif old_step is not None and new_step is not None:
            modified.append(
                f"{step_label} revised from '{_strategy_step_text(old_step)[:140]}' "
                f"to '{_strategy_step_text(new_step)[:140]}'."
            )

    for field in ["usage_conditions", "avoid_conditions"]:
        old_items = set(previous_strategy.get(field, []))
        new_items = set(new_strategy.get(field, []))
        for item in sorted(new_items - old_items):
            added.append(f"{field} added '{item}'.")
        for item in sorted(old_items - new_items):
            removed.append(f"{field} removed '{item}'.")
        for item in sorted(old_items & new_items):
            preserved.append(f"{field} preserved '{item}'.")

    summary_parts = []
    if modified:
        summary_parts.append(f"{len(modified)} modified")
    if added:
        summary_parts.append(f"{len(added)} added")
    if removed:
        summary_parts.append(f"{len(removed)} removed")
    if preserved and not summary_parts:
        summary_parts.append("no material structural change")

    return {
        "summary": "Strategy diff: " + ", ".join(summary_parts) + ".",
        "added": added[:8],
        "removed": removed[:8],
        "modified": modified[:8],
        "preserved": preserved[:8],
        "rollback_notes": [],
    }


def _normalize_lesson_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload) if isinstance(payload, dict) else {}
    summary = _coerce_text(normalized.get("summary"))
    lessons = _coerce_string_list(normalized.get("lessons"))
    case_links = _coerce_string_list(normalized.get("case_links"))
    if not summary and lessons:
        summary = lessons[0]
    return {
        "summary": summary,
        "lessons": lessons,
        "case_links": case_links,
    }
