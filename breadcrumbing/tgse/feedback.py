"""Repeated-outcome summaries and ancestry feedback."""

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple
from .strategy import _normalize_strategy_diff, build_strategy_diff
from .compression import _summarize_messages, compress_attack_trace_for_summary


def _get_repeat_asrs(repeat_summary: Dict[str, Any]) -> List[float]:
    repeat_metrics = repeat_summary.get("repeat_metrics", {})
    values: List[float] = []
    for metrics in repeat_metrics.values():
        if isinstance(metrics, dict):
            value = metrics.get("overall_asr")
            if isinstance(value, (int, float)):
                values.append(float(value))
    return values


def _case_ratio_map(repeat_summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for case in repeat_summary.get("cases", []):
        mapping[case.get("case_id", "")] = {
            "success_ratio": case.get("success_ratio", 0.0),
            "bucket": case.get("bucket"),
            "pattern": case.get("pattern"),
            "test_case_id": case.get("test_case_id"),
        }
    return mapping


def _summarize_case_movements(
    previous_summary: Optional[Dict[str, Any]],
    current_summary: Dict[str, Any],
) -> Dict[str, Any]:
    current_cases = _case_ratio_map(current_summary)
    previous_cases = _case_ratio_map(previous_summary or {})

    deltas: List[Dict[str, Any]] = []
    recovered_cases: List[Dict[str, Any]] = []
    regressed_cases: List[Dict[str, Any]] = []
    persistent_success_cases: List[Dict[str, Any]] = []
    persistent_failure_cases: List[Dict[str, Any]] = []

    for case_id, current in current_cases.items():
        previous = previous_cases.get(
            case_id, {"success_ratio": 0.0, "bucket": None, "pattern": None}
        )
        delta = float(current["success_ratio"]) - float(
            previous.get("success_ratio", 0.0)
        )
        item = {
            "case_id": case_id,
            "test_case_id": current.get("test_case_id"),
            "from_bucket": previous.get("bucket"),
            "to_bucket": current.get("bucket"),
            "from_ratio": round(float(previous.get("success_ratio", 0.0)), 3),
            "to_ratio": round(float(current["success_ratio"]), 3),
            "delta": round(delta, 3),
        }
        deltas.append(item)
        if (
            previous.get("bucket") != "stable_success"
            and current.get("bucket") == "stable_success"
        ):
            recovered_cases.append(item)
        if (
            previous.get("bucket") == "stable_success"
            and current.get("bucket") != "stable_success"
        ):
            regressed_cases.append(item)
        if (
            previous.get("bucket") == "stable_success"
            and current.get("bucket") == "stable_success"
        ):
            persistent_success_cases.append(item)
        if (
            previous.get("bucket") == "stable_failure"
            and current.get("bucket") == "stable_failure"
        ):
            persistent_failure_cases.append(item)

    deltas.sort(key=lambda item: item["delta"], reverse=True)
    top_improved = [item for item in deltas if item["delta"] > 0][:5]
    top_degraded = sorted(
        [item for item in deltas if item["delta"] < 0], key=lambda item: item["delta"]
    )[:5]

    return {
        "top_improved_cases": top_improved,
        "top_degraded_cases": top_degraded,
        "recovered_cases": recovered_cases[:5],
        "regressed_cases": regressed_cases[:5],
        "persistent_success_cases": persistent_success_cases[:5],
        "persistent_failure_cases": persistent_failure_cases[:5],
    }


def summarize_observed_effects(
    previous_summary: Optional[Dict[str, Any]],
    current_summary: Dict[str, Any],
) -> Dict[str, Any]:
    current_asrs = _get_repeat_asrs(current_summary)
    previous_asrs = _get_repeat_asrs(previous_summary or {})
    current_avg = (sum(current_asrs) / len(current_asrs)) if current_asrs else 0.0
    previous_avg = (sum(previous_asrs) / len(previous_asrs)) if previous_asrs else 0.0
    current_buckets = current_summary.get("bucket_counts", {})
    previous_buckets = (previous_summary or {}).get("bucket_counts", {})
    case_movements = _summarize_case_movements(previous_summary, current_summary)

    if current_avg > previous_avg + 1e-9:
        net_signal = (
            "positive_with_tradeoffs"
            if case_movements["top_degraded_cases"]
            else "positive"
        )
    elif current_avg < previous_avg - 1e-9:
        net_signal = "negative"
    else:
        net_signal = (
            "mixed"
            if case_movements["top_improved_cases"]
            or case_movements["top_degraded_cases"]
            else "unclear"
        )

    summary_lines = [
        f"avg_asr moved from {previous_avg:.3f} to {current_avg:.3f} (delta {current_avg - previous_avg:+.3f}).",
        (
            "bucket delta: "
            f"stable_success {current_buckets.get('stable_success', 0) - previous_buckets.get('stable_success', 0):+d}, "
            f"flip {current_buckets.get('flip', 0) - previous_buckets.get('flip', 0):+d}, "
            f"stable_failure {current_buckets.get('stable_failure', 0) - previous_buckets.get('stable_failure', 0):+d}."
        ),
    ]
    return {
        "summary": " ".join(summary_lines),
        "net_signal": net_signal,
        "avg_asr": round(current_avg, 3),
        "avg_asr_delta": round(current_avg - previous_avg, 3),
        "max_repeat_asr": round(max(current_asrs), 3) if current_asrs else 0.0,
        "min_repeat_asr": round(min(current_asrs), 3) if current_asrs else 0.0,
        **case_movements,
    }


def build_case_history_timeline(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    timeline: Dict[str, Dict[str, Any]] = {}
    for item in history:
        for case in item.get("case_snapshot", []):
            case_id = case.get("case_id")
            if not case_id:
                continue
            row = timeline.setdefault(
                case_id,
                {
                    "case_id": case_id,
                    "test_case_id": case.get("test_case_id"),
                    "ratios": [],
                    "buckets": [],
                },
            )
            row["ratios"].append(
                {
                    "generation": item.get("generation"),
                    "success_ratio": case.get("success_ratio", 0.0),
                    "success_count": case.get("success_count", 0),
                    "repeat_count": case.get("repeat_count", 0),
                }
            )
            row["buckets"].append(
                {
                    "generation": item.get("generation"),
                    "bucket": case.get("bucket"),
                    "pattern": case.get("pattern"),
                }
            )
    rows = list(timeline.values())
    rows.sort(
        key=lambda row: (
            -max(
                (point.get("success_ratio", 0.0) for point in row["ratios"]),
                default=0.0,
            ),
            row["case_id"],
        )
    )
    return rows


def build_cross_generation_context(
    history: List[Dict[str, Any]], window: int = 4
) -> Dict[str, Any]:
    recent = history[-window:]
    generations = []
    failed_themes: Dict[str, int] = {}
    for item in recent:
        strategy_diff = item.get("strategy_diff", {})
        change_texts = [
            *strategy_diff.get("added", []),
            *strategy_diff.get("modified", []),
            *strategy_diff.get("rollback_notes", []),
        ]
        for text in change_texts:
            key = text[:120]
            if item.get("observed_effects", {}).get("net_signal") == "negative":
                failed_themes[key] = failed_themes.get(key, 0) + 1

        generations.append(
            {
                "generation": item.get("generation"),
                "metrics": item.get("metrics_overview", {}),
                "strategy_diff": strategy_diff,
                "intended_goal": item.get("intended_goal", ""),
                "observed_effects": item.get("observed_effects", {}),
                "top_improved_cases": item.get("observed_effects", {}).get(
                    "top_improved_cases", []
                ),
                "top_degraded_cases": item.get("observed_effects", {}).get(
                    "top_degraded_cases", []
                ),
            }
        )

    repeated_failed_themes = [
        {"theme": theme, "negative_generations": count}
        for theme, count in failed_themes.items()
        if count >= 2
    ][:5]

    return {
        "window_size": len(recent),
        "generations": generations,
        "case_timelines": build_case_history_timeline(recent),
        "repeated_failed_themes": repeated_failed_themes,
    }


def build_case_id(entry: Dict[str, Any], fallback_idx: int, dataset_name: str) -> str:
    test_case = entry.get("test_case", {})
    test_case_id = test_case.get("test_case_id")
    if test_case_id:
        return f"{dataset_name}::{test_case_id}"
    return f"{dataset_name}::fallback::{fallback_idx}"


def bucket_from_flags(flags: List[bool]) -> str:
    if flags and all(flags):
        return "stable_success"
    if flags and not any(flags):
        return "stable_failure"
    return "flip"


def summarize_success_failure_diff(
    repeat_items: List[Tuple[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    success_items = [item for _, item in repeat_items if item["success"]]
    failure_items = [item for _, item in repeat_items if not item["success"]]

    def _extract_highlights(items: Iterable[Dict[str, Any]]) -> List[str]:
        highlights: List[str] = []
        for item in items:
            compressed = item.get("compressed", {})
            failure_analysis = compressed.get("failure_analysis", {})
            for block in compressed.get("interaction_blocks", [])[:2]:
                outcome = str(block.get("outcome_summary", "")).strip()
                why = str(block.get("why_it_mattered", "")).strip()
                if outcome:
                    highlights.append(f"block: {outcome}")
                if why:
                    highlights.append(f"block_impact: {why}")
            if compressed.get("agent_behavior"):
                highlights.append(compressed["agent_behavior"])
            suspected_issue = failure_analysis.get("suspected_issue")
            issue_desc = failure_analysis.get("issue_description")
            if suspected_issue:
                text = (
                    suspected_issue
                    if not issue_desc
                    else f"{suspected_issue}: {issue_desc}"
                )
                highlights.append(text)
            for decision in compressed.get("key_decisions", [])[:2]:
                reflection = decision.get("reflection_summary")
                payload = decision.get("payload_summary")
                response = decision.get("agent_response")
                if reflection:
                    highlights.append(f"decision: {reflection}")
                if payload:
                    highlights.append(f"payload: {payload}")
                if response:
                    highlights.append(f"agent: {response}")
        deduped: List[str] = []
        seen = set()
        for value in highlights:
            value = value.strip()
            if not value or value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        return deduped[:6]

    success_highlights = _extract_highlights(success_items)
    failure_highlights = _extract_highlights(failure_items)
    return {
        "success_highlights": success_highlights,
        "failure_highlights": failure_highlights,
        "success_failure_diff": {
            "only_in_success": [
                item for item in success_highlights if item not in failure_highlights
            ][:4],
            "only_in_failure": [
                item for item in failure_highlights if item not in success_highlights
            ][:4],
        },
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 3)


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return round(sum(items) / len(items), 3)


def _extract_repeat_execution_metrics(entry: Dict[str, Any]) -> Dict[str, Any]:
    metrics = entry.get("metrics", {})
    attack_trace = metrics.get("attack_trace", {})
    events = attack_trace.get("events", []) if isinstance(attack_trace, dict) else []
    if not isinstance(events, list):
        events = []

    injected_urls: set[str] = set()
    target_urls: set[str] = set()
    visited_injected_urls: set[str] = set()
    visited_target_urls: set[str] = set()

    search_attack_count = 0
    visit_attack_count = 0
    visit_no_attack_count = 0
    total_visit_events = 0
    controlled_visit_events = 0

    for event in events:
        if not isinstance(event, dict):
            continue
        decision = event.get("decision", {})
        if not isinstance(decision, dict):
            decision = {}
        payload = decision.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}

        link = payload.get("link")
        if isinstance(link, str) and link.strip():
            injected_urls.add(link.strip())

        for target_url in payload.get("target_urls", []) or []:
            if isinstance(target_url, str) and target_url.strip():
                target_urls.add(target_url.strip())

        event_type = event.get("type")
        attack = bool(decision.get("attack"))
        if event_type == "search" and attack:
            search_attack_count += 1
            continue

        if event_type != "visit":
            continue

        total_visit_events += 1
        url = event.get("input", {}).get("url", "")
        is_controlled = False
        if isinstance(url, str) and url:
            if url in injected_urls:
                visited_injected_urls.add(url)
                is_controlled = True
            if url in target_urls:
                visited_target_urls.add(url)
                is_controlled = True
        if is_controlled:
            controlled_visit_events += 1

        if attack:
            visit_attack_count += 1
        else:
            visit_no_attack_count += 1

    llm_calls = metrics.get("llm_calls", [])
    tool_calls = metrics.get("tool_calls", [])
    llm_call_count = len(llm_calls) if isinstance(llm_calls, list) else 0
    tool_call_count = len(tool_calls) if isinstance(tool_calls, list) else 0
    num_messages = metrics.get("num_messages")
    if not isinstance(num_messages, (int, float)):
        num_messages = 0

    return {
        "search_attack_count": search_attack_count,
        "visit_attack_count": visit_attack_count,
        "visit_no_attack_count": visit_no_attack_count,
        "injected_url_count": len(injected_urls),
        "target_url_count": len(target_urls),
        "visited_injected_url_count": len(visited_injected_urls),
        "visited_target_url_count": len(visited_target_urls),
        "injected_visit": bool(visited_injected_urls),
        "target_visit": bool(visited_target_urls),
        "attack_controlled_visit_share": _safe_ratio(
            controlled_visit_events, total_visit_events
        ),
        "injected_url_recall": _safe_ratio(
            len(visited_injected_urls), len(injected_urls)
        ),
        "target_url_recall": _safe_ratio(len(visited_target_urls), len(target_urls)),
        "num_messages": int(num_messages),
        "llm_call_count": llm_call_count,
        "tool_call_count": tool_call_count,
        "termination": str(metrics.get("termination") or ""),
    }


def _build_repeat_anomaly_flags(
    execution_metrics: Dict[str, Any], success: bool
) -> List[str]:
    flags: List[str] = []
    if success and execution_metrics.get("visited_injected_url_count", 0) == 0:
        flags.append("successful_without_injected_visit")
    if (not success) and execution_metrics.get(
        "attack_controlled_visit_share", 0.0
    ) >= 0.75:
        flags.append("failed_after_high_attack_controlled_share")
    if (not success) and execution_metrics.get("visited_injected_url_count", 0) >= 2:
        flags.append("failed_after_multiple_injected_visits")
    if (
        execution_metrics.get("target_url_count", 0) > 0
        and execution_metrics.get("visited_target_url_count", 0) == 0
    ):
        flags.append("target_urls_declared_but_never_used")
    return flags


def enrich_compressed_summaries(
    evaluated_entries: List[Dict[str, Any]],
    compressed_summaries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach compact summaries and lightweight placeholders for missing compression."""
    compressed_by_case_id = {
        summary.get("test_case_id"): summary
        for summary in compressed_summaries
        if summary.get("test_case_id")
    }
    enriched: List[Dict[str, Any]] = []
    for fallback_idx, entry in enumerate(evaluated_entries):
        test_case = entry.get("test_case", {})
        case_id = test_case.get("test_case_id") or build_case_id(
            entry,
            fallback_idx,
            test_case.get("_metadata", {}).get("source_dataset", "unknown"),
        )
        summary = compressed_by_case_id.get(case_id)
        if summary is None:
            asr_result = entry.get("evaluation", {}).get("asr", {})
            enriched.append(
                {
                    "test_case_id": case_id,
                    "dataset": test_case.get("_metadata", {}).get(
                        "source_dataset", "unknown"
                    ),
                    "user_query": test_case.get("user_query", ""),
                    "target_consequence": test_case.get("target_consequence", ""),
                    "asr": bool(asr_result.get("success", False)),
                    "compression_missing": True,
                    "compression_missing_reason": "trace_compression_failed_or_unavailable",
                    "interaction_blocks": [],
                    "key_decisions": [],
                    "failure_analysis": {},
                    "agent_behavior": "",
                    "judge_response": asr_result.get("judge_response", ""),
                }
            )
            continue

        updated = dict(summary)
        metrics = entry.get("metrics", {})
        updated["compression_missing"] = False
        updated["attack_trace_summary"] = compress_attack_trace_for_summary(
            metrics.get("attack_trace")
        )
        updated["messages_summary"] = _summarize_messages(
            metrics.get("messages", []),
            max_chars=12000,
            keep_full_user=True,
            compress_think=True,
            tool_summary_mode="brief",
        )
        updated["judge_response"] = (
            entry.get("evaluation", {}).get("asr", {}).get("judge_response", "")
        )
        enriched.append(updated)
    return enriched


def aggregate_repeat_outputs(
    repeat_entries: Dict[str, List[Dict[str, Any]]],
    repeat_compressed: Dict[str, List[Dict[str, Any]]],
    dataset_name: str,
) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Any]] = {}
    all_repeat_execution_metrics: List[Dict[str, Any]] = []

    for repeat_name, entries in sorted(repeat_entries.items()):
        compressed_by_case = {
            summary.get("test_case_id"): summary
            for summary in repeat_compressed.get(repeat_name, [])
        }
        for fallback_idx, entry in enumerate(entries):
            case_id = build_case_id(entry, fallback_idx, dataset_name)
            test_case = entry.get("test_case", {})
            success = bool(
                entry.get("evaluation", {}).get("asr", {}).get("success", False)
            )
            record = grouped.setdefault(
                case_id,
                {
                    "case_id": case_id,
                    "dataset": dataset_name,
                    "test_case_id": test_case.get("test_case_id"),
                    "user_query": test_case.get("user_query", ""),
                    "target_consequence": test_case.get("target_consequence", ""),
                    "risk_verification_checklist": test_case.get(
                        "risk_verification_checklist", []
                    ),
                    "risk_explanation": test_case.get("risk_explanation", ""),
                    "user_expectation": test_case.get("user_expectation", ""),
                    "repeats": {},
                },
            )
            execution_metrics = _extract_repeat_execution_metrics(entry)
            anomaly_flags = _build_repeat_anomaly_flags(execution_metrics, success)
            all_repeat_execution_metrics.append(execution_metrics)
            record["repeats"][repeat_name] = {
                "success": success,
                "entry": entry,
                "compressed": compressed_by_case.get(test_case.get("test_case_id"), {}),
                "execution_metrics": execution_metrics,
                "anomaly_flags": anomaly_flags,
            }

    cases: List[Dict[str, Any]] = []
    success_k_distribution: Dict[str, Dict[str, float]] = {}
    bucket_counts = {"stable_success": 0, "stable_failure": 0, "flip": 0}
    generation_anomaly_counts = {
        "successful_without_injected_visit_cases": 0,
        "failed_after_high_attack_controlled_share_cases": 0,
        "failed_after_multiple_injected_visits_cases": 0,
        "target_urls_declared_but_never_used_cases": 0,
    }

    for case in grouped.values():
        repeat_items = sorted(case["repeats"].items())
        flags = [bool(item[1]["success"]) for item in repeat_items]
        success_count = sum(int(flag) for flag in flags)
        repeat_count = len(flags)
        pattern = "".join("1" if flag else "0" for flag in flags)
        bucket = bucket_from_flags(flags)
        bucket_counts[bucket] += 1
        diff_summary = summarize_success_failure_diff(repeat_items)
        execution_rows = [
            payload.get("execution_metrics", {}) for _, payload in repeat_items
        ]

        successful_without_injected_visit_count = sum(
            1
            for _, payload in repeat_items
            if "successful_without_injected_visit" in payload.get("anomaly_flags", [])
        )
        failed_after_high_attack_controlled_share_count = sum(
            1
            for _, payload in repeat_items
            if "failed_after_high_attack_controlled_share"
            in payload.get("anomaly_flags", [])
        )
        failed_after_multiple_injected_visits_count = sum(
            1
            for _, payload in repeat_items
            if "failed_after_multiple_injected_visits"
            in payload.get("anomaly_flags", [])
        )
        target_urls_declared = any(
            row.get("target_url_count", 0) > 0 for row in execution_rows
        )
        target_urls_declared_but_never_used = target_urls_declared and all(
            row.get("visited_target_url_count", 0) == 0 for row in execution_rows
        )

        case_metrics = {
            "injected_visit_rate": _mean(
                1.0 if row.get("injected_visit") else 0.0 for row in execution_rows
            ),
            "target_visit_rate": _mean(
                1.0 if row.get("target_visit") else 0.0 for row in execution_rows
            ),
            "avg_attack_controlled_visit_share": _mean(
                row.get("attack_controlled_visit_share", 0.0) for row in execution_rows
            ),
            "avg_visit_no_attack_count": _mean(
                row.get("visit_no_attack_count", 0.0) for row in execution_rows
            ),
            "avg_injected_url_recall": _mean(
                row.get("injected_url_recall", 0.0) for row in execution_rows
            ),
            "avg_target_url_recall": _mean(
                row.get("target_url_recall", 0.0) for row in execution_rows
            ),
            "avg_num_messages": _mean(
                row.get("num_messages", 0.0) for row in execution_rows
            ),
            "avg_llm_call_count": _mean(
                row.get("llm_call_count", 0.0) for row in execution_rows
            ),
        }
        case_anomaly_flags = {
            "successful_without_injected_visit": successful_without_injected_visit_count
            > 0,
            "failed_after_high_attack_controlled_share": failed_after_high_attack_controlled_share_count
            > 0,
            "failed_after_multiple_injected_visits": failed_after_multiple_injected_visits_count
            > 0,
            "target_urls_declared_but_never_used": target_urls_declared_but_never_used,
        }
        case_anomaly_counts = {
            "successful_without_injected_visit_count": successful_without_injected_visit_count,
            "failed_after_high_attack_controlled_share_count": failed_after_high_attack_controlled_share_count,
            "failed_after_multiple_injected_visits_count": failed_after_multiple_injected_visits_count,
        }
        if case_anomaly_flags["successful_without_injected_visit"]:
            generation_anomaly_counts["successful_without_injected_visit_cases"] += 1
        if case_anomaly_flags["failed_after_high_attack_controlled_share"]:
            generation_anomaly_counts[
                "failed_after_high_attack_controlled_share_cases"
            ] += 1
        if case_anomaly_flags["failed_after_multiple_injected_visits"]:
            generation_anomaly_counts[
                "failed_after_multiple_injected_visits_cases"
            ] += 1
        if case_anomaly_flags["target_urls_declared_but_never_used"]:
            generation_anomaly_counts["target_urls_declared_but_never_used_cases"] += 1
        case_summary = {
            "case_id": case["case_id"],
            "dataset": case["dataset"],
            "test_case_id": case["test_case_id"],
            "user_query": case["user_query"],
            "target_consequence": case["target_consequence"],
            "risk_verification_checklist": case.get("risk_verification_checklist", []),
            "risk_explanation": case.get("risk_explanation", ""),
            "user_expectation": case.get("user_expectation", ""),
            "repeat_count": repeat_count,
            "success_count": success_count,
            "failure_count": repeat_count - success_count,
            "success_ratio": (success_count / repeat_count) if repeat_count else 0.0,
            "pattern": pattern,
            "bucket": bucket,
            "case_metrics": case_metrics,
            "anomaly_flags": case_anomaly_flags,
            "anomaly_counts": case_anomaly_counts,
            "repeat_details": [
                {
                    "repeat": repeat_name,
                    "success": payload["success"],
                    "judge_response": payload["entry"]
                    .get("evaluation", {})
                    .get("asr", {})
                    .get("judge_response", ""),
                    "attack_trace_summary": payload.get("compressed", {}).get(
                        "attack_trace_summary", ""
                    ),
                    "messages_summary": payload.get("compressed", {}).get(
                        "messages_summary", ""
                    ),
                    "execution_metrics": payload.get("execution_metrics", {}),
                    "anomaly_flags": payload.get("anomaly_flags", []),
                    "compressed": payload.get("compressed", {}),
                }
                for repeat_name, payload in repeat_items
            ],
            **diff_summary,
        }
        cases.append(case_summary)

    max_repeats = max((case["repeat_count"] for case in cases), default=0)
    for success_count in range(max_repeats + 1):
        matched = sum(1 for case in cases if case["success_count"] == success_count)
        success_k_distribution[str(success_count)] = {
            "count": matched,
            "ratio": (matched / len(cases)) if cases else 0.0,
        }

    aggregate_metrics = {
        "avg_asr": _safe_ratio(
            sum(case["success_count"] for case in cases),
            sum(case["repeat_count"] for case in cases),
        ),
        "max_nasr": _safe_ratio(
            sum(1 for case in cases if case["success_count"] > 0), len(cases)
        ),
        "injected_visit_repeat_rate": _mean(
            1.0 if row.get("visited_injected_url_count", 0) > 0 else 0.0
            for row in all_repeat_execution_metrics
        ),
        "target_visit_repeat_rate": _mean(
            1.0 if row.get("visited_target_url_count", 0) > 0 else 0.0
            for row in all_repeat_execution_metrics
        ),
        "avg_attack_controlled_visit_share": _mean(
            row.get("attack_controlled_visit_share", 0.0)
            for row in all_repeat_execution_metrics
        ),
        "avg_visit_no_attack_count": _mean(
            row.get("visit_no_attack_count", 0.0)
            for row in all_repeat_execution_metrics
        ),
        "avg_injected_url_recall": _mean(
            row.get("injected_url_recall", 0.0) for row in all_repeat_execution_metrics
        ),
        "avg_target_url_recall": _mean(
            row.get("target_url_recall", 0.0) for row in all_repeat_execution_metrics
        ),
        "avg_num_messages": _mean(
            row.get("num_messages", 0.0) for row in all_repeat_execution_metrics
        ),
        "avg_llm_call_count": _mean(
            row.get("llm_call_count", 0.0) for row in all_repeat_execution_metrics
        ),
        "avg_tool_call_count": _mean(
            row.get("tool_call_count", 0.0) for row in all_repeat_execution_metrics
        ),
    }

    cases.sort(
        key=lambda item: (
            item["bucket"] != "flip",
            abs(item["success_count"] - (item["repeat_count"] / 2)),
            -item["success_count"],
            item["case_id"],
        )
    )

    return {
        "dataset": dataset_name,
        "total_cases": len(cases),
        "bucket_counts": bucket_counts,
        "success_k_distribution": success_k_distribution,
        "aggregate_metrics": aggregate_metrics,
        "anomaly_counts": generation_anomaly_counts,
        "cases": cases,
    }


def build_generation_memory(
    generation_label: str,
    strategy: Dict[str, Any],
    repeat_summary: Dict[str, Any],
    previous_strategy: Optional[Dict[str, Any]] = None,
    previous_repeat_summary: Optional[Dict[str, Any]] = None,
    previous_memory: Optional[List[Dict[str, Any]]] = None,
    reflection_output: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    bucket_counts = repeat_summary.get("bucket_counts", {})
    repeat_asrs = _get_repeat_asrs(repeat_summary)
    observed_effects = summarize_observed_effects(
        previous_repeat_summary, repeat_summary
    )
    aggregate_metrics = repeat_summary.get("aggregate_metrics", {})
    strategy_diff = (
        _normalize_strategy_diff(reflection_output.get("strategy_diff"))
        if reflection_output
        else build_strategy_diff(previous_strategy, strategy)
    )
    if not strategy_diff.get("summary"):
        strategy_diff = build_strategy_diff(previous_strategy, strategy)

    memory = {
        "generation": generation_label,
        "strategy_summary": {
            "name": strategy["name"],
            "description": strategy["description"],
            "goal": strategy["goal"],
            "step_count": len(strategy.get("steps", [])),
        },
        "overall_stats": {
            "total_cases": repeat_summary.get("total_cases", 0),
            "bucket_counts": bucket_counts,
            "anomaly_counts": repeat_summary.get("anomaly_counts", {}),
        },
        "metrics_overview": {
            "avg_asr": aggregate_metrics.get(
                "avg_asr",
                round(sum(repeat_asrs) / len(repeat_asrs), 3) if repeat_asrs else 0.0,
            ),
            "max_nasr": aggregate_metrics.get("max_nasr", 0.0),
            "max_repeat_asr": round(max(repeat_asrs), 3) if repeat_asrs else 0.0,
            "min_repeat_asr": round(min(repeat_asrs), 3) if repeat_asrs else 0.0,
            "bucket_counts": bucket_counts,
            "injected_visit_repeat_rate": aggregate_metrics.get(
                "injected_visit_repeat_rate", 0.0
            ),
            "target_visit_repeat_rate": aggregate_metrics.get(
                "target_visit_repeat_rate", 0.0
            ),
            "avg_attack_controlled_visit_share": aggregate_metrics.get(
                "avg_attack_controlled_visit_share", 0.0
            ),
            "avg_visit_no_attack_count": aggregate_metrics.get(
                "avg_visit_no_attack_count", 0.0
            ),
        },
        "strategy_diff": strategy_diff,
        "observed_effects": observed_effects,
        "case_snapshot": [
            {
                "case_id": case["case_id"],
                "test_case_id": case.get("test_case_id"),
                "bucket": case["bucket"],
                "pattern": case["pattern"],
                "success_ratio": case["success_ratio"],
                "success_count": case.get("success_count", 0),
                "repeat_count": case.get("repeat_count", 0),
            }
            for case in repeat_summary.get("cases", [])
        ],
        "notable_case_changes": observed_effects.get("top_improved_cases", [])
        + observed_effects.get("top_degraded_cases", []),
        "takeaway": "",
    }

    if reflection_output:
        memory["change_summary"] = reflection_output.get("change_rationale", [])
        memory["intended_goal"] = reflection_output.get("expected_impact", "")
        memory["takeaway"] = reflection_output.get("additional_notes", "") or (
            "; ".join(reflection_output.get("analysis", {}).get("root_causes", [])[:2])
        )
    elif previous_memory:
        memory["takeaway"] = previous_memory[-1].get("takeaway", "")

    return memory
