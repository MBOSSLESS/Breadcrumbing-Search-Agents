"""Training subset selection and staged promotion rules."""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import hashlib
from .config import TGSEConfig
from .feedback import _get_repeat_asrs

LOW_BAND_THRESHOLD = 0.3
HIGH_BAND_THRESHOLD = 0.7


def _compute_repeat_metrics_from_cases(
    cases: List[Dict[str, Any]],
    dataset_name: str,
    strategy_name: str,
) -> Dict[str, Dict[str, Any]]:
    repeat_names = sorted(
        {
            detail.get("repeat")
            for case in cases
            for detail in case.get("repeat_details", [])
            if detail.get("repeat")
        }
    )
    metrics: Dict[str, Dict[str, Any]] = {}
    for repeat_name in repeat_names:
        successes = 0
        total = 0
        for case in cases:
            for detail in case.get("repeat_details", []):
                if detail.get("repeat") == repeat_name:
                    successes += int(bool(detail.get("success")))
                    total += 1
        overall_asr = (successes / total) if total else 0.0
        metrics[repeat_name] = {
            "total_cases": total,
            "total_successes": successes,
            "overall_asr": overall_asr,
            "asr_by_dataset": {
                dataset_name: {
                    "asr": overall_asr,
                    "total": total,
                    "success": successes,
                }
            },
            "asr_by_strategy": {
                strategy_name: {
                    "asr": overall_asr,
                    "total": total,
                    "success": successes,
                }
            },
        }
    return metrics


def _success_rate(case: Dict[str, Any]) -> float:
    repeat_count = int(case.get("repeat_count", 0) or 0)
    success_count = int(case.get("success_count", 0) or 0)
    return (success_count / repeat_count) if repeat_count else 0.0


def _band_from_rate(success_rate: float) -> str:
    if success_rate >= HIGH_BAND_THRESHOLD:
        return "high"
    if success_rate >= LOW_BAND_THRESHOLD:
        return "mid"
    return "low"


def _with_band_metadata(repeat_summary: Dict[str, Any]) -> Dict[str, Any]:
    summary = dict(repeat_summary)
    cases_with_band: List[Dict[str, Any]] = []
    band_counts = {"low": 0, "mid": 0, "high": 0}
    for case in repeat_summary.get("cases", []):
        item = dict(case)
        rate = _success_rate(item)
        item["success_rate"] = round(rate, 4)
        item["band"] = _band_from_rate(rate)
        band_counts[item["band"]] += 1
        cases_with_band.append(item)
    summary["cases"] = cases_with_band
    summary["band_counts"] = band_counts
    summary["band_thresholds"] = {
        "low_lt": LOW_BAND_THRESHOLD,
        "high_ge": HIGH_BAND_THRESHOLD,
    }
    return summary


def _subset_repeat_summary(
    repeat_summary: Dict[str, Any],
    selected_test_case_ids: List[str],
    strategy_name: str,
) -> Dict[str, Any]:
    selected = set(selected_test_case_ids)
    cases = [
        case
        for case in repeat_summary.get("cases", [])
        if case.get("test_case_id") in selected
    ]
    bucket_counts = {"stable_success": 0, "stable_failure": 0, "flip": 0}
    for case in cases:
        bucket_counts[case.get("bucket", "flip")] += 1

    max_repeats = max((case.get("repeat_count", 0) for case in cases), default=0)
    success_k_distribution: Dict[str, Dict[str, float]] = {}
    for success_count in range(max_repeats + 1):
        matched = sum(
            1 for case in cases if case.get("success_count", 0) == success_count
        )
        success_k_distribution[str(success_count)] = {
            "count": matched,
            "ratio": (matched / len(cases)) if cases else 0.0,
        }

    summary = {
        "dataset": repeat_summary.get("dataset"),
        "total_cases": len(cases),
        "bucket_counts": bucket_counts,
        "success_k_distribution": success_k_distribution,
        "cases": cases,
    }
    summary["repeat_metrics"] = _compute_repeat_metrics_from_cases(
        cases,
        repeat_summary.get("dataset", "Advertisement-1"),
        strategy_name,
    )
    return _with_band_metadata(summary)


def _stage_average_asr(repeat_summary: Dict[str, Any]) -> float:
    values = _get_repeat_asrs(repeat_summary)
    return (sum(values) / len(values)) if values else 0.0


def _build_stage1_metrics(
    child_stage1_summary: Dict[str, Any],
    parent_stage1_summary: Dict[str, Any],
) -> Dict[str, Any]:
    parent_cases = {
        case.get("test_case_id"): case
        for case in parent_stage1_summary.get("cases", [])
        if case.get("test_case_id")
    }
    child_cases = {
        case.get("test_case_id"): case
        for case in child_stage1_summary.get("cases", [])
        if case.get("test_case_id")
    }

    per_case: List[Dict[str, Any]] = []
    worse_count = 0
    same_count = 0
    improved_count = 0
    high_drop_count = 0
    severe_drop_count = 0
    mid_to_high_count = 0
    low_to_mid_count = 0
    band_transition_counts: Dict[str, int] = {}

    for test_case_id in sorted(parent_cases):
        if test_case_id not in child_cases:
            continue
        parent_case = parent_cases[test_case_id]
        child_case = child_cases[test_case_id]
        old_rate = float(parent_case.get("success_rate", _success_rate(parent_case)))
        new_rate = float(child_case.get("success_rate", _success_rate(child_case)))
        old_band = parent_case.get("band") or _band_from_rate(old_rate)
        new_band = child_case.get("band") or _band_from_rate(new_rate)
        delta = new_rate - old_rate

        if delta < 0:
            worse_count += 1
        elif delta > 0:
            improved_count += 1
        else:
            same_count += 1

        if old_band == "high" and new_band != "high":
            high_drop_count += 1
        if old_band in {"mid", "high"} and new_band == "low":
            severe_drop_count += 1
        if old_band == "mid" and new_band == "high":
            mid_to_high_count += 1
        if old_band == "low" and new_band in {"mid", "high"}:
            low_to_mid_count += 1

        transition_key = f"{old_band}->{new_band}"
        band_transition_counts[transition_key] = (
            band_transition_counts.get(transition_key, 0) + 1
        )
        per_case.append(
            {
                "test_case_id": test_case_id,
                "old_success_rate": round(old_rate, 4),
                "new_success_rate": round(new_rate, 4),
                "delta": round(delta, 4),
                "old_band": old_band,
                "new_band": new_band,
                "old_bucket": parent_case.get("bucket"),
                "new_bucket": child_case.get("bucket"),
            }
        )

    total_cases = len(per_case)
    mean_delta = (
        (sum(item["delta"] for item in per_case) / total_cases) if total_cases else 0.0
    )
    worse_rate = (worse_count / total_cases) if total_cases else 0.0
    return {
        "total_cases": total_cases,
        "mean_delta": round(mean_delta, 4),
        "worse_count": worse_count,
        "worse_rate": round(worse_rate, 4),
        "same_count": same_count,
        "improved_count": improved_count,
        "high_drop_count": high_drop_count,
        "severe_drop_count": severe_drop_count,
        "mid_to_high_count": mid_to_high_count,
        "low_to_mid_count": low_to_mid_count,
        "band_transition_counts": band_transition_counts,
        "per_case": per_case,
    }


def _select_stage1_cases(
    parent_summary: Dict[str, Any],
    full_cases: List[Dict[str, Any]],
    config: TGSEConfig,
    selection_salt: str = "",
) -> List[Dict[str, Any]]:
    full_case_by_id = {
        case.get("test_case_id"): case
        for case in full_cases
        if case.get("test_case_id")
    }
    bucket_to_ids: Dict[str, List[str]] = {
        "flip": [],
        "stable_success": [],
        "stable_failure": [],
    }
    ordered_ids: List[str] = []
    for case in parent_summary.get("cases", []):
        test_case_id = case.get("test_case_id")
        if not test_case_id or test_case_id not in full_case_by_id:
            continue
        ordered_ids.append(test_case_id)
        bucket_to_ids.setdefault(case.get("bucket", "flip"), []).append(test_case_id)

    if (
        config.stage1_selection_mode == "dynamic_bucket"
        and config.cheap_cases_per_dataset == 10
    ):
        if bucket_to_ids.get("stable_failure"):
            quota = {"stable_success": 2, "flip": 6, "stable_failure": 2}
        else:
            quota = {"stable_success": 3, "flip": 7, "stable_failure": 0}
    else:
        quota = {
            "flip": config.cheap_flip_cases,
            "stable_success": config.cheap_stable_success_cases,
            "stable_failure": config.cheap_stable_failure_cases,
        }

    case_lookup = {
        case.get("test_case_id"): case
        for case in parent_summary.get("cases", [])
        if case.get("test_case_id")
    }

    def _stable_key(test_case_id: str, namespace: str) -> str:
        raw = f"{config.sampling_seed}:{selection_salt}:{namespace}:{test_case_id}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _ordered_sample(ids: List[str], namespace: str) -> List[str]:
        return sorted(
            ids, key=lambda test_case_id: _stable_key(test_case_id, namespace)
        )

    def _success_rate_for_id(test_case_id: str) -> float:
        case = case_lookup.get(test_case_id, {})
        repeat_count = int(case.get("repeat_count", 0) or 0)
        success_count = int(case.get("success_count", 0) or 0)
        return (success_count / repeat_count) if repeat_count else 0.0

    def _take_flip(limit: int) -> List[str]:
        flip_ids = list(bucket_to_ids.get("flip", []))
        if limit <= 0 or not flip_ids:
            return []
        high = [
            test_case_id
            for test_case_id in flip_ids
            if _success_rate_for_id(test_case_id) >= 0.7
        ]
        mid = [
            test_case_id
            for test_case_id in flip_ids
            if 0.3 <= _success_rate_for_id(test_case_id) < 0.7
        ]
        low = [
            test_case_id
            for test_case_id in flip_ids
            if _success_rate_for_id(test_case_id) < 0.3
        ]
        if limit >= 7:
            target = {"high": 2, "mid": 3, "low": limit - 5}
        elif limit == 6:
            target = {"high": 2, "mid": 2, "low": 2}
        else:
            target = {
                "high": max(1, round(limit * 0.3)),
                "mid": max(1, round(limit * 0.4)),
                "low": max(
                    0, limit - max(1, round(limit * 0.3)) - max(1, round(limit * 0.4))
                ),
            }
        groups = {"high": high, "mid": mid, "low": low}
        selected: List[str] = []
        for group_name in ["high", "mid", "low"]:
            for test_case_id in _ordered_sample(
                groups[group_name], f"flip:{group_name}"
            ):
                if (
                    len([item for item in selected if item in groups[group_name]])
                    >= target[group_name]
                ):
                    break
                if test_case_id not in selected:
                    selected.append(test_case_id)
        for test_case_id in _ordered_sample(flip_ids, "flip:backfill"):
            if len(selected) >= limit:
                break
            if test_case_id not in selected:
                selected.append(test_case_id)
        return selected[:limit]

    selected_ids: List[str] = []

    def _take(bucket: str, limit: int) -> None:
        if bucket == "flip":
            candidates = _take_flip(limit)
        else:
            candidates = _ordered_sample(bucket_to_ids.get(bucket, []), bucket)
        for test_case_id in candidates:
            if test_case_id in selected_ids:
                continue
            selected_ids.append(test_case_id)
            if (
                len([x for x in selected_ids if x in bucket_to_ids.get(bucket, [])])
                >= limit
            ):
                break

    _take("stable_success", quota.get("stable_success", 0))
    _take("flip", quota.get("flip", 0))
    _take("stable_failure", quota.get("stable_failure", 0))

    for test_case_id in ordered_ids:
        if len(selected_ids) >= config.cheap_cases_per_dataset:
            break
        if test_case_id in selected_ids:
            continue
        selected_ids.append(test_case_id)

    return [
        full_case_by_id[test_case_id]
        for test_case_id in selected_ids[: config.cheap_cases_per_dataset]
    ]


def _build_stage1_decision(
    child_stage1_summary: Dict[str, Any],
    parent_stage1_summary: Dict[str, Any],
    config: TGSEConfig,
) -> Dict[str, Any]:
    child_avg = _stage_average_asr(child_stage1_summary)
    parent_avg = _stage_average_asr(parent_stage1_summary)
    metrics = _build_stage1_metrics(child_stage1_summary, parent_stage1_summary)

    reasons: List[str] = []
    avg_drop = parent_avg - child_avg
    if metrics["mean_delta"] < config.stage1_min_mean_delta:
        reasons.append(
            f"mean_delta={metrics['mean_delta']:.3f} fell below hard-regression threshold {config.stage1_min_mean_delta:.3f}"
        )
    if (
        metrics["mean_delta"] < 0
        and metrics["worse_rate"] >= config.stage1_max_worse_rate
    ):
        reasons.append(
            f"negative mean_delta with worse_rate={metrics['worse_rate']:.3f} reached hard-regression threshold {config.stage1_max_worse_rate:.3f}"
        )
    if (
        metrics["mean_delta"] < 0
        and metrics["severe_drop_count"] >= config.stage1_reject_severe_drop_count
    ):
        reasons.append(
            f"negative mean_delta with severe_drop_count={metrics['severe_drop_count']} reached hard-regression threshold {config.stage1_reject_severe_drop_count}"
        )

    return {
        "pass_stage1": not reasons,
        "parent_stage1_avg_asr": round(parent_avg, 3),
        "child_stage1_avg_asr": round(child_avg, 3),
        "avg_drop": round(avg_drop, 3),
        "high_drop_count": metrics["high_drop_count"],
        "high_drop_threshold": config.stage1_max_high_drop_count,
        "high_drop_is_diagnostic_only": True,
        "worse_count": metrics["worse_count"],
        "worse_rate": metrics["worse_rate"],
        "mean_delta": metrics["mean_delta"],
        "reject_reasons": reasons,
        "recorded_only_metrics": {
            "same_count": metrics["same_count"],
            "improved_count": metrics["improved_count"],
            "severe_drop_count": metrics["severe_drop_count"],
            "mid_to_high_count": metrics["mid_to_high_count"],
            "low_to_mid_count": metrics["low_to_mid_count"],
            "band_transition_counts": metrics["band_transition_counts"],
        },
    }


def _stage1_rank_key(
    child_result: Dict[str, Any],
) -> Tuple[float, float, int, int, str]:
    decision = child_result.get("stage1_decision", {})
    metrics = child_result.get("stage1_metrics", {})
    mean_delta = float(
        decision.get("mean_delta", metrics.get("mean_delta", 0.0)) or 0.0
    )
    worse_rate = float(
        decision.get("worse_rate", metrics.get("worse_rate", 1.0)) or 0.0
    )
    severe_drop_count = int(
        decision.get(
            "recorded_only_metrics",
            {},
        ).get("severe_drop_count", metrics.get("severe_drop_count", 0))
        or 0
    )
    high_drop_count = int(
        decision.get("high_drop_count", metrics.get("high_drop_count", 0)) or 0
    )
    run_id = str(child_result.get("run_id", ""))
    return (mean_delta, -worse_rate, -severe_drop_count, -high_drop_count, run_id)


def _select_children_for_full(
    child_results: List[Dict[str, Any]],
    config: TGSEConfig,
) -> Dict[str, Any]:
    pending = [item for item in child_results if item.get("status") == "stage1_done"]
    passable = [
        item
        for item in pending
        if item.get("stage1_decision", {}).get("pass_stage1", False)
    ]
    rejected = [
        item
        for item in pending
        if not item.get("stage1_decision", {}).get("pass_stage1", False)
    ]
    selected = sorted(passable, key=_stage1_rank_key, reverse=True)[
        : max(0, config.full_children_per_generation)
    ]

    if len(selected) < config.min_full_children_per_generation:
        selected_ids = {item["run_id"] for item in selected}
        fallback_pool = [
            item
            for item in rejected
            if item.get("run_id") not in selected_ids
            and float(item.get("stage1_decision", {}).get("mean_delta", 0.0) or 0.0)
            >= config.stage1_fallback_min_mean_delta
        ]
        needed = config.min_full_children_per_generation - len(selected)
        selected.extend(
            sorted(fallback_pool, key=_stage1_rank_key, reverse=True)[:needed]
        )

    selected_ids = {item["run_id"] for item in selected}
    hard_rejected = [
        item for item in rejected if item.get("run_id") not in selected_ids
    ]
    pruned = [item for item in passable if item.get("run_id") not in selected_ids]
    return {
        "selected": selected,
        "hard_rejected": hard_rejected,
        "pruned": pruned,
    }
