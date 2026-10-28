"""Trace-guided archive search over natural-language strategies."""

from dataclasses import asdict, replace
from datetime import date
import fcntl
import hashlib
import json
import logging
import math
from pathlib import Path
import random

import yaml

from .config import TGSEConfig
from .evaluation import evaluate, combine
from .feedback import build_generation_memory
from .reflection import propose, extract_lessons
from .selection import (
    _select_stage1_cases,
    _subset_repeat_summary,
    _build_stage1_decision,
    _select_children_for_full,
    _stage_average_asr,
)
from .storage import load_json, save_json
from .strategy import _normalize_strategy

logger = logging.getLogger(__name__)


def select_parent(archive, rng):
    """Sample by score and the number of already archived descendants."""
    counts = {node["run_id"]: 0 for node in archive}
    for node in archive:
        if node.get("parent_run_id") in counts:
            counts[node["parent_run_id"]] += 1
    weights = [
        1
        / (1 + math.exp(-10 * (node["full_score"] - 0.5)))
        / (1 + counts[node["run_id"]])
        for node in archive
    ]
    return (
        rng.choices(archive, weights=weights, k=1)[0]
        if len(archive) > 1
        else archive[0]
    )


def lineage(archive, parent):
    by_id = {node["run_id"]: node for node in archive}
    history = []
    while parent:
        history.append(parent["memory"])
        parent = by_id.get(parent.get("parent_run_id"))
    return history[::-1]


def lesson_memory(archive, rejections, limit):
    def recent(entries, key):
        result = []
        values = [v for entry in entries for v in entry.get(key, [])]
        for value in reversed(values):
            value = value.strip()
            if value and value not in result:
                result.append(value)
                if len(result) >= limit:
                    break
        return result

    positive = recent(archive, "positive_lessons")
    negative = recent(rejections, "negative_lessons")
    parts = []
    if positive:
        parts.append("Preserve signals: " + " | ".join(positive[:3]))
    if negative:
        parts.append("Avoid signals: " + " | ".join(negative[:3]))
    return {
        "positive_lessons": positive,
        "negative_lessons": negative,
        "summary": " ".join(parts),
    }


def _save_strategy(strategy, directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "strategy.yaml"
    if not path.exists():
        path.write_text(
            yaml.safe_dump([strategy], sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    return path


def _tuple_state(value):
    return tuple(_tuple_state(x) for x in value) if isinstance(value, list) else value


def evolve(train_cases, initial_strategy, config: TGSEConfig, output_dir):
    """Evolve on one family's training cases and return its train-selected strategy.

    Artifacts support resuming the same inputs and settings. A single writer owns
    the output directory. Test evaluation is an explicit, separate operation.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".writer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another evolution writer owns this output directory"
            ) from exc
        return _evolve(train_cases, initial_strategy, config, output_dir)


def _evolve(train_cases, initial_strategy, config, output_dir):
    initial_strategy = _normalize_strategy(
        initial_strategy, initial_strategy.get("name", "strategy")
    )
    ids = [case["test_case_id"] for case in train_cases]
    if len(ids) != len(set(ids)) or len(ids) < config.full_cases_per_dataset:
        raise ValueError(
            "Training input needs unique case IDs and enough cases for the full slice"
        )
    if any(case.get("dataset") != config.dataset_name for case in train_cases):
        raise ValueError(
            "One evolution run must contain only its configured task family"
        )
    identity_path = output_dir / "inputs.json"
    if config.prompt_date is None:
        saved_date = (
            load_json(identity_path)["config"].get("prompt_date")
            if identity_path.exists()
            else None
        )
        config = replace(config, prompt_date=saved_date or date.today().isoformat())
    identity = {
        "config": asdict(config),
        "strategy": initial_strategy,
        "cases": train_cases,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    if (
        identity_path.exists()
        and load_json(identity_path)["fingerprint"] != fingerprint
    ):
        raise ValueError(
            "Output directory belongs to different inputs/settings; choose a new directory"
        )
    save_json({"fingerprint": fingerprint, **identity}, identity_path)
    rng = random.Random(config.sampling_seed)
    full_cases = [
        dict(
            case,
            _metadata={
                "source_dataset": config.dataset_name,
                "sampled_with_seed": config.sampling_seed,
            },
        )
        for case in rng.sample(train_cases, config.full_cases_per_dataset)
    ]
    save_json(full_cases, output_dir / "train_cases.json")
    state_path = output_dir / "state.json"
    state = (
        load_json(state_path)
        if state_path.exists()
        else {"archive": [], "rejections": [], "completed_generations": 0}
    )
    if state.get("rng_state"):
        rng.setstate(_tuple_state(state["rng_state"]))
    archive, rejections = state["archive"], state["rejections"]
    if not archive:
        strategy_path = _save_strategy(initial_strategy, output_dir / "initial")
        logger.info(
            "TGSE initial evaluation: %d cases x %d repeats",
            len(full_cases),
            config.repeats,
        )
        initial = evaluate(
            full_cases, strategy_path, config, output_dir / "initial" / "full"
        )
        summary = initial["summary"]
        archive.append(
            {
                "run_id": "init",
                "parent_run_id": None,
                "strategy": initial_strategy,
                "strategy_path": str(strategy_path.relative_to(output_dir)),
                "summary": summary,
                "full_score": round(_stage_average_asr(summary), 3),
                "positive_lessons": [],
                "memory": build_generation_memory("init", initial_strategy, summary),
            }
        )
        state["rng_state"] = rng.getstate()
        save_json(state, state_path)

    for generation in range(
        state["completed_generations"] + 1, config.num_generations + 1
    ):
        directory = output_dir / f"generation_{generation:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        plan_path = directory / "plan.json"
        if plan_path.exists():
            plan = load_json(plan_path)
            parent = next(node for node in archive if node["run_id"] == plan["parent"])
            rng.setstate(_tuple_state(plan["rng_state"]))
        else:
            parent = select_parent(archive, rng)
            plan = {"parent": parent["run_id"], "rng_state": rng.getstate()}
            save_json(plan, plan_path)
        logger.info(
            "TGSE generation %d/%d, parent=%s, training ASR=%.3f",
            generation,
            config.num_generations,
            parent["run_id"],
            parent["full_score"],
        )
        cheap_cases = _select_stage1_cases(
            parent["summary"],
            full_cases,
            config,
            selection_salt=f"{parent['run_id']}:gen_{generation:03d}",
        )
        save_json(cheap_cases, directory / "cheap_cases.json")
        cheap_ids = {case["test_case_id"] for case in cheap_cases}
        parent_cheap = _subset_repeat_summary(
            parent["summary"], list(cheap_ids), parent["strategy"]["name"]
        )
        history = lineage(archive, parent)
        lessons = (
            lesson_memory(archive, rejections, config.lesson_limit)
            if config.enable_lesson_memory
            else {"positive_lessons": [], "negative_lessons": [], "summary": ""}
        )
        children = []
        for index in range(config.children_per_generation):
            child_dir = directory / f"child_{index:02d}"
            proposal = propose(
                parent["strategy"],
                parent["summary"],
                history,
                lessons,
                generation,
                index,
                config,
                child_dir,
            )
            candidate = proposal["updated_strategy"]
            strategy_path = _save_strategy(candidate, child_dir)
            oracle = None
            if config.oracle_full_only:
                oracle = evaluate(full_cases, strategy_path, config, child_dir / "full")
                cheap = {
                    "summary": _subset_repeat_summary(
                        oracle["summary"], list(cheap_ids), candidate["name"]
                    )
                }
            else:
                cheap = evaluate(
                    cheap_cases, strategy_path, config, child_dir / "cheap"
                )
            decision = _build_stage1_decision(cheap["summary"], parent_cheap, config)
            if config.oracle_full_only:
                decision.update(pass_stage1=True, oracle_full_only=True)
            children.append(
                {
                    "run_id": f"gen_{generation:03d}_child_{index:02d}",
                    "status": "stage1_done",
                    "stage1_decision": decision,
                    "proposal": proposal,
                    "cheap": cheap,
                    "directory": child_dir,
                    "strategy_path": strategy_path,
                    "oracle": oracle,
                }
            )
        selection = (
            _select_children_for_full(children, config)
            if config.enable_stage1_gate and not config.oracle_full_only
            else {"selected": children, "hard_rejected": [], "pruned": []}
        )
        save_json(
            {
                name: [
                    {"run_id": child["run_id"], "decision": child["stage1_decision"]}
                    for child in group
                ]
                for name, group in selection.items()
            },
            directory / "promotion.json",
        )
        new_archive, new_rejections = [], []
        for child in children:
            child_dir = child["directory"]
            if child in selection["pruned"]:
                save_json(
                    {"status": "pruned", "reason": "full_evaluation_budget"},
                    child_dir / "outcome.json",
                )
                continue
            if child in selection["hard_rejected"]:
                negative = (
                    extract_lessons(
                        "negative",
                        child["proposal"]["strategy_diff"],
                        child["cheap"]["summary"],
                        child["stage1_decision"],
                        config,
                        child_dir,
                    )
                    if config.enable_lesson_memory
                    else []
                )
                new_rejections.append(
                    {
                        "run_id": child["run_id"],
                        "parent_run_id": parent["run_id"],
                        "negative_lessons": negative,
                    }
                )
                save_json(
                    {"status": "rejected", "decision": child["stage1_decision"]},
                    child_dir / "outcome.json",
                )
                continue
            remaining = [
                case for case in full_cases if case["test_case_id"] not in cheap_ids
            ]
            full = child["oracle"] or child["cheap"]
            if remaining and not config.oracle_full_only:
                rest = evaluate(
                    remaining, child["strategy_path"], config, child_dir / "remaining"
                )
                full = combine(full, rest, config.dataset_name)
            summary = full["summary"]
            save_json(summary, child_dir / "full_summary.json")
            context = {
                "stage": "full",
                "full_score": round(_stage_average_asr(summary), 3),
                "band_counts": summary["band_counts"],
                "bucket_counts": summary["bucket_counts"],
            }
            positive = (
                extract_lessons(
                    "positive",
                    child["proposal"]["strategy_diff"],
                    summary,
                    context,
                    config,
                    child_dir,
                )
                if config.enable_lesson_memory
                else []
            )
            candidate = child["proposal"]["updated_strategy"]
            memory = build_generation_memory(
                child["run_id"],
                candidate,
                summary,
                parent["strategy"],
                parent["summary"],
                history,
                child["proposal"],
            )
            new_archive.append(
                {
                    "run_id": child["run_id"],
                    "parent_run_id": parent["run_id"],
                    "strategy": candidate,
                    "strategy_path": str(
                        child["strategy_path"].relative_to(output_dir)
                    ),
                    "summary": summary,
                    "memory": memory,
                    "full_score": context["full_score"],
                    "positive_lessons": positive,
                }
            )
            save_json(
                {
                    "status": "archived",
                    "fallback_selected": config.enable_stage1_gate
                    and not config.oracle_full_only
                    and not child["stage1_decision"]["pass_stage1"],
                    "full_score": context["full_score"],
                },
                child_dir / "outcome.json",
            )
        archive.extend(new_archive)
        rejections.extend(new_rejections)
        state.update(completed_generations=generation, rng_state=rng.getstate())
        save_json(state, state_path)
        save_json(
            lesson_memory(archive, rejections, config.lesson_limit),
            output_dir / "lessons.json",
        )

    # Stable max keeps the earliest archived candidate when means tie.
    best = max(archive, key=lambda node: round(_stage_average_asr(node["summary"]), 6))
    best_path = output_dir / "selected_strategy.yaml"
    best_path.write_text(
        yaml.safe_dump([best["strategy"]], sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    result = {
        "selected_run_id": best["run_id"],
        "selection_metric": "mean_full_training_asr",
        "train_asr": _stage_average_asr(best["summary"]),
        "strategy_path": str(best_path),
        "archive_size": len(archive),
        "completed_generations": state["completed_generations"],
        "prompt_date": config.prompt_date,
    }
    save_json(result, output_dir / "selection.json")
    return result


def evaluate_selected(test_cases, selection, config, output_dir):
    """Evaluate a frozen train-selected strategy; no compression or feedback update."""
    if config.prompt_date is None and selection.get("prompt_date"):
        config = replace(config, prompt_date=selection["prompt_date"])
    return evaluate(
        test_cases, selection["strategy_path"], config, output_dir, compress=False
    )["summary"]
