"""Evaluate repeated trajectories and form the feedback used by TGSE."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .compression import compress_cases_with_resume
from .feedback import aggregate_repeat_outputs, enrich_compressed_summaries
from .selection import _with_band_metadata
from .storage import load_json, save_json


def summarize(rows, compressed, dataset):
    entries = {}
    for row in rows:
        entries.setdefault(f"rep_{row['repeat_index']:02d}", []).append(row)
    summary = aggregate_repeat_outputs(entries, compressed, dataset)
    summary["repeat_metrics"] = {
        name: {
            "overall_asr": sum(row["evaluation"]["asr"]["success"] for row in values)
            / len(values),
            "total_cases": len(values),
        }
        for name, values in entries.items()
    }
    return _with_band_metadata(summary)


def evaluate(cases, strategy_path, config, output_dir, *, compress=True):
    """Run a stage; incomplete generation/judging never becomes a failed attack."""
    from breadcrumbing.runner import evaluate_strategy

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "cases": cases,
        "strategy": Path(strategy_path).read_text(encoding="utf-8"),
        "config": asdict(config),
        "compress": compress,
    }
    signature = hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    identity_path = output_dir / "identity.json"
    if identity_path.exists() and load_json(identity_path)["fingerprint"] != signature:
        raise ValueError(
            "Stage output belongs to different cases, strategy, or evaluation settings"
        )
    save_json({"fingerprint": signature}, identity_path)
    result_file = output_dir / "feedback.json"
    if result_file.exists():
        return load_json(result_file)
    settings = asdict(config)
    settings["method"] = "ach"
    settings["temperature"] = config.researcher_temperature
    settings["retry_failed"] = True
    rows = evaluate_strategy(cases, strategy_path, settings, output_dir / "evaluation")
    expected = {
        (case["test_case_id"], r)
        for case in cases
        for r in range(1, config.repeats + 1)
    }
    keys = [(row["test_case"]["test_case_id"], row["repeat_index"]) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Stage results do not match the expected case/repeat grid")
    if any(
        type(row.get("evaluation", {}).get("asr", {}).get("success")) is not bool
        for row in rows
    ):
        raise ValueError("Stage has a missing or invalid judge outcome")
    compressed = {}
    if compress:
        for repeat in range(1, config.repeats + 1):
            repeat_rows = [row for row in rows if row["repeat_index"] == repeat]
            directory = output_dir / "compression" / f"rep_{repeat:02d}"
            summaries, _, usage = compress_cases_with_resume(
                repeat_rows,
                str(directory / "cases.jsonl"),
                config.compression_model,
                checkpoint_file=str(directory / "checkpoint.json"),
                temperature=config.compression_temperature,
                num_workers=config.workers,
            )
            save_json(usage, directory / "usage.json")
            missing_ratio = 1 - len(summaries) / len(repeat_rows)
            if missing_ratio > config.max_missing_compressed_case_ratio:
                raise RuntimeError(
                    f"Compression missing for {missing_ratio:.1%} of repeat {repeat}; resume after recovery"
                )
            compressed[f"rep_{repeat:02d}"] = enrich_compressed_summaries(
                repeat_rows, summaries
            )
    result = {
        "rows": rows,
        "compressed": compressed,
        "summary": summarize(rows, compressed, config.dataset_name),
    }
    save_json(result, result_file)
    return result


def combine(first, second, dataset):
    """Reuse cheap-stage repeats and add only the remaining training cases."""
    rows = first["rows"] + second["rows"]
    compressed = {
        name: first["compressed"].get(name, []) + second["compressed"].get(name, [])
        for name in first["compressed"].keys() | second["compressed"].keys()
    }
    return {
        "rows": rows,
        "compressed": compressed,
        "summary": summarize(rows, compressed, dataset),
    }
