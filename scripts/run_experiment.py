#!/usr/bin/env python3
"""Run a configured experiment on a caller-supplied dataset."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import yaml
    from breadcrumbing.data import load_cases
    from breadcrumbing.runner import run_experiment

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        help="YAML defaults; explicit options override only their corresponding fields",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--data", help="Replace configured datasets with this JSONL file"
    )
    parser.add_argument("--dataset-name", help="Family label used with --data")
    parser.add_argument(
        "--method",
        choices=[
            "no_attack",
            "ach",
            "agentdojo",
            "geo",
            "safesearch",
            "objective_drift",
            "aitm",
            "mast",
            "intent_hijacking",
            "evo_attacker",
        ],
    )
    parser.add_argument("--researcher-model")
    parser.add_argument("--attacker-model")
    parser.add_argument("--judge-model")
    parser.add_argument("--strategy", dest="strategies_path")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--retry-failed", action="store_true", default=None)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    for key in (
        "method",
        "researcher_model",
        "attacker_model",
        "judge_model",
        "strategies_path",
        "workers",
        "repeats",
        "max_attempts",
        "retry_failed",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
            if key == "strategies_path":
                config.pop("strategy_map", None)
    if args.dataset_name and not args.data:
        parser.error("--dataset-name requires --data")
    if args.data:
        config["datasets"] = {args.dataset_name or Path(args.data).stem: args.data}
    if not config.get("datasets"):
        parser.error("Supply --data or a config containing datasets")
    if not config.get("researcher_model"):
        parser.error("Supply --researcher-model or configure researcher_model")
    if config.get("method", "ach") != "no_attack" and not config.get("attacker_model"):
        parser.error("Supply --attacker-model or configure attacker_model")
    if config.get("method", "ach") == "ach" and not (
        config.get("strategies_path") or config.get("strategy_map")
    ):
        parser.error("ACH requires --strategy or configured strategies_path")
    run_experiment(load_cases(config["datasets"]), config, args.output_dir)


if __name__ == "__main__":
    main()
