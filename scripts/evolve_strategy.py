#!/usr/bin/env python3
"""Evolve a strategy on user-supplied training cases, then optionally test it."""

import argparse
from dataclasses import replace
import json
import logging
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from breadcrumbing.tgse import TGSEConfig, evolve, evaluate_selected
from breadcrumbing.data import load_cases as load_dataset


def load_cases(path, dataset):
    return load_dataset({dataset: path})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-data", required=True, help="Training JSONL for one task family"
    )
    parser.add_argument(
        "--strategy", required=True, help="Initial strategy YAML (one strategy)"
    )
    parser.add_argument(
        "--config", required=True, help="YAML mapping of TGSEConfig fields"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--researcher-model", help="Override the researcher model name")
    parser.add_argument("--attacker-model", help="Override the attacker model name")
    parser.add_argument(
        "--evolution-model", help="Override reflection_model in the config"
    )
    parser.add_argument("--judge-model", help="Override the judge model name")
    parser.add_argument(
        "--compression-model", help="Override the trace compression model name"
    )
    parser.add_argument(
        "--workers", type=int, help="Override the global case worker cap"
    )
    parser.add_argument(
        "--repeats", type=int, help="Override repeated evaluations per case"
    )
    parser.add_argument("--test-data", help="Optional disjoint held-out JSONL")
    parser.add_argument("--test-cases", type=int, default=20)
    parser.add_argument("--test-seed", type=int, default=4242)
    parser.add_argument(
        "--test-researcher-mode",
        choices=("inherit", "on", "off", "both"),
        default="inherit",
        help="Cross-checking instructions during held-out evaluation",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = TGSEConfig(**yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
    overrides = {
        key: getattr(args, key)
        for key in (
            "researcher_model",
            "attacker_model",
            "judge_model",
            "compression_model",
            "workers",
            "repeats",
        )
        if getattr(args, key) is not None
    }
    if args.evolution_model is not None:
        overrides["reflection_model"] = args.evolution_model
    config = replace(config, **overrides)
    strategy = yaml.safe_load(Path(args.strategy).read_text(encoding="utf-8"))
    if isinstance(strategy, list):
        if len(strategy) != 1:
            raise ValueError("Initial strategy file must contain exactly one strategy")
        strategy = strategy[0]
    train_cases = load_cases(args.train_data, config.dataset_name)
    test_cases = None
    if args.test_data:
        test_cases = load_cases(args.test_data, config.dataset_name)
        train_ids = {case["test_case_id"] for case in train_cases}
        test_ids = [case["test_case_id"] for case in test_cases]
        if len(test_ids) != len(set(test_ids)) or train_ids.intersection(test_ids):
            raise ValueError(
                "Held-out IDs must be unique and disjoint from training IDs"
            )
        if args.test_cases < 1 or len(test_cases) < args.test_cases:
            raise ValueError("Not enough held-out cases for the requested test slice")
        test_cases = random.Random(args.test_seed).sample(test_cases, args.test_cases)
    logging.info(
        "TGSE: %d generations, %d children, %d full, %d train x %d repeats; workers=%d; max_llm_calls_per_run=%d",
        config.num_generations,
        config.children_per_generation,
        config.full_children_per_generation,
        config.full_cases_per_dataset,
        config.repeats,
        config.workers,
        config.max_llm_calls_per_run,
    )
    selection = evolve(train_cases, strategy, config, args.output_dir)
    if test_cases is not None:
        modes = {
            "inherit": {"inherit": config.researcher_adversarial_enhancement},
            "on": {"cross_check": True},
            "off": {"standard": False},
            "both": {"standard": False, "cross_check": True},
        }[args.test_researcher_mode]
        selection["test_metrics"] = {}
        for label, enabled in modes.items():
            test_config = replace(config, researcher_adversarial_enhancement=enabled)
            summary = evaluate_selected(
                test_cases,
                selection,
                test_config,
                Path(args.output_dir) / "test" / label,
            )
            selection["test_metrics"][label] = summary["aggregate_metrics"]
    print(json.dumps(selection, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
