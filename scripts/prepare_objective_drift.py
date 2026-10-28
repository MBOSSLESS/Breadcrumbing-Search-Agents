#!/usr/bin/env python3
"""Train Objective Drift feedback memory on supplied training cases and freeze it."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import yaml
    from breadcrumbing.data import load_cases
    from breadcrumbing.objective_drift import prepare_memory

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True, help="Caller-supplied training JSONL")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Training repeats per case; original preparation uses one",
    )
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    config["repeats"] = args.repeats
    output = prepare_memory(
        load_cases({args.dataset_name: args.data}),
        config,
        args.output_dir,
        args.batch_size,
    )
    print(f"Frozen memory: {output}")


if __name__ == "__main__":
    main()
