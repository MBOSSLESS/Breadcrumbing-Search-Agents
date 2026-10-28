#!/usr/bin/env python3
"""Judge completed generation outputs with the checklist evaluator."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from breadcrumbing.evaluation import judge_results

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    judge_results(args.output_dir, args.judge_model, args.workers)


if __name__ == "__main__":
    main()
