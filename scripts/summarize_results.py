#!/usr/bin/env python3
"""Compute family macro statistics only for complete, valid experiments."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from breadcrumbing.evaluation import summarize_results

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("--judge-model", required=True)
    args = parser.parse_args()
    summary = summarize_results(args.output_dir, args.judge_model)
    print(f"{'Family':<24} {'ASR':>8} {'MAXNASR':>8} {'SS':>5} {'Flip':>5} {'SF':>5}")
    for family, row in [
        *summary["families"].items(),
        ("Macro / total counts", summary),
    ]:
        print(
            f"{family:<24} {row['ASR']:>7.1%} {row['MAXNASR']:>7.1%} {row['SS']:>5} {row['Flip']:>5} {row['SF']:>5}"
        )
    print(f"Full report: {Path(args.output_dir) / 'summary.json'}")


if __name__ == "__main__":
    main()
