#!/usr/bin/env python3
"""Reconstruct the indexed SafeSearch split from verified upstream files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def prepare(source: Path | None, output: Path) -> dict[str, int]:
    metadata = json.loads((ROOT / "data/source.json").read_text(encoding="utf-8"))
    indices = {
        split: json.loads(
            (ROOT / f"data/splits/{split}.json").read_text(encoding="utf-8")
        )
        for split in ("train", "test", "excluded", "runtime_test20", "tgse_test20")
    }
    records = {}
    families = [item["family"] for item in metadata["files"]]
    for item in metadata["files"]:
        if source is None:
            url = f"{metadata['repository']}/resolve/{metadata['revision']}/{item['path']}"
            with urlopen(url, timeout=60) as response:
                content = response.read()
            expected = item["sha256"]
        else:
            path = source / item["path"]
            expected = item["sha256"]
            if not path.is_file():
                path = source / item["checkout_path"]
                expected = item["checkout_sha256"]
            content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(
                f"SHA-256 mismatch for {item['family']}; use the pinned upstream files"
            )
        rows = [
            json.loads(line)
            for line in content.decode("utf-8").split("\n")
            if line.strip()
        ]
        if len(rows) != item["count"]:
            raise ValueError(f"Unexpected row count for {item['family']}")
        for row in rows:
            key = (item["family"], row["test_case_id"])
            if key in records:
                raise ValueError(f"Duplicate source case: {key}")
            records[key] = row
    seen = set()
    for split in ("train", "test", "excluded"):
        entries = indices[split]
        for entry in entries:
            key = (entry["family"], entry["id"])
            if key not in records or key in seen:
                raise ValueError(f"Unknown or repeated case in {split}: {key}")
            seen.add(key)
    if seen != records.keys():
        raise ValueError("Split indices do not cover all source cases")
    held_out = {(entry["family"], entry["id"]) for entry in indices["test"]}
    for split in ("runtime_test20", "tgse_test20"):
        keys = [(entry["family"], entry["id"]) for entry in indices[split]]
        if len(set(keys)) != len(keys) or not set(keys) <= held_out:
            raise ValueError(f"Invalid held-out evaluation subset: {split}")
        if any(sum(family == item for item, _ in keys) != 20 for family in families):
            raise ValueError(f"Expected 20 cases per family: {split}")
    # Validate every source and index before creating output files.
    for split in ("train", "test", "runtime_test20", "tgse_test20"):
        target = output / split
        target.mkdir(parents=True, exist_ok=True)
        for family in families:
            rows = [
                records[(entry["family"], entry["id"])]
                for entry in indices[split]
                if entry["family"] == family
            ]
            content = "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in rows
            )
            (target / f"{family}.jsonl").write_text(content, encoding="utf-8")
    return {split: len(entries) for split, entries in indices.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        help="Local upstream dataset or SafeSearch repository checkout; omit to download pinned files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/safesearch",
        help="Output directory containing train/, test/, runtime_test20/, and tgse_test20/",
    )
    args = parser.parse_args()
    try:
        counts = prepare(args.source, args.output)
    except (OSError, URLError, ValueError, KeyError, TypeError) as exc:
        print(f"Data preparation failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Prepared {counts['train']} train and {counts['test']} test cases in {args.output}; excluded {counts['excluded']} cases. Also prepared runtime_test20 and tgse_test20 (100 each)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
