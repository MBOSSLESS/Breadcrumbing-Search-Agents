"""Build a frozen success-only Evo retrieval memory from judged training results."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_CACHE_ROOT = Path("cache/attacker/evo_attacker")
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _memory_output_dir(cache_root: Path, memory_id: str) -> Path:
    """Resolve a single cache child; reject path traversal and implicit nesting."""
    normalized = memory_id.strip()
    if (
        not normalized
        or normalized in {".", ".."}
        or Path(normalized).name != normalized
    ):
        raise ValueError("--memory-id must be one non-empty directory name")
    return cache_root / normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a frozen, success-only Evo-Attacker memory from already judged "
            "training results; no LLM rollout is started."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Judged train result root, normally results/<train_experiment_tag>",
    )
    parser.add_argument(
        "--memory-id",
        required=True,
        help="Artifact directory name under --cache-root",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help=(
            "Source dataset directory used to audit case IDs/family counts; "
            "required by memory_io for the formal 100-case build"
        ),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help=f"Memory cache root (default: {DEFAULT_CACHE_ROOT})",
    )
    parser.add_argument(
        "--judge-llm",
        default=None,
        help="Exact judge suffix used in evaluation keys; auto-detect when omitted",
    )
    parser.add_argument(
        "--expected-cases",
        type=int,
        default=100,
        help="Require this many unique train cases (default: 100)",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Sentence-transformers model (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_cases <= 0:
        raise ValueError("--expected-cases must be positive")
    if not args.input_root.is_dir():
        raise FileNotFoundError(f"Input result root does not exist: {args.input_root}")
    if args.expected_cases == 100 and args.dataset_dir is None:
        raise ValueError("--dataset-dir is required for the formal 100-case build")
    if args.dataset_dir is not None and not args.dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {args.dataset_dir}")

    output_dir = _memory_output_dir(args.cache_root, args.memory_id)

    from breadcrumbing.attacks.evo_attacker.memory_io import build_memory_artifact

    build_memory_artifact(
        input_root=args.input_root,
        output_dir=output_dir,
        dataset_dir=args.dataset_dir,
        judge_llm=args.judge_llm,
        expected_cases=args.expected_cases,
        embedding_model=args.embedding_model,
    )
    print(f"Frozen Evo memory built: {output_dir}")
    print("Payloads were not printed; inspect manifest.json for the auditable summary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
