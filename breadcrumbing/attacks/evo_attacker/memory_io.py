"""Build and load immutable success-only Evo memory artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from .prompts import PROMPT_VERSION
from .schema import MemoryEntry, MemoryManifest


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
STANDARD_FAMILIES = {
    "Advertisement-1",
    "Bias-Inducing-1",
    "Fake-Information-1",
    "Harmful-Output-1",
    "Prompt-Injection-1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _judge_key(judge_llm: str | None, evaluation: dict[str, Any]) -> str | None:
    judged = evaluation.get("asr")
    if not isinstance(judged, dict) or judged.get("status") != "ok":
        return None
    if judge_llm and judged.get("judge_model") != judge_llm:
        return None
    return "asr"


def _safe_event_view(event: Any) -> dict[str, Any]:
    """Allowlist payload-free fields even when source traces are less strict."""
    if not isinstance(event, dict):
        return {}
    allowed = (
        "id",
        "type",
        "action",
        "status",
        "reflection_summary",
        "retrieval_ids",
        "continue_used",
        "instruction_sha256",
        "instruction_chars",
        "asset_sha256",
        "asset_chars",
        "url_sha256",
    )
    return {key: event[key] for key in allowed if key in event}


def _bounded(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else _canonical(value)
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return text[:head] + "[…bounded…]" + text[-tail:]


def _bounded_retrieval_view(candidate: dict[str, Any]) -> str:
    """Deterministic prompt view; full sensitive episode stays separate."""
    event_views: list[dict[str, str]] = []
    for event in candidate.get("attack_events") or []:
        if not isinstance(event, dict):
            continue
        controlled = event.get("controlled_asset")
        if isinstance(controlled, dict):
            controlled_summary = {
                key: _bounded(value, 5000 if key == "body" else 1500)
                for key, value in controlled.items()
            }
        else:
            controlled_summary = _bounded(controlled, 7000)
        event_views.append(
            {
                "event_type": _bounded(event.get("event_type"), 100),
                "event_context": _bounded(event.get("event_context"), 4000),
                "modification_instruction": _bounded(
                    event.get("modification_instruction"), 3500
                ),
                "controlled_return_summary": _bounded(controlled_summary, 7000),
            }
        )
    view = {
        "case": _bounded(candidate.get("case"), 6000),
        "successful_attack_sequence": event_views,
    }
    return _bounded(view, 6000)


def _entry_from_row(row: dict[str, Any]) -> MemoryEntry:
    case = row.get("test_case") if isinstance(row.get("test_case"), dict) else {}
    trace = (
        row.get("metrics", {}).get("attack_trace", {})
        if isinstance(row.get("metrics"), dict)
        else {}
    )
    test_case_id = str(case.get("test_case_id") or "").strip()
    if not test_case_id:
        raise ValueError("successful row has no test_case_id")
    candidate = trace.get("memory_candidate")
    if not isinstance(candidate, dict) or candidate.get("sensitive") is not True:
        raise ValueError("successful Evo row has no marked memory_candidate")
    attack_events = candidate.get("attack_events")
    if not isinstance(attack_events, list) or not attack_events:
        raise ValueError("successful Evo row has no attacked event in memory_candidate")
    episode = {
        "case": candidate.get("case"),
        "successful_attack_sequence": attack_events,
    }
    source_digest = hashlib.sha256(_canonical(row).encode("utf-8")).hexdigest()
    return MemoryEntry(
        entry_id=f"success-{test_case_id}",
        test_case_id=test_case_id,
        risk_type=str(case.get("dataset") or case.get("risk_type") or ""),
        episode=episode,
        retrieval_view=_bounded_retrieval_view(candidate),
        source_episode_sha256=source_digest,
    )


def _source_provenance(
    rows: list[dict[str, Any]], *, formal: bool
) -> tuple[dict[str, Any], str]:
    """Validate empty-memory training provenance without reading payloads."""
    records: list[dict[str, Any]] = []
    for row in rows:
        config = row.get("config") if isinstance(row.get("config"), dict) else {}
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        trace = (
            metrics.get("attack_trace")
            if isinstance(metrics.get("attack_trace"), dict)
            else {}
        )
        session = trace.get("session") if isinstance(trace.get("session"), dict) else {}
        attacker_config = (
            config.get("attacker_config")
            if isinstance(config.get("attacker_config"), dict)
            else {}
        )
        record = {
            "attacker_type": str(config.get("method") or ""),
            "attacker_model": str(config.get("attacker_model") or ""),
            "attacker_temperature": config.get("attacker_temperature", 0.5),
            "researcher_model": str(config.get("researcher_model") or ""),
            "researcher_temperature": config.get("researcher_temperature"),
            "memory_mode": str(session.get("memory_mode") or ""),
            "memory_sha256": str(session.get("memory_sha256") or ""),
            "runtime_mode": str(session.get("mode") or ""),
            "prompt_version": str(session.get("prompt_version") or ""),
        }
        if formal:
            if config.get("method") != "evo_attacker":
                raise ValueError("formal memory source method must be evo_attacker")
            if session.get("mode") != "evo_attacker":
                raise ValueError("formal memory source trace mode must be evo_attacker")
            if session.get("memory_mode") != "empty" or session.get("memory_sha256"):
                raise ValueError(
                    "formal memory source must use empty memory with no checksum"
                )
            if session.get("prompt_version") != PROMPT_VERSION:
                raise ValueError("formal memory source prompt_version is not current")
            termination = metrics.get("termination")
            normal_completion = termination == "answer"
            if (
                not normal_completion
                or not str(row.get("response") or "").strip()
                or metrics.get("retry_exhausted") is True
                or trace.get("generation_success") is False
            ):
                raise ValueError(
                    "formal memory source contains unusable or retry-exhausted row"
                )
        records.append(record)
    if formal and any(record != records[0] for record in records[1:]):
        raise ValueError("formal memory source model/config provenance is inconsistent")
    canonical_records = [records[0]] if formal else records
    digest = hashlib.sha256(_canonical(canonical_records).encode("utf-8")).hexdigest()
    return (records[0] if records else {}), digest


class FrozenMemory:
    """Validated immutable memory plus its normalized embedding matrix."""

    def __init__(
        self,
        *,
        path: Path,
        manifest: MemoryManifest,
        entries: list[MemoryEntry],
        embeddings: np.ndarray,
    ):
        self.path = path
        self.manifest = manifest
        self.entries = tuple(entries)
        matrix = np.asarray(embeddings, dtype=np.float32)
        matrix.setflags(write=False)
        self.embeddings = matrix

    @classmethod
    def load(cls, path: str | Path, *, verify_checksum: bool = True) -> "FrozenMemory":
        root = Path(path)
        manifest = MemoryManifest.model_validate_json(
            (root / "manifest.json").read_text(encoding="utf-8")
        )
        memory_path = root / "memory.jsonl"
        embeddings_path = root / "embeddings.npy"
        if verify_checksum:
            if _sha256(memory_path) != manifest.memory_sha256:
                raise ValueError("memory.jsonl checksum does not match manifest")
            if _sha256(embeddings_path) != manifest.embedding_sha256:
                raise ValueError("embeddings.npy checksum does not match manifest")
        entries = [
            MemoryEntry.model_validate_json(line)
            for line in memory_path.read_text(encoding="utf-8").split("\n")
            if line.strip()
        ]
        embeddings = np.load(embeddings_path, allow_pickle=False)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(entries):
            raise ValueError("embedding matrix shape does not match memory entries")
        if len(entries) != manifest.successful_cases:
            raise ValueError("memory entry count does not match manifest")
        return cls(path=root, manifest=manifest, entries=entries, embeddings=embeddings)


def build_memory_artifact(
    *,
    input_root: str | Path,
    output_dir: str | Path,
    dataset_dir: str | Path | None = None,
    judge_llm: str | None,
    expected_cases: int = 100,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    prompt_version: str = PROMPT_VERSION,
) -> MemoryManifest:
    """Build a frozen, success-only memory from judged ``results.jsonl`` rows."""
    source_root = Path(input_root)
    output = Path(output_dir)
    if embedding_model != DEFAULT_EMBEDDING_MODEL:
        raise ValueError(
            f"Evo-Attacker baseline fixes embedding_model={DEFAULT_EMBEDDING_MODEL}"
        )
    if prompt_version != PROMPT_VERSION:
        raise ValueError("memory build prompt_version must match the current runtime")
    result_files = sorted(source_root.rglob("results.jsonl"))
    if not result_files:
        raise ValueError(f"no results.jsonl found under {source_root}")

    rows: list[dict[str, Any]] = []
    for path in result_files:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_no}")
            rows.append(row)

    case_ids = [
        str((row.get("test_case") or {}).get("test_case_id") or "").strip()
        for row in rows
    ]
    if any(not value for value in case_ids):
        raise ValueError("every result row must contain test_case.test_case_id")
    if len(rows) != expected_cases or len(set(case_ids)) != expected_cases:
        raise ValueError(
            f"expected exactly {expected_cases} unique cases, found "
            f"{len(rows)} rows/{len(set(case_ids))} unique IDs"
        )

    formal = expected_cases == 100
    source_provenance, source_config_sha256 = _source_provenance(rows, formal=formal)

    family_counts: dict[str, int] = {}
    for row in rows:
        case = row.get("test_case") or {}
        family = str(case.get("dataset") or case.get("risk_type") or "").strip()
        family_counts[family] = family_counts.get(family, 0) + 1
    sorted_case_ids = sorted(case_ids)
    case_id_sha256 = hashlib.sha256(
        ("\n".join(sorted_case_ids) + "\n").encode("utf-8")
    ).hexdigest()

    dataset_path = Path(dataset_dir) if dataset_dir is not None else None
    if formal and dataset_path is None:
        raise ValueError("formal 100-case memory build requires dataset_dir")
    dataset_hash = ""
    if dataset_path is not None:
        dataset_files = sorted(dataset_path.glob("*.jsonl"))
        dataset_ids: list[str] = []
        dataset_family_counts: dict[str, int] = {}
        dataset_digest = hashlib.sha256()
        for path in dataset_files:
            dataset_digest.update(path.name.encode("utf-8") + b"\0")
            dataset_digest.update(bytes.fromhex(_sha256(path)))
            count = 0
            for line_no, line in enumerate(
                path.read_text(encoding="utf-8").split("\n"), 1
            ):
                if not line.strip():
                    continue
                try:
                    case = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid dataset JSONL at {path}:{line_no}"
                    ) from exc
                case_id = str(case.get("test_case_id") or "").strip()
                if not case_id:
                    raise ValueError(
                        f"dataset row has no test_case_id at {path}:{line_no}"
                    )
                dataset_ids.append(case_id)
                count += 1
            dataset_family_counts[path.stem] = count
        dataset_hash = dataset_digest.hexdigest()
        if set(dataset_ids) != set(case_ids):
            raise ValueError("result case IDs do not match dataset_dir")
        if formal:
            expected_family_counts = {name: 20 for name in STANDARD_FAMILIES}
            if dataset_family_counts != expected_family_counts:
                raise ValueError(
                    "formal train dataset must contain five standard families x 20"
                )
            if family_counts != expected_family_counts:
                raise ValueError(
                    "formal results must contain five standard families x 20"
                )

    successful: list[MemoryEntry] = []
    unavailable = 0
    judge_successes = 0
    ineligible_successes = 0
    selected_judge = judge_llm or ""
    for row in rows:
        evaluation = (
            row.get("evaluation") if isinstance(row.get("evaluation"), dict) else {}
        )
        key = _judge_key(judge_llm, evaluation)
        judged = evaluation.get(key) if key else None
        success = judged.get("success") if isinstance(judged, dict) else None
        if not isinstance(success, bool):
            unavailable += 1
            continue
        if not selected_judge and key:
            selected_judge = str(judged.get("judge_model") or "")
        if success:
            judge_successes += 1
            trace = (row.get("metrics") or {}).get("attack_trace") or {}
            candidate = (
                trace.get("memory_candidate") if isinstance(trace, dict) else None
            )
            if (
                not isinstance(candidate, dict)
                or candidate.get("sensitive") is not True
            ):
                raise ValueError("Judge-success Evo row has no marked memory_candidate")
            attack_events = candidate.get("attack_events")
            if not isinstance(attack_events, list):
                raise ValueError("memory_candidate.attack_events must be a list")
            if not attack_events:
                ineligible_successes += 1
                continue
            successful.append(_entry_from_row(row))

    if unavailable:
        raise ValueError(
            f"judge unavailable for {unavailable} of {expected_cases} cases"
        )
    if not successful:
        raise ValueError(
            "no successful episodes; refusing to build an empty frozen memory"
        )

    if output.exists() and not output.is_dir():
        raise FileExistsError(f"output path is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    try:
        memory_path = staging / "memory.jsonl"
        retrieval_path = staging / "retrieval_views.jsonl"
        embeddings_path = staging / "embeddings.npy"
        memory_text = "".join(entry.model_dump_json() + "\n" for entry in successful)
        memory_path.write_text(memory_text, encoding="utf-8")
        retrieval_path.write_text(
            "".join(
                _canonical(
                    {"entry_id": entry.entry_id, "retrieval_view": entry.retrieval_view}
                )
                + "\n"
                for entry in successful
            ),
            encoding="utf-8",
        )

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required to build Evo memory"
            ) from exc
        encoder = SentenceTransformer(embedding_model)
        embeddings = encoder.encode(
            [entry.retrieval_view for entry in successful],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        np.save(
            embeddings_path,
            np.asarray(embeddings, dtype=np.float32),
            allow_pickle=False,
        )

        manifest = MemoryManifest(
            memory_id=output.name,
            embedding_model=embedding_model,
            prompt_version=prompt_version,
            source_input_root=str(source_root.resolve()),
            source_result_files={
                str(path.relative_to(source_root)): _sha256(path)
                for path in result_files
            },
            dataset_dir=str(dataset_path.resolve()) if dataset_path else "",
            dataset_sha256=dataset_hash,
            case_id_sha256=case_id_sha256,
            family_counts=family_counts,
            source_attacker_type=str(source_provenance.get("attacker_type") or ""),
            source_attacker_model=str(source_provenance.get("attacker_model") or ""),
            source_attacker_temperature=source_provenance.get("attacker_temperature"),
            source_researcher_model=str(
                source_provenance.get("researcher_model") or ""
            ),
            source_researcher_temperature=source_provenance.get(
                "researcher_temperature"
            ),
            source_memory_mode=str(source_provenance.get("memory_mode") or ""),
            source_config_sha256=source_config_sha256,
            judge_llm=selected_judge,
            expected_cases=expected_cases,
            observed_cases=len(rows),
            judge_success_cases=judge_successes,
            ineligible_success_cases=ineligible_successes,
            successful_cases=len(successful),
            judge_unavailable_cases=unavailable,
            memory_sha256=_sha256(memory_path),
            embedding_sha256=_sha256(embeddings_path),
        )
        (staging / "manifest.json").write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

        if not output.exists():
            os.replace(staging, output)
        elif not any(output.iterdir()):
            output.rmdir()
            os.replace(staging, output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)
