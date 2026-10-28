"""Strict input loading and atomic, single-writer result storage."""

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_cases(cases):
    seen = set()
    for case in cases:
        for field in (
            "test_case_id",
            "dataset",
            "user_query",
            "user_expectation",
            "target_consequence",
            "risk_verification_checklist",
        ):
            if (
                field not in case
                or case[field] is None
                or case[field] == ""
                or case[field] == []
            ):
                raise ValueError(f"Missing or empty case field: {field}")
        for field in (
            "dataset",
            "user_query",
            "user_expectation",
            "target_consequence",
        ):
            if not isinstance(case[field], str) or not case[field].strip():
                raise ValueError(f"Case field must be a nonempty string: {field}")
        if type(case["test_case_id"]) not in {str, int}:
            raise ValueError("test_case_id must be a string or integer")
        checklist = case["risk_verification_checklist"]
        if not (isinstance(checklist, str) and checklist.strip()) and not (
            isinstance(checklist, list)
            and checklist
            and all(isinstance(item, str) and item.strip() for item in checklist)
        ):
            raise ValueError(
                "risk_verification_checklist must be a nonempty string or list of nonempty strings"
            )
        identity = case_id(case)
        if identity in seen:
            raise ValueError(f"Duplicate case ID: {identity}")
        seen.add(identity)
    if not cases:
        raise ValueError("No cases supplied")
    return cases


def case_id(case):
    return f"{case['dataset']}::{case['test_case_id']}"


def load_cases(datasets):
    cases = []
    for family, path in datasets.items():
        for case in read_jsonl(path):
            case = dict(case)
            if case.get("dataset", family) != family:
                raise ValueError("Dataset label disagrees with input mapping")
            case["dataset"] = family
            cases.append(case)
    return validate_cases(cases)


def atomic_write(path, value, jsonl=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            if jsonl:
                for row in value:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


@contextlib.contextmanager
def result_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".writer.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "Another generation or judge writer owns this result directory"
            ) from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def indexed_rows(rows):
    indexed = {}
    for row in rows:
        key = (row["case_id"], row["repeat_index"])
        if key in indexed:
            raise ValueError(f"Duplicate result identity: {key}")
        indexed[key] = row
    return indexed


def _pool_worker_init():
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


@contextlib.contextmanager
def process_pool(workers):
    """Stop paid worker jobs promptly when the caller interrupts or fails."""
    from concurrent.futures import ProcessPoolExecutor
    import signal
    import threading

    pool = ProcessPoolExecutor(max_workers=workers, initializer=_pool_worker_init)
    main_thread = threading.current_thread() is threading.main_thread()
    old_term = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise KeyboardInterrupt("Termination requested")

    if main_thread:
        signal.signal(signal.SIGTERM, interrupted)
    try:
        yield pool
    except BaseException:
        from multiprocessing.connection import wait

        processes = list((getattr(pool, "_processes", None) or {}).values())
        for process in processes:
            process.terminate()
        # The executor's management thread owns waitpid/join until shutdown.
        # Wait on OS sentinels here; competing Process.join/is_alive calls can
        # race its reaping and temporarily report an already-exited worker alive.
        sentinels = [process.sentinel for process in processes]
        exited = set(wait(sentinels, timeout=1)) if sentinels else set()
        for process in processes:
            if process.sentinel not in exited:
                process.kill()
        pool.shutdown(wait=True, cancel_futures=True)
        for process in processes:
            process.join()
        raise
    else:
        pool.shutdown(wait=True)
    finally:
        if main_thread:
            signal.signal(signal.SIGTERM, old_term)
