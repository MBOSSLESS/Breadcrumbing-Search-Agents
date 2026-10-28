"""One process pool across cases and repeats; the parent alone writes results."""

from concurrent.futures import as_completed
import json
from datetime import date
from pathlib import Path
import time

from .data import (
    atomic_write,
    case_id,
    fingerprint,
    indexed_rows,
    read_jsonl,
    result_lock,
    validate_cases,
    process_pool,
)


def _run_case(case, repeat_index, config):
    from breadcrumbing.agent.react_agent import MultiTurnReactAgent

    row = {
        "case_id": case_id(case),
        "repeat_index": repeat_index,
        "test_case": case,
        "config": config,
    }
    for attempt in range(1, config.get("max_attempts", 1) + 1):
        try:
            method = config.get("method", "ach")
            kwargs = (
                {}
                if method in {"agentdojo", "geo", "no_attack"}
                else {
                    "model_name": config.get("attacker_model", ""),
                    "temperature": config.get(
                        "attacker_temperature", 0.8 if method == "safesearch" else 0.5
                    ),
                }
            )
            kwargs.update(config.get("attacker_config", {}))
            kwargs["case_context"] = case
            strategy = config.get("strategy_map", {}).get(
                case["dataset"], config.get("strategies_path")
            )
            if strategy and method == "ach":
                kwargs["strategies_path"] = strategy
            agent = MultiTurnReactAgent(
                max_llm_calls_per_run=config.get("max_llm_calls_per_run", 10),
                llm={
                    "model": config.get("tokenizer_path", ""),
                    "generate_cfg": {
                        "temperature": config.get(
                            "temperature", config.get("researcher_temperature", 0.0)
                        ),
                        "top_p": 0.95,
                        "max_tokens": config.get("researcher_max_tokens", 10000),
                    },
                },
                attacker=None if method in {"no_attack"} else method,
                attacker_kwargs=None if method in {"no_attack"} else kwargs,
                researcher_adversarial_enhancement=config.get(
                    "researcher_adversarial_enhancement", False
                ),
            )
            started = time.monotonic()
            result = agent._run(
                {"item": {"question": case["user_query"]}},
                model_name=config["researcher_model"],
                prompt_date=config.get("prompt_date"),
            )
            messages = result.get("messages", [])
            metrics = dict(result.get("metadata", {}))
            metrics.update(
                total_time=time.monotonic() - started,
                termination=result.get("termination"),
                messages=messages,
                num_messages=len(messages),
                attempts=attempt,
            )
            attacker = getattr(agent, "attacker", None)
            metrics["attack_trace"] = (
                attacker.get_behavior_logs()
                if attacker and hasattr(attacker, "get_behavior_logs")
                else None
            )
            response = result.get("prediction", "")
            row.update(response=response, messages=messages, metrics=metrics)
            if (
                not isinstance(response, str)
                or not response.strip()
                or response.startswith("ERROR:")
            ):
                raise ValueError("Empty or error response")
            if metrics["termination"] not in {
                "answer",
                "token limit exceeded - forced answer",
            }:
                # Preserve the outcome but require review before scoring an abnormal rollout.
                raise ValueError(f"Unrecognized termination: {metrics['termination']}")
            trace = metrics.get("attack_trace")
            if isinstance(trace, dict) and trace.get("generation_success") is False:
                raise ValueError("Attacker generation failed")
            if response.strip() == "No answer found.":
                raise ValueError("No answer found")
            if "error" in metrics:
                raise ValueError("Runtime metadata contains an error")
            metrics["retry_exhausted"] = False
            return row
        except Exception as error:
            row.setdefault("response", "")
            row.setdefault("messages", [])
            row.setdefault("metrics", {}).update(
                retry_exhausted=True,
                error_type=type(error).__name__,
                failure_reason=str(error),
                attempts=attempt,
            )
    return row


def run_experiment(cases, config, output_dir):
    cases = validate_cases(cases)
    config = dict(config)
    repeats = int(config.get("repeats", 5))
    workers = int(config.get("workers", 1))
    if min(repeats, workers, int(config.get("max_attempts", 1))) < 1:
        raise ValueError("repeats, workers and max_attempts must be positive")
    directory = Path(output_dir)
    path = directory / "results.jsonl"
    semantic = {
        k: v
        for k, v in config.items()
        if k not in {"workers", "judge_workers", "retry_failed"}
    }
    strategy = config.get("strategies_path")
    strategy_map = config.get("strategy_map", {})
    if strategy_map:
        missing = {case["dataset"] for case in cases} - set(strategy_map)
        if missing:
            raise ValueError(
                f"Strategy map is missing task families: {sorted(missing)}"
            )
    strategy_hashes = {
        family: fingerprint(Path(path).read_text())
        for family, path in strategy_map.items()
    }
    memory = config.get("attacker_config", {}).get("memory_file")
    evo_memory_sha256 = None
    attacker_config = config.get("attacker_config", {})
    if (
        config.get("method") == "evo_attacker"
        and attacker_config.get("memory_mode") == "frozen"
    ):
        from .attacks.evo_attacker.memory_io import FrozenMemory

        memory_path = attacker_config.get("memory_path")
        if memory_path is None:
            memory_id = attacker_config.get("memory_id")
            if (
                not memory_id
                or Path(memory_id).name != memory_id
                or memory_id in {".", ".."}
            ):
                raise ValueError(
                    "frozen memory requires memory_path or a valid memory_id"
                )
            memory_path = (
                Path(attacker_config.get("cache_root", "cache/attacker/evo_attacker"))
                / memory_id
            )
        frozen_memory = FrozenMemory.load(memory_path)
        evo_memory_sha256 = fingerprint(frozen_memory.manifest.model_dump())
    manifest = {
        "schema_version": 1,
        "config": semantic,
        "cases_sha256": fingerprint(cases),
        "strategy_sha256": fingerprint(Path(strategy).read_text())
        if strategy
        else None,
        "strategy_map_sha256": strategy_hashes,
        "memory_sha256": fingerprint(Path(memory).read_text()) if memory else None,
        "evo_memory_sha256": evo_memory_sha256,
        "expected": [
            {"case_id": case_id(case), "repeat_index": rep}
            for case in cases
            for rep in range(1, repeats + 1)
        ],
    }
    with result_lock(directory):
        manifest_path = directory / "manifest.json"
        previous = (
            json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        )
        if not config.get("prompt_date"):
            config["prompt_date"] = (
                previous["config"].get("prompt_date", date.today().isoformat())
                if previous
                else date.today().isoformat()
            )
        semantic["prompt_date"] = config["prompt_date"]
        if previous is not None:
            if previous != manifest:
                raise ValueError(
                    "Resume rejected: configuration, dataset, strategy or memory changed"
                )
        else:
            if path.exists():
                raise ValueError("Existing results have no manifest")
            atomic_write(manifest_path, manifest)
        rows = indexed_rows(read_jsonl(path) if path.exists() else [])
        expected = {
            (item["case_id"], item["repeat_index"]) for item in manifest["expected"]
        }
        if set(rows) - expected:
            raise ValueError("Results contain unexpected case/repeat identities")
        jobs = [
            (case, rep)
            for case in cases
            for rep in range(1, repeats + 1)
            if (case_id(case), rep) not in rows
            or (
                config.get("retry_failed", False)
                and rows[(case_id(case), rep)]["metrics"].get("retry_exhausted")
            )
        ]
        print(
            f"generation: {len(jobs)} pending; workers={workers}; max_llm_calls_per_run={config.get('max_llm_calls_per_run', 10)}",
            flush=True,
        )
        with process_pool(workers) as pool:
            futures = {
                pool.submit(_run_case, case, rep, config): (case, rep)
                for case, rep in jobs
            }
            for future in as_completed(futures):
                row = future.result()
                rows[(row["case_id"], row["repeat_index"])] = row
                atomic_write(path, [rows[key] for key in sorted(rows)], jsonl=True)
                print(
                    f"generation: {len(rows)}/{len(expected)} stored; failed={sum(bool(r['metrics'].get('retry_exhausted')) for r in rows.values())}",
                    flush=True,
                )
    return [rows[key] for key in sorted(rows)]


def evaluate_strategy(cases, strategy, config, output_dir):
    from .evaluation import judge_results, summarize_results

    config = dict(config, method="ach", strategies_path=str(strategy))
    config.pop("strategy_map", None)
    run_experiment(cases, config, output_dir)
    judge_results(output_dir, config["judge_model"], config.get("judge_workers", 1))
    summarize_results(output_dir, config["judge_model"])
    return read_jsonl(Path(output_dir) / "results.jsonl")
