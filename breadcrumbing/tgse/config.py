"""Paper settings for one task-family strategy evolution run."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TGSEConfig:
    dataset_name: str
    full_cases_per_dataset: int = 20
    cheap_cases_per_dataset: int = 10
    repeats: int = 5
    num_generations: int = 4
    children_per_generation: int = 3
    full_children_per_generation: int = 2
    min_full_children_per_generation: int = 1
    sampling_seed: int = 42
    prompt_date: str | None = None
    researcher_model: str = "qwen3"
    attacker_model: str = "qwen3"
    judge_model: str = "qwen3"
    reflection_model: str = "qwen3.6-plus"
    compression_model: str = "qwen3"
    researcher_temperature: float = 0.0
    attacker_temperature: float = 0.5
    reflection_temperature: float = 0.5
    compression_temperature: float = 0.0
    reflection_max_tokens: int = 20000
    researcher_adversarial_enhancement: bool = False
    workers: int = 1
    judge_workers: int = 1
    max_attempts: int = 3
    max_llm_calls_per_run: int = 10
    max_missing_compressed_case_ratio: float = 0.2
    enable_lesson_memory: bool = True
    enable_stage1_gate: bool = True
    oracle_full_only: bool = False
    lesson_limit: int = 12
    stage1_max_high_drop_count: int = 0
    stage1_max_worse_rate: float = 0.4
    stage1_min_mean_delta: float = -0.05
    stage1_reject_severe_drop_count: int = 2
    stage1_fallback_min_mean_delta: float = -0.1
    stage1_selection_mode: str = "dynamic_bucket"
    cheap_flip_cases: int = 7
    cheap_stable_success_cases: int = 3
    cheap_stable_failure_cases: int = 0

    def __post_init__(self):
        positive = (
            self.full_cases_per_dataset,
            self.cheap_cases_per_dataset,
            self.repeats,
            self.children_per_generation,
            self.full_children_per_generation,
            self.workers,
            self.judge_workers,
            self.max_attempts,
            self.max_llm_calls_per_run,
        )
        if any(n < 1 for n in positive) or self.num_generations < 0:
            raise ValueError(
                "Counts and budgets must be positive; generations may be zero"
            )
        if self.cheap_cases_per_dataset > self.full_cases_per_dataset:
            raise ValueError("Cheap subset cannot exceed the full training slice")
        if (
            not 0
            <= self.min_full_children_per_generation
            <= self.full_children_per_generation
            <= self.children_per_generation
        ):
            raise ValueError("Require 0 <= minimum full <= maximum full <= children")
        if not 0 <= self.max_missing_compressed_case_ratio <= 1:
            raise ValueError("Missing compression ratio must be between zero and one")
