"""Typed contracts for the Evo-Attacker no-RL attacker."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _checklist(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [text for item in value if (text := _text(item))]
    text = _text(value)
    return [text] if text else []


@dataclass(frozen=True)
class MatchedCaseContext:
    """The exact benchmark record ``c=(x,e,g,K)`` supplied by the runner."""

    test_case_id: str
    risk_type: str
    current_date: str
    user_query: str
    user_expectation: str
    target_consequence: str
    risk_verification_checklist: list[str]

    @classmethod
    def from_mapping(cls, case: dict[str, Any]) -> "MatchedCaseContext":
        if not isinstance(case, dict):
            raise TypeError("case_context must be a dictionary")
        context = cls(
            test_case_id=_text(case.get("test_case_id")),
            risk_type=_text(case.get("risk_type") or case.get("dataset")),
            current_date=_text(case.get("current_date")),
            user_query=_text(case.get("user_query")),
            user_expectation=_text(case.get("user_expectation")),
            target_consequence=_text(case.get("target_consequence")),
            risk_verification_checklist=_checklist(
                case.get("risk_verification_checklist")
            ),
        )
        missing = [
            name
            for name in ("test_case_id", "user_query", "target_consequence")
            if not getattr(context, name)
        ]
        if missing:
            raise ValueError(
                "case_context is missing required field(s): " + ", ".join(missing)
            )
        return context

    def semantic_dict(self) -> dict[str, Any]:
        """Return c=(x,e,g,K), deliberately excluding ACH-only fields."""
        return {
            "x_user_query": self.user_query,
            "e_user_expectation": self.user_expectation,
            "g_target_consequence": self.target_consequence,
            "K_risk_verification_checklist": self.risk_verification_checklist,
        }

    def audit_dict(self) -> dict[str, Any]:
        return asdict(self)


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)


class EventDecision(BaseModel):
    """First decision: ATTACK, one CONTINUE, or NOOP."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["ATTACK", "CONTINUE", "NOOP"]
    reflection_summary: str = Field(default="", max_length=1000)
    modification_instruction: str = Field(default="", max_length=4000)
    next_retrieval_query: str = Field(default="", max_length=2000)

    @model_validator(mode="before")
    @classmethod
    def exact_shape(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        action = value.get("action")
        expected = {
            "ATTACK": {"action", "reflection_summary", "modification_instruction"},
            "CONTINUE": {"action", "reflection_summary", "next_retrieval_query"},
            "NOOP": {"action", "reflection_summary"},
        }.get(action)
        if expected is not None and set(value) != expected:
            raise ValueError(f"{action} must contain exactly {sorted(expected)}")
        return value

    @model_validator(mode="after")
    def fields_match_action(self) -> "EventDecision":
        if not self.reflection_summary.strip():
            raise ValueError("reflection_summary must be non-empty")
        if self.action == "ATTACK" and not self.modification_instruction.strip():
            raise ValueError("ATTACK requires modification_instruction")
        if self.action == "CONTINUE" and not self.next_retrieval_query.strip():
            raise ValueError("CONTINUE requires next_retrieval_query")
        if self.action != "ATTACK" and self.modification_instruction:
            raise ValueError("only ATTACK may include modification_instruction")
        if self.action != "CONTINUE" and self.next_retrieval_query:
            raise ValueError("only CONTINUE may include next_retrieval_query")
        return self


class FinalEventDecision(BaseModel):
    """Decision after CONTINUE; a second CONTINUE is impossible by schema."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["ATTACK", "NOOP"]
    reflection_summary: str = Field(min_length=1, max_length=1000)
    modification_instruction: str = Field(default="", max_length=4000)

    @model_validator(mode="before")
    @classmethod
    def exact_shape(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        action = value.get("action")
        expected = {
            "ATTACK": {"action", "reflection_summary", "modification_instruction"},
            "NOOP": {"action", "reflection_summary"},
        }.get(action)
        if expected is not None and set(value) != expected:
            raise ValueError(f"{action} must contain exactly {sorted(expected)}")
        return value

    @model_validator(mode="after")
    def fields_match_action(self) -> "FinalEventDecision":
        if self.action == "ATTACK" and not self.modification_instruction.strip():
            raise ValueError("ATTACK requires modification_instruction")
        if self.action == "NOOP" and self.modification_instruction:
            raise ValueError("NOOP must not include modification_instruction")
        return self


class SearchCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=500)
    link: str = Field(min_length=1, max_length=2000)
    snippet: str = Field(min_length=1, max_length=2000)
    date: str = Field(default="", max_length=100)
    source: str = Field(min_length=1, max_length=300)


class LandingPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=500)
    body: str = Field(min_length=1, max_length=30000)


class MemoryEntry(BaseModel):
    """Success-only, episode-level retrieval record."""

    model_config = ConfigDict(extra="forbid")

    entry_id: str = Field(min_length=1, max_length=200)
    test_case_id: str = Field(min_length=1, max_length=200)
    risk_type: str = Field(default="", max_length=200)
    episode: dict[str, Any]
    retrieval_view: str = Field(min_length=1, max_length=30000)
    source_episode_sha256: str = Field(min_length=64, max_length=64)


class RetrievedMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_id: str
    score: float
    retrieval_view: str


class MemoryManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = 2
    memory_id: str = Field(min_length=1)
    frozen: Literal[True] = True
    embedding_model: str
    normalized_cosine: Literal[True] = True
    retrieval_k: Literal[5] = 5
    prompt_version: str
    source_input_root: str
    source_result_files: dict[str, str]
    dataset_dir: str
    dataset_sha256: str
    case_id_sha256: str
    family_counts: dict[str, int]
    source_attacker_type: str
    source_attacker_model: str
    source_attacker_temperature: float | None
    source_researcher_model: str
    source_researcher_temperature: float | None
    source_memory_mode: str
    source_config_sha256: str = Field(min_length=64, max_length=64)
    judge_llm: str
    expected_cases: int = Field(gt=0)
    observed_cases: int = Field(gt=0)
    judge_success_cases: int = Field(ge=0)
    ineligible_success_cases: int = Field(ge=0)
    successful_cases: int = Field(ge=0)
    judge_unavailable_cases: int = Field(ge=0)
    memory_sha256: str = Field(min_length=64, max_length=64)
    embedding_sha256: str = Field(min_length=64, max_length=64)
