"""Typed contracts for the AgentLAB IH Search adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [item for raw in value if (item := _text(raw))]
    cleaned = _text(value)
    return [cleaned] if cleaned else []


def bounded_text(text: str, max_chars: int, marker: str) -> str:
    """Deterministically retain equal-sized head and tail excerpts."""
    value = str(text or "")
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    head = max_chars // 2
    tail = max_chars - head
    return value[:head] + marker + value[-tail:]


@dataclass(frozen=True)
class MatchedCaseContext:
    """Normalized full-information SafeSearch dossier for the IH Planner."""

    test_case_id: str
    risk_type: str
    current_date: str
    user_query: str
    user_expectation: str
    target_consequence: str
    risk_verification_checklist: list[str]
    risk_understanding: str
    potential_users: list[str]
    time_gap_discussion: str
    sources_of_unreliable_website: list[str]
    rationales_behind_unreliable_website_crafting: str
    website_generation_guideline: str
    risk_explanation: str

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
            risk_verification_checklist=_text_list(
                case.get("risk_verification_checklist")
            ),
            risk_understanding=_text(case.get("risk_understanding")),
            potential_users=_text_list(case.get("potential_users")),
            time_gap_discussion=_text(case.get("time_gap_discussion")),
            sources_of_unreliable_website=_text_list(
                case.get("sources_of_unreliable_website")
            ),
            rationales_behind_unreliable_website_crafting=_text(
                case.get("rationales_behind_unreliable_website_crafting")
            ),
            website_generation_guideline=_text(
                case.get("website_generation_guideline")
            ),
            risk_explanation=_text(case.get("risk_explanation")),
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

    def planner_prompt_view(self) -> dict[str, Any]:
        return {
            "test_case_id": self.test_case_id,
            "risk_type": self.risk_type,
            "current_date": self.current_date,
            "x_user_query": self.user_query,
            "e_user_expectation": self.user_expectation,
            "g_target_consequence": self.target_consequence,
            "K_risk_verification_checklist": self.risk_verification_checklist,
            "risk_understanding": self.risk_understanding,
            "potential_users": self.potential_users,
            "time_gap_discussion": self.time_gap_discussion,
            "sources_of_unreliable_website": self.sources_of_unreliable_website,
            "rationales_behind_unreliable_website_crafting": (
                self.rationales_behind_unreliable_website_crafting
            ),
            "website_generation_guideline": self.website_generation_guideline,
            "risk_explanation": self.risk_explanation,
        }

    def trace_view(self) -> dict[str, str]:
        """Limited case identity retained in runtime telemetry."""
        return {
            "test_case_id": self.test_case_id,
            "risk_type": self.risk_type,
            "task_description": self.user_query,
            "attack_goal": self.target_consequence,
        }


class FixedStrategy(BaseModel):
    """One fixed seven-stage strategy generated before the rollout."""

    model_config = ConfigDict(extra="forbid")

    persona: str = Field(min_length=1)
    initial_context: str = Field(min_length=1)
    approach: str = Field(min_length=1)
    conversation_plan: dict[str, str]
    key_phrases: list[str] = Field(default_factory=list)
    fallback_tactics: list[str] = Field(default_factory=list)

    @field_validator("persona", "initial_context", "approach", mode="before")
    @classmethod
    def validate_nonempty_strategy_text(cls, value: Any) -> str:
        cleaned = _text(value)
        if not cleaned:
            raise ValueError("strategy text fields must be non-empty")
        return cleaned

    @field_validator("key_phrases", "fallback_tactics", mode="before")
    @classmethod
    def clean_string_lists(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, (list, tuple)):
            raise ValueError("strategy phrase fields must be lists")
        return [cleaned for item in value if (cleaned := _text(item))]

    @field_validator("conversation_plan")
    @classmethod
    def validate_exact_plan(cls, value: dict[str, str]) -> dict[str, str]:
        expected = [f"turn_{index}" for index in range(1, 8)]
        if list(value.keys()) != expected:
            raise ValueError(
                "conversation_plan must contain ordered turn_1 through turn_7"
            )
        cleaned = {key: _text(item) for key, item in value.items()}
        if any(not item for item in cleaned.values()):
            raise ValueError("conversation_plan instructions must be non-empty")
        return cleaned


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    link: str = Field(min_length=1)
    snippet: str = Field(min_length=1)
    date: str = ""
    source: str = Field(min_length=1)

    @field_validator("title", "link", "snippet", "source", mode="before")
    @classmethod
    def validate_nonempty_result_text(cls, value: Any) -> str:
        cleaned = _text(value)
        if not cleaned:
            raise ValueError("search-result fields must be non-empty")
        return cleaned

    @field_validator("date", mode="before")
    @classmethod
    def clean_date(cls, value: Any) -> str:
        return _text(value)


class LandingPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    body: str = Field(min_length=1)

    @field_validator("title", "body", mode="before")
    @classmethod
    def validate_nonempty_page_text(cls, value: Any) -> str:
        cleaned = _text(value)
        if not cleaned:
            raise ValueError("landing-page fields must be non-empty")
        return cleaned


class GeneratedAsset(BaseModel):
    """Schema-constrained result and landing-page asset."""

    model_config = ConfigDict(extra="forbid")

    search_result: SearchResult
    page: LandingPage


@dataclass
class SyntheticAsset:
    search_result: dict[str, Any]
    page_title: str
    page_body: str
    stage: int
    query: str
    exposed_event_id: int
    visited: bool = False

    @property
    def url(self) -> str:
        return str(self.search_result["link"])

    def rendered_page(self) -> str:
        return (
            f"Title: {self.page_title}\n"
            f"URL Source: {self.url}\n\n"
            "Markdown Content:\n"
            f"{self.page_body}"
        )

    def prompt_view(self, page_chars: int) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "query": self.query,
            "search_result": {
                key: self.search_result.get(key, "")
                for key in ("title", "link", "snippet", "source")
            },
            "page": {
                "title": self.page_title,
                "body": bounded_text(
                    self.page_body,
                    page_chars,
                    "\n[... bounded page content omitted ...]\n",
                ),
            },
            "visited": self.visited,
        }
