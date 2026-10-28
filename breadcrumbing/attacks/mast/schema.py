"""Typed runtime contracts for the MAST-Direct-Search baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _checklist(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [text for item in value if (text := _text(item))]
    text = _text(value)
    return [text] if text else []


def bounded_text(text: str, max_chars: int, marker: str) -> str:
    """Keep deterministic head/tail context without the ``[-0:]`` edge case."""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + marker + text[-tail:]


@dataclass(frozen=True)
class MatchedCaseContext:
    """Exact SafeSearch intent record supplied by the experiment runner."""

    test_case_id: str
    risk_type: str
    current_date: str
    user_query: str
    user_expectation: str
    target_consequence: str
    risk_verification_checklist: list[str]
    rationales_behind_unreliable_website_crafting: str
    website_generation_guideline: str

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
            rationales_behind_unreliable_website_crafting=_text(
                case.get("rationales_behind_unreliable_website_crafting")
            ),
            website_generation_guideline=_text(
                case.get("website_generation_guideline")
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

    @property
    def global_goal(self) -> str:
        """SafeSearch supplies MAST's global goal directly: G* := g."""
        return self.target_consequence

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PlannerDecision(BaseModel):
    """Minimal field mapping of MAST's public sub-goal decision."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["inject", "no_op"]
    target_behavior: str = ""
    subgoal: str = ""
    strategy: str = ""

    @model_validator(mode="before")
    @classmethod
    def validate_exact_shape(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        action = value.get("action")
        expected = (
            {"action"}
            if action == "no_op"
            else {"action", "target_behavior", "subgoal", "strategy"}
        )
        if set(value) != expected:
            raise ValueError(
                f"{action or 'unknown'} decision must contain exactly "
                f"{sorted(expected)}"
            )
        return value

    @model_validator(mode="after")
    def validate_action_fields(self) -> "PlannerDecision":
        if self.action == "inject":
            missing = [
                name
                for name in ("target_behavior", "subgoal", "strategy")
                if not getattr(self, name).strip()
            ]
            if missing:
                raise ValueError(
                    "inject decision is missing field(s): " + ", ".join(missing)
                )
        elif self.target_behavior or self.subgoal or self.strategy:
            raise ValueError("no_op must not contain planning fields")
        return self


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    link: str = Field(min_length=1)
    snippet: str = Field(min_length=1)
    date: str = ""
    source: str = Field(min_length=1)


class LandingPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    body: str = Field(min_length=1)


class GeneratedAsset(BaseModel):
    """The only output of MAST's Search-channel tampering stage."""

    model_config = ConfigDict(extra="forbid")

    search_result: SearchResult
    page: LandingPage


@dataclass
class SyntheticAsset:
    search_result: dict[str, Any]
    page_title: str
    page_body: str
    query: str
    planner_decision: dict[str, Any]
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

    def summary(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.search_result.get("title", ""),
            "snippet": self.search_result.get("snippet", ""),
            "source": self.search_result.get("source", ""),
            "query": self.query,
            "planner_decision": dict(self.planner_decision),
            "exposed_event_id": self.exposed_event_id,
            "visited": self.visited,
        }

    def prompt_view(self, max_page_chars: int) -> dict[str, Any]:
        """Content-bearing view used only inside the next policy prompt."""
        body = bounded_text(
            self.page_body,
            max_page_chars,
            "\n[... bounded synthetic page omitted ...]\n",
        )
        return {
            **self.summary(),
            "page": {
                "title": self.page_title,
                "body": body,
            },
        }
