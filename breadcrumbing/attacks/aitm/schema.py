"""Typed runtime contracts for the paper-guided AiTM Search adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MAX_TITLE_CHARS = 240
MAX_SNIPPET_CHARS = 1200
MAX_SOURCE_CHARS = 240
# Approximately the visible text that can fit under the shared 10k-token
# attacker completion budget; recorded in the trace for fairness audits.
MAX_INSTRUCTION_CHARS = 40000


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [cleaned for item in value if (cleaned := _text(item))]
    cleaned = _text(value)
    return [cleaned] if cleaned else []


def bounded_text(text: Any, max_chars: int, marker: str) -> str:
    """Deterministically retain equal-sized head and tail excerpts."""
    value = str(text or "")
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    marker = marker[:max_chars]
    remaining = max(0, max_chars - len(marker))
    head = remaining // 2
    tail = remaining - head
    return value[:head] + marker + (value[-tail:] if tail else "")


@dataclass(frozen=True)
class MatchedCaseContext:
    """Only c=(x,e,g,K), plus a non-semantic identifier for auditing/URLs."""

    test_case_id: str
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
            user_query=_text(case.get("user_query")),
            user_expectation=_text(case.get("user_expectation")),
            target_consequence=_text(case.get("target_consequence")),
            risk_verification_checklist=_text_list(
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

    def prompt_view(self) -> dict[str, Any]:
        return {
            "x_user_query": self.user_query,
            "e_user_expectation": self.user_expectation,
            "g_target_consequence": self.target_consequence,
            "K_risk_verification_checklist": self.risk_verification_checklist,
        }


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=MAX_TITLE_CHARS)
    link: str = Field(min_length=1, max_length=2048)
    snippet: str = Field(min_length=1, max_length=MAX_SNIPPET_CHARS)
    date: str = Field(default="", max_length=100)
    source: str = Field(min_length=1, max_length=MAX_SOURCE_CHARS)

    @field_validator("title", "link", "snippet", "date", "source", mode="before")
    @classmethod
    def clean_text(cls, value: Any) -> str:
        return _text(value)


class AiTMGeneration(BaseModel):
    """One AiTM call: I_t plus its Search-channel delivery envelope."""

    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(min_length=1, max_length=MAX_INSTRUCTION_CHARS)
    search_result: SearchResult

    @field_validator("instruction", mode="before")
    @classmethod
    def clean_instruction(cls, value: Any) -> str:
        cleaned = _text(value)
        if not cleaned:
            raise ValueError("instruction must be non-empty")
        return cleaned


class AiTMProviderResponse(BaseModel):
    """Provider compatibility envelope normalized to ``AiTMGeneration``.

    OpenRouter-style backends without native response-schema support may
    return direct reader-facing text, which the unified client exposes as a
    single ``result`` field. This transport envelope lets the policy preserve
    that same generated I_t and deterministically serialize its Search card;
    it is never the runtime/public output schema.
    """

    model_config = ConfigDict(extra="forbid")

    instruction: str | None = None
    search_result: SearchResult | None = None
    result: str | None = Field(default=None, max_length=MAX_INSTRUCTION_CHARS)

    @model_validator(mode="after")
    def validate_mode(self) -> "AiTMProviderResponse":
        has_structured = self.instruction is not None or self.search_result is not None
        has_result = bool(str(self.result or "").strip())
        if has_structured:
            if self.instruction is None or self.search_result is None or has_result:
                raise ValueError(
                    "provider response must contain either instruction+search_result "
                    "or one direct result"
                )
        elif not has_result:
            raise ValueError("provider response is empty")
        return self


@dataclass
class SyntheticAsset:
    search_result: dict[str, Any]
    instruction: str
    query: str
    exposed_event_id: int
    generation_status: str
    visited: bool = False
    visit_count: int = 0

    @property
    def url(self) -> str:
        return str(self.search_result["link"])

    def rendered_page(self) -> str:
        """The body is exactly I_t; only the transport wrapper is added."""
        return (
            f"Title: {self.search_result['title']}\n"
            f"URL Source: {self.url}\n\n"
            "Markdown Content:\n"
            f"{self.instruction}"
        )
