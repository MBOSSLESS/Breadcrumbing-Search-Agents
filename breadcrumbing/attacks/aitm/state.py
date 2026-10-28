"""Per-rollout state for AiTM-Search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schema import MatchedCaseContext, SyntheticAsset


@dataclass
class AiTMSearchState:
    case: MatchedCaseContext

    # The only semantic state allowed into the next model prompt.
    previous_instruction: str = ""

    # Everything below is operational enforcement or audit state only.
    assets: dict[str, SyntheticAsset] = field(default_factory=dict)
    real_urls: set[str] = field(default_factory=set)
    real_hosts: set[str] = field(default_factory=set)
    events: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    generation_events: int = 0
    successful_injections: int = 0
    fallback_injections: int = 0
    url_rewrites: int = 0
    next_event_id: int = 1

    def add_event(self, event_type: str, **fields: Any) -> dict[str, Any]:
        event = {"id": self.next_event_id, "type": event_type, **fields}
        self.next_event_id += 1
        self.events.append(event)
        return event
