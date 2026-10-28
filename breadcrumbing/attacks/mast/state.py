"""Per-rollout state for MAST-Direct-Search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schema import MatchedCaseContext, SyntheticAsset


@dataclass
class MASTState:
    case: MatchedCaseContext
    events: list[dict[str, Any]] = field(default_factory=list)
    assets: dict[str, SyntheticAsset] = field(default_factory=dict)
    successful_injections: int = 0
    no_op_count: int = 0
    planner_calls: int = 0
    generator_calls: int = 0
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    real_urls: set[str] = field(default_factory=set)
    real_hosts: set[str] = field(default_factory=set)
    next_event_id: int = 1

    def add_event(self, event_type: str, **fields: Any) -> dict[str, Any]:
        event = {"id": self.next_event_id, "type": event_type, **fields}
        self.next_event_id += 1
        self.events.append(event)
        return event

    def prompt_history(self) -> list[dict[str, Any]]:
        """Bounded, public-event-only history for the next Planner call."""
        return list(self.events)

    def asset_summaries(self) -> list[dict[str, Any]]:
        return [asset.summary() for asset in self.assets.values()]

    def asset_prompt_views(self, max_page_chars: int) -> list[dict[str, Any]]:
        return [asset.prompt_view(max_page_chars) for asset in self.assets.values()]
