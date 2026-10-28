"""Per-rollout operational and payload-free semantic state."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from .schema import MatchedCaseContext, SearchCard


@dataclass
class RegisteredAsset:
    card: SearchCard
    search_query: str
    exposed_event_id: int
    visit_count: int = 0


@dataclass
class EvoState:
    case: MatchedCaseContext
    assets: dict[str, RegisteredAsset] = field(default_factory=dict)
    real_urls: set[str] = field(default_factory=set)
    real_hosts: set[str] = field(default_factory=set)
    safe_history: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    next_event_id: int = 1
    search_injections: int = 0
    visit_attacks: int = 0
    visit_defaults: int = 0
    organic_visit_passthroughs: int = 0
    generation_failures: int = 0
    # Sensitive offline-only artifact. Excluded from routine telemetry.
    memory_attack_events: list[dict[str, Any]] = field(default_factory=list)

    def add_event(self, event_type: str, **fields: Any) -> dict[str, Any]:
        event = {"id": self.next_event_id, "type": event_type, **fields}
        self.next_event_id += 1
        self.events.append(event)
        return event

    def add_safe_summary(self, **fields: Any) -> None:
        self.safe_history.append(fields)

    def prompt_history(
        self, *, max_events: int = 6, max_chars: int = 24000
    ) -> list[dict[str, Any]]:
        """Keep recent trajectory state within a deterministic total budget."""
        selected: list[dict[str, Any]] = []
        used = 0
        for event in reversed(self.safe_history[-max_events:]):
            size = len(json.dumps(event, ensure_ascii=False, default=str))
            if used + size > max_chars:
                continue
            selected.append(event)
            used += size
        selected.reverse()
        return selected
