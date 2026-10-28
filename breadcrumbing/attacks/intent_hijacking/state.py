"""Per-rollout state for Intent Hijacking Search."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .schema import FixedStrategy, MatchedCaseContext, SyntheticAsset, bounded_text


@dataclass
class IntentHijackingSearchState:
    case: MatchedCaseContext
    stage: int = 1
    strategy: FixedStrategy | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    assets: dict[str, SyntheticAsset] = field(default_factory=dict)
    real_urls: set[str] = field(default_factory=set)
    real_hosts: set[str] = field(default_factory=set)
    fixed_source_name: str = ""
    fixed_source_host: str = ""
    planner_calls: int = 0
    generator_calls: int = 0
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    last_injection_event_id: int = 0
    next_event_id: int = 1

    def add_event(self, event_type: str, **fields: Any) -> dict[str, Any]:
        event = {"id": self.next_event_id, "type": event_type, **fields}
        self.next_event_id += 1
        self.events.append(event)
        return event

    def delta_since_last_injection(self) -> list[dict[str, Any]]:
        return [
            event
            for event in self.events
            if int(event.get("id", 0)) > self.last_injection_event_id
        ]

    @staticmethod
    def _bounded_events(
        events: list[dict[str, Any]],
        max_chars: int,
    ) -> list[dict[str, Any]]:
        if max_chars <= 0 or not events:
            return []
        # Preserve structural metadata for every event. Only the content-bearing
        # observation/output fields compete for the bounded history budget.
        views: list[dict[str, Any]] = []
        contents: list[dict[str, Any]] = []
        for event in events:
            view = {
                key: deepcopy(value)
                for key, value in event.items()
                if key not in {"observation", "output", "meta"}
            }
            meta = event.get("meta")
            if isinstance(meta, dict):
                compact_meta = {
                    key: deepcopy(value)
                    for key, value in meta.items()
                    if key != "llm_calls"
                }
                if compact_meta:
                    view["meta"] = compact_meta
            views.append(view)
            contents.append(
                {
                    key: deepcopy(event[key])
                    for key in ("observation", "output")
                    if key in event
                }
            )

        metadata_chars = len(json.dumps(views, ensure_ascii=False, default=str))
        remaining = max(0, max_chars - metadata_chars)
        for index in range(len(views) - 1, -1, -1):
            if remaining <= 0 or not contents[index]:
                continue
            serialized = json.dumps(
                contents[index],
                ensure_ascii=False,
                default=str,
            )
            excerpt = bounded_text(
                serialized,
                remaining,
                "\n[... bounded event content omitted ...]\n",
            )
            views[index]["content_excerpt"] = excerpt
            remaining = max(0, remaining - len(excerpt))
        return views

    def prompt_history(self, max_chars: int) -> list[dict[str, Any]]:
        return self._bounded_events(self.events, max_chars)

    def prompt_delta(self, max_chars: int) -> list[dict[str, Any]]:
        return self._bounded_events(
            self.delta_since_last_injection(),
            max_chars,
        )
