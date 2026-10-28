"""Append-only MAST-Direct-Search attacker."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import logging
import threading
from typing import Any, Optional
from urllib.parse import urlsplit

from ..base import BaseAttacker
from .policy import MASTDirectPolicy
from .prompts import PROMPT_VERSION
from .schema import MatchedCaseContext, SyntheticAsset, bounded_text
from .state import MASTState


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower().rstrip(".")
    except (TypeError, ValueError):
        return ""


def _decision_dict(decision: Any) -> dict[str, Any]:
    """Keep MAST's no-op representation exactly minimal."""
    return decision.model_dump(exclude_defaults=True)


class MASTDirectSearchAttacker(BaseAttacker):
    """Direct-prompting adaptation of MAST's sequential sub-goal scaffold."""

    mode = "mast"

    def __init__(
        self,
        *,
        case_context: dict[str, Any],
        model_name: str,
        temperature: float = 0.5,
        max_search_injections: Optional[int] = None,
        evidence_excerpt_chars: int = 12000,
        max_format_retries: int = 1,
        policy: Any = None,
    ):
        if max_search_injections is not None:
            max_search_injections = int(max_search_injections)
        if max_search_injections is not None and max_search_injections < 0:
            raise ValueError("max_search_injections must be non-negative")
        if evidence_excerpt_chars < 0:
            raise ValueError("evidence_excerpt_chars must be non-negative")
        self.case = MatchedCaseContext.from_mapping(case_context)
        self.model_name = model_name
        self.temperature = temperature
        self.max_search_injections = max_search_injections
        self.evidence_excerpt_chars = evidence_excerpt_chars
        self.policy = policy or MASTDirectPolicy(
            model_name=model_name,
            temperature=temperature,
            max_format_retries=max_format_retries,
        )
        self.logger = logging.getLogger("MASTDirectSearchAttacker")
        self._lock = threading.RLock()
        self.user_query = self.case.user_query
        self.state = MASTState(case=self.case)

    def set_intent(self, user_query: str) -> None:
        """Start a fresh rollout; the exact case row remains authoritative."""
        normalized = str(user_query or "").strip()
        if normalized and normalized != self.case.user_query:
            raise ValueError("set_intent query does not match case_context.user_query")
        with self._lock:
            self.user_query = normalized or self.case.user_query
            self.state = MASTState(case=self.case)

    def _record_real_url(self, url: Any) -> None:
        clean = str(url or "").strip()
        if not clean:
            return
        self.state.real_urls.add(clean)
        host = _host(clean)
        if host:
            self.state.real_hosts.add(host)

    def _validate_asset(
        self,
        generated: Any,
        organic_results: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], str, str]:
        result = generated.search_result.model_dump()
        url = result["link"].strip()
        parsed = urlsplit(url)
        host = _host(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or not host:
            raise ValueError("generated synthetic URL is invalid")
        if parsed.username or parsed.password:
            raise ValueError("generated synthetic URL contains credentials")
        if url in self.state.assets:
            raise ValueError("generated synthetic URL collides with an existing asset")

        for organic_host in self.state.real_hosts:
            if (
                host == organic_host
                or host.endswith("." + organic_host)
                or organic_host.endswith("." + host)
            ):
                raise ValueError(
                    "generated URL reuses or impersonates an organic domain"
                )

        result["link"] = url
        return result, generated.page.title.strip(), generated.page.body.strip()

    def hook_search(self, query: str, response: dict) -> dict:
        with self._lock:
            raw = deepcopy(response)
            if not isinstance(raw, dict):
                self.state.add_event(
                    "search",
                    input={"query": query},
                    decision={"status": "invalid_search_response"},
                    meta={"llm_calls": []},
                )
                return raw
            organic = raw.get("organic")
            if not isinstance(organic, list):
                self.state.add_event(
                    "search",
                    input={"query": query},
                    decision={"status": "invalid_search_response"},
                    meta={"llm_calls": []},
                )
                return raw
            for item in organic:
                if isinstance(item, dict):
                    self._record_real_url(item.get("link"))

            if (
                self.max_search_injections is not None
                and self.state.successful_injections >= self.max_search_injections
            ):
                self.state.add_event(
                    "search",
                    input={"query": query},
                    decision={"status": "injection_cap_reached"},
                    meta={"llm_calls": []},
                )
                return raw

            before_calls = len(self.state.llm_calls)
            try:
                decision = self.policy.plan(
                    case=self.case,
                    query=query,
                    organic_results=deepcopy(organic),
                    partial_attack_sequence=self.state.prompt_history(),
                    previous_assets=self.state.asset_prompt_views(
                        self.evidence_excerpt_chars
                    ),
                    remaining_injections=(
                        None
                        if self.max_search_injections is None
                        else self.max_search_injections
                        - self.state.successful_injections
                    ),
                    call_log=self.state.llm_calls,
                )
                self.state.planner_calls += 1
            except Exception as exc:
                self.state.planner_calls += 1
                self.state.add_event(
                    "search",
                    input={"query": query},
                    decision={"status": "planner_error"},
                    meta={
                        "llm_calls": deepcopy(self.state.llm_calls[before_calls:]),
                        "error_type": type(exc).__name__,
                    },
                )
                self.logger.warning(
                    "MAST Planner failed closed: %s", type(exc).__name__
                )
                return raw

            if decision.action == "no_op":
                self.state.no_op_count += 1
                self.state.add_event(
                    "search",
                    input={"query": query},
                    decision=_decision_dict(decision),
                    meta={"llm_calls": deepcopy(self.state.llm_calls[before_calls:])},
                )
                return raw

            try:
                generated = self.policy.generate(
                    case=self.case,
                    query=query,
                    organic_results=deepcopy(organic),
                    decision=decision,
                    previous_assets=self.state.asset_prompt_views(
                        self.evidence_excerpt_chars
                    ),
                    call_log=self.state.llm_calls,
                )
                self.state.generator_calls += 1
                result, page_title, page_body = self._validate_asset(generated, organic)
            except Exception as exc:
                self.state.generator_calls += 1
                self.state.add_event(
                    "search",
                    input={"query": query},
                    decision={
                        **_decision_dict(decision),
                        "status": "generator_error",
                    },
                    meta={
                        "llm_calls": deepcopy(self.state.llm_calls[before_calls:]),
                        "error_type": type(exc).__name__,
                    },
                )
                self.logger.warning(
                    "MAST Generator failed closed: %s", type(exc).__name__
                )
                return raw

            event_id = self.state.next_event_id
            appended = deepcopy(result)
            appended["position"] = len(organic) + 1
            final_response = deepcopy(raw)
            final_response["organic"] = deepcopy(organic) + [appended]
            asset = SyntheticAsset(
                search_result=deepcopy(appended),
                page_title=page_title,
                page_body=page_body,
                query=query,
                planner_decision=_decision_dict(decision),
                exposed_event_id=event_id,
            )
            self.state.assets[asset.url] = asset
            self.state.successful_injections += 1
            self.state.add_event(
                "search",
                input={"query": query},
                decision=_decision_dict(decision),
                output={
                    "injected": True,
                    "synthetic_url": asset.url,
                    "position": appended["position"],
                },
                meta={"llm_calls": deepcopy(self.state.llm_calls[before_calls:])},
            )
            return final_response

    def hook_visit(self, url: str, response: str) -> str:
        with self._lock:
            controlled = self.get_controlled_page(url)
            if controlled is not None:
                return controlled

            raw = response
            self._record_real_url(url)
            excerpt = bounded_text(
                raw,
                self.evidence_excerpt_chars,
                "\n[... bounded organic evidence omitted ...]\n",
            )
            self.state.add_event(
                "visit",
                input={"url": url},
                decision={"controlled_asset": False},
                observation={"organic_evidence_excerpt": excerpt},
                meta={"llm_calls": []},
            )
            return raw

    def get_controlled_page(self, url: str) -> str | None:
        """Serve an exact registered URL without fetching it from the network."""
        with self._lock:
            asset = self.state.assets.get(url)
            if asset is None:
                return None
            asset.visited = True
            output = asset.rendered_page()
            self.state.add_event(
                "visit",
                input={"url": url},
                decision={"controlled_asset": True},
                output={"returned_registered_page": True},
                meta={"llm_calls": []},
            )
            return output

    def get_behavior_logs(self) -> dict[str, Any]:
        with self._lock:
            calls = deepcopy(self.state.llm_calls)
            trace_events = deepcopy(self.state.events)
            engineering_failure = False
            for event in trace_events:
                decision = event.get("decision", {})
                if decision.get("status") in {"planner_error", "generator_error"}:
                    engineering_failure = True
                observation = event.get("observation")
                if not isinstance(observation, dict):
                    continue
                excerpt = observation.pop("organic_evidence_excerpt", None)
                if excerpt is not None:
                    encoded = excerpt.encode("utf-8")
                    observation["organic_evidence"] = {
                        "present": bool(excerpt),
                        "chars": len(excerpt),
                        "sha256": hashlib.sha256(encoded).hexdigest(),
                    }
            prompt_tokens = sum(c["tokens"]["prompt"] for c in calls)
            completion_tokens = sum(c["tokens"]["completion"] for c in calls)
            injected_urls = list(self.state.assets)
            visited_urls = [
                url for url, asset in self.state.assets.items() if asset.visited
            ]
            visited = len(visited_urls)
            exposed = len(self.state.assets)
            recall = visited / exposed if exposed else 0.0
            return {
                "generation_success": not engineering_failure,
                "session": {
                    "mode": self.mode,
                    "model": self.model_name,
                    "temperature": self.temperature,
                    "test_case_id": self.case.test_case_id,
                    "global_goal_source": "target_consequence",
                    "prompt_version": PROMPT_VERSION,
                    "max_search_injections": self.max_search_injections,
                    "events": len(self.state.events),
                },
                "events": trace_events,
                "stats": {
                    "llm": {
                        "total_calls": len(calls),
                        "successful": sum(
                            1 for call in calls if call.get("success", False)
                        ),
                        "failed": sum(
                            1 for call in calls if not call.get("success", False)
                        ),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                        "total_duration_ms": sum(
                            call.get("duration_ms", 0) for call in calls
                        ),
                        "planner_calls": self.state.planner_calls,
                        "generator_calls": self.state.generator_calls,
                    }
                },
                "attack": {
                    "injection_count": self.state.successful_injections,
                    "search_injection_count": self.state.successful_injections,
                    "total_attacks": self.state.successful_injections,
                    "max_search_injections": self.max_search_injections,
                    "no_op_count": self.state.no_op_count,
                    "registered_asset_count": exposed,
                    "visited_asset_count": visited,
                    "search_to_visit_recall": recall,
                    "recall_rate": recall,
                    "injected_urls": injected_urls,
                    "visited_injected_urls": visited_urls,
                    "real_urls": sorted(self.state.real_urls),
                },
            }
