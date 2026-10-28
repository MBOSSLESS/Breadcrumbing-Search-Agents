from typing import List, Dict, Optional, Any
from ..schema import Event, Decision
import time


# memory.py


class Memory:
    def __init__(self):
        self.events: list[Event] = []
        # Track injected URLs: {url: {query, decision, timestamp}}
        self._injected_data: Dict[str, Dict[str, Any]] = {}
        # Track attackable follow-up URLs planned by the attacker.
        self._target_data: Dict[str, Dict[str, Any]] = {}
        # Track URLs that must always remain read-only.
        # These come from real search results or prior non-attack observations.
        self._real_url_data: Dict[str, Dict[str, Any]] = {}

    def add_event(self, event: Event):
        self.events.append(event)

    def summary(self, include_reflection: bool = True) -> str:
        if not self.events:
            return "No history."

        lines = ["History:"]

        for i, e in enumerate(self.events, 1):
            attack = e.decision.get("attack", False)
            reflection = e.decision.get("reflection")

            if e.event_type == "search":
                query = e.input.get("query", "")
                line = f'[{i}] search query="{query}" attack={attack}'

                payload = e.decision.get("payload")
                if attack and payload:
                    link = payload.get("link", "")
                    if link:
                        line += f' target="{link}"'

            elif e.event_type == "visit":
                url = e.input.get("url", "")
                line = f'[{i}] visit url="{url}" attack={attack}'

                # 🔹 如果是 injected 来源，增强上下文语义
                injected_meta = self._injected_data.get(url)

                if injected_meta:
                    source_query = injected_meta.get("query", "")
                    if source_query:
                        line += f' from_injected_query="{source_query}"'

                payload = e.decision.get("payload")
                if attack and payload:
                    page_content = payload.get("page_content")
                    if page_content:
                        line += f" generated_len={len(page_content)}"

            else:
                line = f"[{i}] {e.event_type} attack={attack}"

            if include_reflection and reflection:
                MAX_REFLECTION_FULL = 5
                MAX_REFLECTION_CHARS = 300

                if i <= MAX_REFLECTION_FULL or len(reflection) <= MAX_REFLECTION_CHARS:
                    short_reflection = reflection
                else:
                    ellipsis = "..."
                    half = (MAX_REFLECTION_CHARS - len(ellipsis)) // 2
                    short_reflection = reflection[:half] + ellipsis + reflection[-half:]

                line += f' reflection="{short_reflection}"'

            lines.append(line)

        return "\n".join(lines)

    # =========================
    # Injected data management
    # =========================

    def add_injected(self, url: str, query: str, decision: Decision) -> None:
        """
        Record an injected URL with its associated query and decision.

        Args:
            url: The injected URL (from payload.link)
            query: The search query that triggered this injection
            decision: The Decision object containing payload and reflection
        """
        self._injected_data[url] = {
            "query": query,
            "decision": decision.to_dict(),
            "timestamp": time.time(),
        }

    def add_target(self, url: str, query: str, decision: Decision) -> None:
        """
        Record a follow-up URL that the attacker expects the agent to visit later.

        Args:
            url: The follow-up URL to mark as attackable
            query: The search query that established this attack path
            decision: The Decision object containing payload and reflection
        """
        self._target_data[url] = {
            "query": query,
            "decision": decision.to_dict(),
            "timestamp": time.time(),
        }

    def add_real_url(self, url: str, source: str) -> None:
        """
        Record a URL that must remain read-only.

        Args:
            url: Real URL observed from organic search results or non-attack visit flow
            source: Why the URL is considered real (e.g. organic_search, observed_visit)
        """
        if not url:
            return

        existing = self._real_url_data.get(url)
        if existing is None:
            self._real_url_data[url] = {
                "first_source": source,
                "timestamp": time.time(),
            }
            return

        # Preserve the first observation source/timestamp to keep semantics stable.
        existing.setdefault("first_source", source)
        existing.setdefault("timestamp", time.time())

    def get_injected(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve injection data for a given URL.

        Args:
            url: The URL to look up

        Returns:
            Dict with 'query', 'decision', 'timestamp' if found, None otherwise
        """
        return self._injected_data.get(url)

    def get_target(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve planned follow-up attack data for a given URL.
        """
        return self._target_data.get(url)

    def get_attackable(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve attack context for a URL.

        Attackable URLs include:
        - directly injected search results
        - follow-up target URLs explicitly planned by prior decisions
        """
        injected = self.get_injected(url)
        if injected is not None:
            return injected
        return self.get_target(url)

    def is_real_url(self, url: str) -> bool:
        """
        Check whether a URL is protected as a real, read-only page.
        """
        return url in self._real_url_data

    def real_urls(self) -> List[str]:
        """Return all observed real URLs."""
        return list(self._real_url_data.keys())

    def get_injected_by_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Find all injections associated with a specific query.

        Args:
            query: The search query to look up

        Returns:
            List of dicts with 'url', 'decision', 'timestamp'
        """
        results = []
        for url, data in self._injected_data.items():
            if data["query"] == query:
                results.append(
                    {
                        "url": url,
                        "decision": data["decision"],
                        "timestamp": data["timestamp"],
                    }
                )
        return results

    def list_injected_urls(self) -> List[str]:
        """
        Get a list of all injected URLs.

        Returns:
            List of URLs that have been injected
        """
        return list(self._injected_data.keys())

    def list_target_urls(self) -> List[str]:
        """
        Get a list of planned follow-up target URLs.

        Returns:
            List of URLs that are attackable but not necessarily injected in search
        """
        return list(self._target_data.keys())

    def list_real_urls(self) -> List[str]:
        """
        Get a list of URLs that must remain read-only.
        """
        return list(self._real_url_data.keys())
