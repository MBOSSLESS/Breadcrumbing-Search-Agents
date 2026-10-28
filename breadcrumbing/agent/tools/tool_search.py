# Derived from Tongyi DeepResearch (Apache-2.0); modified for this project.
# See NOTICE.md and licenses/DeepResearch-APACHE-2.0.txt.
import json
from typing import List, Union
from qwen_agent.tools.base import BaseTool, register_tool
from typing import Dict, List, Optional, Union
import http.client
import json

import os


# Import cache manager
from breadcrumbing.cache import cached_search
from breadcrumbing.runtime_trace import trace_span

import logging

_logger = logging.getLogger(
    __name__
)  # Module-private logger (not exported by import *)


SERPER_KEY = os.environ.get("SERPER_KEY_ID")


@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of query strings. Include multiple complementary search queries in a single call.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self.attacker = None
        self.last_call_metadata: Dict[
            str,
            Union[
                str, int, List[Dict[str, Union[str, int, bool, None]]], Dict[str, int]
            ],
        ] = {}

    def set_attacker(self, attacker):
        """Set attacker for response hijacking."""
        self.attacker = attacker

    def _reset_metadata(self) -> None:
        self.last_call_metadata = {
            "tool_name": "search",
            "calls": [],
            "aggregate": {
                "query_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "cache_hit_count": 0,
                "results_count_total": 0,
            },
        }

    @cached_search
    def _search_api_call(self, query: str):
        """
        Minimal API call layer - only caches the raw API response.
        Returns: dict (raw JSON response from Serper API)
        """

        def contains_chinese_basic(text: str) -> bool:
            return any("\u4e00" <= char <= "\u9fff" for char in text)

        conn = http.client.HTTPSConnection("google.serper.dev")
        if contains_chinese_basic(query):
            payload = json.dumps(
                {"q": query, "location": "China", "gl": "cn", "hl": "zh-cn"}
            )

        else:
            payload = json.dumps(
                {"q": query, "location": "United States", "gl": "us", "hl": "en"}
            )
        headers = {"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"}

        for i in range(5):
            try:
                conn.request("POST", "/search", payload, headers)
                res = conn.getresponse()
                break
            except Exception as e:
                _logger.debug(f"Search API error (retrying): {e}")
                if i == 4:
                    return None
                continue

        data = res.read()
        results = json.loads(data.decode("utf-8"))
        return results

    def google_search_with_serp(self, query: str):
        """
        Wrapper that calls cached API and processes the result.
        Post-processing logic (formatting) is NOT cached.
        """
        cache_hit = self._was_search_cache_hit(query)
        with trace_span("tool.backend", tool_name="search", cache_hit=cache_hit):
            results = self._search_api_call(query)
        # Apply attacker hook
        if self.attacker:
            with trace_span(
                "attacker.hook",
                role="attacker",
                method=getattr(self.attacker, "mode", type(self.attacker).__name__),
                event_type="search",
            ):
                results = self.attacker.hook_search(query, results)

        if results is None:
            self._record_search_metadata(
                query, cache_hit=cache_hit, results=None, succeeded=False
            )
            return f"Google search Timeout, return None, Please try again later."

        try:
            if "organic" not in results:
                raise Exception(
                    f"No results found for query: '{query}'. Use a less specific query."
                )

            web_snippets = list()
            idx = 0
            if "organic" in results:
                for page in results["organic"]:
                    idx += 1
                    date_published = ""
                    if "date" in page:
                        date_published = "\nDate published: " + page["date"]

                    source = ""
                    if "source" in page:
                        source = "\nSource: " + page["source"]

                    snippet = ""
                    if "snippet" in page:
                        snippet = "\n" + page["snippet"]

                    redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                    redacted_version = redacted_version.replace(
                        "Your browser can't play this video.", ""
                    )
                    web_snippets.append(redacted_version)

            self._record_search_metadata(
                query, cache_hit=cache_hit, results=results, succeeded=True
            )
            content = (
                f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n"
                + "\n\n".join(web_snippets)
            )
            return content
        except:
            self._record_search_metadata(
                query, cache_hit=cache_hit, results=results, succeeded=False
            )
            return f"No results found for '{query}'. Try with a more general query."

    def search_with_serp(self, query: str):
        result = self.google_search_with_serp(query)
        return result

    def call(self, params: Union[str, dict], **kwargs) -> str:
        self._reset_metadata()
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"

        if isinstance(query, str):
            # 单个查询
            response = self.search_with_serp(query)
        else:
            # 多个查询
            assert isinstance(query, List)
            responses = []
            for q in query:
                responses.append(self.search_with_serp(q))
            response = "\n=======\n".join(responses)

        return response

    def _was_search_cache_hit(self, query: str) -> bool:
        try:
            from breadcrumbing.cache import _search_cache

            key = _search_cache._generate_key("_search_api_call", (self, query), {})
            return _search_cache.get(key) is not None
        except Exception:
            return False

    def _record_search_metadata(
        self, query: str, cache_hit: bool, results: Optional[dict], succeeded: bool
    ) -> None:
        organic_results = 0
        if isinstance(results, dict):
            organic = results.get("organic")
            if isinstance(organic, list):
                organic_results = len(organic)

        call_meta = {
            "query": query,
            "cache_hit": cache_hit,
            "succeeded": succeeded,
            "results_count": organic_results,
        }
        self.last_call_metadata["calls"].append(call_meta)

        aggregate = self.last_call_metadata["aggregate"]
        aggregate["query_count"] += 1
        if succeeded:
            aggregate["success_count"] += 1
        else:
            aggregate["failure_count"] += 1
        if cache_hit:
            aggregate["cache_hit_count"] += 1
        aggregate["results_count_total"] += organic_results
