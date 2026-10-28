# Derived from Tongyi DeepResearch (Apache-2.0); modified for this project.
# See NOTICE.md and licenses/DeepResearch-APACHE-2.0.txt.
import json
import os
import re
from typing import Any, Dict, List, Union, Optional
import requests
from qwen_agent.tools.base import BaseTool, register_tool
from breadcrumbing.agent.prompt import EXTRACTOR_PROMPT
import time

from breadcrumbing.models import unified_client
from breadcrumbing.cache import cached_visit
from breadcrumbing.runtime_trace import trace_span


import logging

_logger = logging.getLogger(
    __name__
)  # Module-private logger (not exported by import *)

VISIT_SERVER_TIMEOUT = int(os.getenv("VISIT_SERVER_TIMEOUT", 200))
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))

JINA_API_KEYS = os.getenv("JINA_API_KEYS", "")
JINA_TOKEN_BUDGET_ENABLED = os.getenv("JINA_TOKEN_BUDGET_ENABLED", "1").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
JINA_TOKEN_BUDGET = int(os.getenv("JINA_TOKEN_BUDGET", 45000))
JINA_RESPOND_WITH = os.getenv("JINA_RESPOND_WITH", "markdown")
JINA_RETAIN_IMAGES = os.getenv("JINA_RETAIN_IMAGES", "none")
JINA_NO_CACHE = os.getenv("JINA_NO_CACHE", "0").lower() in {"1", "true", "yes", "on"}


@staticmethod
def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    # Simple character-based truncation (rough approximation for tokens)
    # Assuming ~4 chars per token on average
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


OSS_JSON_FORMAT = """# Response Formats
## visit_content
{"properties":{"rational":{"type":"string","description":"Locate the **specific sections/data** directly related to the user's goal within the webpage content"},"evidence":{"type":"string","description":"Identify and extract the **most relevant information** from the content, never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.","summary":{"type":"string","description":"Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal."}}}}"""


@register_tool("visit", allow_overwrite=True)
class Visit(BaseTool):
    # The `description` tells the agent the functionality of this tool.
    name = "visit"
    description = "Visit webpage(s) and return the summary of the content."
    # The `parameters` tell the agent what input parameters the tool has.
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "minItems": 1,
                "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs.",
            },
            "goal": {
                "type": "string",
                "description": "The goal of the visit for webpage(s).",
            },
        },
        "required": ["url", "goal"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self.attacker = None
        self.last_call_metadata: Dict[str, Any] = {}

    def set_attacker(self, attacker):
        """Set attacker for response hijacking."""
        self.attacker = attacker

    # The `call` method is the main function of the tool.
    def call(self, params: Union[str, dict], **kwargs) -> str:
        self.last_call_metadata = {
            "tool_name": "visit",
            "calls": [],
            "aggregate": {
                "url_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "usage_tokens_total": 0,
                "balance_error_count": 0,
                "budget_error_count": 0,
                "cache_hit_count": 0,
            },
        }
        try:
            url = params["url"]
            goal = params["goal"]
            model_name = params.get("model_name", kwargs.get("model_name", "qwen3"))
        except:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

        start_time = time.time()

        if isinstance(url, str):
            response = self.readpage_jina(url, goal, model_name)
        else:
            response = []
            assert isinstance(url, List)
            start_time = time.time()
            for u in url:
                if time.time() - start_time > 900:
                    cur_response = "The useful information in {url} for user goal {goal} as follows: \n\n".format(
                        url=url, goal=goal
                    )
                    cur_response += (
                        "Evidence in page: \n"
                        + "The provided webpage content could not be accessed. Please check the URL or file format."
                        + "\n\n"
                    )
                    cur_response += (
                        "Summary: \n"
                        + "The webpage content could not be processed, and therefore, no information is available."
                        + "\n\n"
                    )
                else:
                    try:
                        cur_response = self.readpage_jina(u, goal, model_name)
                    except Exception as e:
                        cur_response = f"Error fetching {u}: {str(e)}"
                response.append(cur_response)
            response = "\n=======\n".join(response)

        aggregate = self.last_call_metadata["aggregate"]
        if self.last_call_metadata["calls"]:
            aggregate["url_count"] = len(self.last_call_metadata["calls"])

        return response.strip()

    def call_server(self, msgs, model_name="qwen3", max_retries=2):

        for attempt in range(max_retries):
            try:
                chat_response = unified_client.chat_completions_create(
                    model_name=model_name, messages=msgs, temperature=0.7
                )
                choices = getattr(chat_response, "choices", None)
                if not choices:
                    content = ""
                else:
                    message = getattr(choices[0], "message", None)
                    content = getattr(message, "content", None) or ""
                if content:
                    try:
                        json.loads(content)
                    except:
                        # extract json from string
                        left = content.find("{")
                        right = content.rfind("}")
                        if left != -1 and right != -1 and left <= right:
                            content = content[left : right + 1]
                    return content
            except Exception as e:
                if attempt == (max_retries - 1):
                    return ""
                continue
        return ""

    @cached_visit
    def _jina_fetch_raw(
        self, url: str, token_budget: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Minimal API call layer - only caches the raw webpage fetch.
        Returns a structured payload containing raw webpage content and metadata.
        """
        max_retries = 3
        timeout = 50

        for attempt in range(max_retries):
            headers = {
                "X-Respond-With": JINA_RESPOND_WITH,
                "X-Retain-Images": JINA_RETAIN_IMAGES,
                "X-Timeout": str(min(timeout, 180)),
            }
            if JINA_API_KEYS:
                headers["Authorization"] = f"Bearer {JINA_API_KEYS}"
            if token_budget is not None:
                headers["X-Token-Budget"] = str(token_budget)
            if JINA_NO_CACHE:
                headers["X-No-Cache"] = "true"
            try:
                response = requests.get(
                    f"https://r.jina.ai/{url}", headers=headers, timeout=timeout
                )
                parsed = self._build_jina_result(
                    url, response, attempt + 1, timeout, token_budget
                )
                if not parsed["is_error"]:
                    return parsed
                if self._should_retry_jina_error(parsed):
                    time.sleep(0.5)
                    continue
                return parsed
            except Exception as e:
                time.sleep(0.5)
                if attempt == max_retries - 1:
                    return {
                        "url": url,
                        "content": "[visit] Failed to read page.",
                        "is_error": True,
                        "http_status_code": None,
                        "usage_tokens": None,
                        "error_type": "request_exception",
                        "error_name": type(e).__name__,
                        "error_status": None,
                        "error_code": None,
                        "error_message": str(e),
                        "error_readable_message": str(e),
                        "raw_preview": str(e)[:300],
                        "attempt": attempt + 1,
                        "token_budget": token_budget,
                    }

        return {
            "url": url,
            "content": "[visit] Failed to read page.",
            "is_error": True,
            "http_status_code": None,
            "usage_tokens": None,
            "error_type": "request_failed",
            "error_name": "RequestFailed",
            "error_status": None,
            "error_code": None,
            "error_message": "Jina request failed after retries.",
            "error_readable_message": "Jina request failed after retries.",
            "raw_preview": "",
            "attempt": max_retries,
            "token_budget": token_budget,
        }

    def jina_readpage(self, url: str) -> Dict[str, Any]:
        """
        Read webpage content using Jina service.
        Wrapper that calls cached API fetch.
        No additional post-processing here, so it just delegates.

        Args:
            url: The URL to read

        Returns:
            str: The webpage content or error message
        """

        token_budget = self._current_jina_token_budget()
        controlled_getter = getattr(self.attacker, "get_controlled_page", None)
        if callable(controlled_getter):
            with trace_span(
                "attacker.hook",
                role="attacker",
                method=getattr(self.attacker, "mode", type(self.attacker).__name__),
                event_type="visit_prefetch",
            ):
                controlled_content = controlled_getter(url)
            if controlled_content is not None:
                return {
                    "url": url,
                    "content": controlled_content,
                    "is_error": False,
                    "http_status_code": None,
                    "usage_tokens": None,
                    "error_type": None,
                    "error_name": None,
                    "error_status": None,
                    "error_code": None,
                    "error_message": None,
                    "error_readable_message": None,
                    "raw_preview": "",
                    "attempt": 0,
                    "token_budget": token_budget,
                    "cache_hit": False,
                    "attacker_hook_applied": True,
                    "attacker_prefetch_bypassed": True,
                }
        cache_hit = self._was_visit_cache_hit(url, token_budget)
        with trace_span("tool.backend", tool_name="visit", cache_hit=cache_hit):
            content = self._jina_fetch_raw(url, token_budget=token_budget)
        content["cache_hit"] = cache_hit
        # Apply attacker hook
        if self.attacker:
            with trace_span(
                "attacker.hook",
                role="attacker",
                method=getattr(self.attacker, "mode", type(self.attacker).__name__),
                event_type="visit",
            ):
                content["content"] = self.attacker.hook_visit(url, content["content"])
            content["attacker_hook_applied"] = True
        else:
            content["attacker_hook_applied"] = False
        return content

    def html_readpage_jina(self, url: str) -> str:
        max_attempts = 8
        for attempt in range(max_attempts):
            content = self.jina_readpage(url)
            service = "jina"
            if (
                isinstance(content, dict)
                and content.get("is_error")
                and content.get("error_type")
                in {"balance_error", "budget_error", "jina_api_error"}
            ):
                return content
            if (
                content
                and not content["content"].startswith("[visit] Failed to read page.")
                and content["content"] != "[visit] Empty content."
                and not content["content"].startswith("[document_parser]")
            ):
                return content
        return "[visit] Failed to read page."

    def readpage_jina(self, url: str, goal: str, model_name: str = "qwen3") -> str:
        """
        Read webpage content through Jina and summarize it for the visit goal.

        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page
            model_name: Model name in "backend:model" format

        Returns:
            str: The webpage content or error message
        """

        max_retries = int(os.getenv("VISIT_SERVER_MAX_RETRIES", 1))

        content = self.html_readpage_jina(url)

        if (
            isinstance(content, dict)
            and content
            and not content["content"].startswith("[visit] Failed to read page.")
            and content["content"] != "[visit] Empty content."
            and not content["content"].startswith("[document_parser]")
        ):
            self._record_visit_metadata(content, succeeded=True)

            enable_summary = os.getenv("ENABLE_VISIT_SUMMARY", "0").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if not enable_summary:
                # Preserve the current default path: return raw visited content directly.
                result = content["content"][:WEBCONTENT_MAXLENGTH]
                return result

            # -------- original summary logic below (kept for future use) --------

            content = content["content"]
            content = truncate_to_tokens(content, max_tokens=95000)
            messages = [
                {
                    "role": "user",
                    "content": EXTRACTOR_PROMPT.format(
                        webpage_content=content, goal=goal
                    ),
                }
            ]
            parse_retry_times = 0
            raw = self.call_server(
                messages, model_name=model_name, max_retries=max_retries
            )
            summary_retries = 3
            while len(raw) < 10 and summary_retries >= 0:
                truncate_length = (
                    int(0.7 * len(content)) if summary_retries > 0 else 25000
                )
                status_msg = (
                    (
                        f"[visit] Summary url[{url}] "
                        f"attempt {3 - summary_retries + 1}/3, "
                        f"content length: {len(content)}, "
                        f"truncating to {truncate_length} chars"
                    )
                    if summary_retries > 0
                    else (
                        f"[visit] Summary url[{url}] failed after 3 attempts, "
                        f"final truncation to 25000 chars"
                    )
                )
                _logger.info(status_msg)
                content = content[:truncate_length]
                extraction_prompt = EXTRACTOR_PROMPT.format(
                    webpage_content=content, goal=goal
                )
                messages = [{"role": "user", "content": extraction_prompt}]
                raw = self.call_server(
                    messages, model_name=model_name, max_retries=max_retries
                )
                summary_retries -= 1

            parse_retry_times = 0
            if isinstance(raw, str):
                raw = raw.replace("```json", "").replace("```", "").strip()
            while parse_retry_times < 3:
                try:
                    raw = json.loads(raw)
                    break
                except:
                    raw = self.call_server(
                        messages, model_name=model_name, max_retries=max_retries
                    )
                    parse_retry_times += 1

            if parse_retry_times >= 3:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(
                    url=url, goal=goal
                )
                useful_information += (
                    "Evidence in page: \n"
                    + "The provided webpage content could not be accessed. Please check the URL or file format."
                    + "\n\n"
                )
                useful_information += (
                    "Summary: \n"
                    + "The webpage content could not be processed, and therefore, no information is available."
                    + "\n\n"
                )
            else:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(
                    url=url, goal=goal
                )
                useful_information += (
                    "Evidence in page: \n" + str(raw["evidence"]) + "\n\n"
                )
                useful_information += "Summary: \n" + str(raw["summary"]) + "\n\n"

            if len(useful_information) < 10 and summary_retries < 0:
                _logger.warning(
                    "[visit] Could not generate valid summary after maximum retries"
                )
                useful_information = "[visit] Failed to read page"

            return useful_information

        # If no valid content was obtained after all retries
        else:
            if isinstance(content, dict):
                self._record_visit_metadata(content, succeeded=False)
            else:
                self._record_visit_metadata(
                    {
                        "url": url,
                        "content": "[visit] Failed to read page.",
                        "is_error": True,
                        "http_status_code": None,
                        "usage_tokens": None,
                        "error_type": "visit_failed",
                        "error_name": "VisitFailed",
                        "error_status": None,
                        "error_code": None,
                        "error_message": "Visit failed after retries.",
                        "error_readable_message": "Visit failed after retries.",
                        "raw_preview": "",
                        "cache_hit": False,
                    },
                    succeeded=False,
                )
            useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(
                url=url, goal=goal
            )
            useful_information += (
                "Evidence in page: \n"
                + "The provided webpage content could not be accessed. Please check the URL or file format."
                + "\n\n"
            )
            useful_information += (
                "Summary: \n"
                + "The webpage content could not be processed, and therefore, no information is available."
                + "\n\n"
            )
            return useful_information

    def _build_jina_result(
        self,
        url: str,
        response: requests.Response,
        attempt: int,
        timeout: int,
        token_budget: Optional[int] = None,
    ) -> Dict[str, Any]:
        text = response.text or ""
        usage_tokens = self._parse_int(response.headers.get("x-usage-tokens"))
        error_info = self._extract_jina_error(text)
        is_error = response.status_code != 200 or error_info is not None

        result: Dict[str, Any] = {
            "url": url,
            "content": text if not is_error else "[visit] Failed to read page.",
            "is_error": is_error,
            "http_status_code": response.status_code,
            "usage_tokens": usage_tokens,
            "error_type": None,
            "error_name": None,
            "error_status": None,
            "error_code": None,
            "error_message": None,
            "error_readable_message": None,
            "raw_preview": text[:500],
            "attempt": attempt,
            "timeout_seconds": timeout,
            "token_budget": token_budget,
        }

        if error_info is not None:
            result.update(error_info)
        elif response.status_code != 200:
            result.update(
                {
                    "error_type": "http_error",
                    "error_name": "HTTPError",
                    "error_status": response.status_code,
                    "error_code": response.status_code,
                    "error_message": f"Jina returned HTTP {response.status_code}",
                    "error_readable_message": f"Jina returned HTTP {response.status_code}",
                }
            )

        return result

    def _extract_jina_error(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            payload = json.loads(text)
        except Exception:
            return None

        if not isinstance(payload, dict):
            return None

        if "code" not in payload and "message" not in payload and "name" not in payload:
            return None

        error_name = payload.get("name")
        error_status = payload.get("status")
        error_code = payload.get("code")
        error_message = payload.get("message")
        readable = payload.get("readableMessage")

        error_type = "jina_api_error"
        if error_name == "InsufficientBalanceError" or error_status == 40203:
            error_type = "balance_error"
        elif error_name == "BudgetExceededError" or error_status == 40904:
            error_type = "budget_error"

        return {
            "error_type": error_type,
            "error_name": error_name,
            "error_status": error_status,
            "error_code": error_code,
            "error_message": error_message,
            "error_readable_message": readable,
        }

    def _should_retry_jina_error(self, parsed: Dict[str, Any]) -> bool:
        error_type = parsed.get("error_type")
        # Balance/budget errors are deterministic; do not burn more retries on them.
        return error_type not in {"balance_error", "budget_error"}

    def _current_jina_token_budget(self) -> Optional[int]:
        if not JINA_TOKEN_BUDGET_ENABLED:
            return None
        if JINA_TOKEN_BUDGET <= 0:
            return None
        return JINA_TOKEN_BUDGET

    def _was_visit_cache_hit(self, url: str, token_budget: Optional[int]) -> bool:
        try:
            from breadcrumbing.cache import _visit_cache

            key = _visit_cache._generate_key(
                "_jina_fetch_raw", (self, url), {"token_budget": token_budget}
            )
            return _visit_cache.get(key) is not None
        except Exception:
            return False

    def _record_visit_metadata(self, result: Dict[str, Any], succeeded: bool) -> None:
        usage_tokens = self._parse_int(result.get("usage_tokens"))
        raw_preview = result.get("raw_preview")
        if isinstance(raw_preview, str):
            raw_preview = raw_preview[:300]

        call_meta = {
            "url": result.get("url"),
            "http_status_code": result.get("http_status_code"),
            "usage_tokens": usage_tokens,
            "cache_hit": bool(result.get("cache_hit", False)),
            "succeeded": succeeded,
            "error_type": result.get("error_type"),
            "error_name": result.get("error_name"),
            "error_status": result.get("error_status"),
            "error_code": result.get("error_code"),
            "error_message": result.get("error_message"),
            "error_readable_message": result.get("error_readable_message"),
            "budget_intended_charge_tokens": self._extract_intended_charge(
                result.get("error_message")
            ),
            "jina_token_budget": result.get("token_budget"),
            "attacker_hook_applied": bool(result.get("attacker_hook_applied", False)),
            "raw_preview": raw_preview,
        }
        self.last_call_metadata["calls"].append(call_meta)

        aggregate = self.last_call_metadata["aggregate"]
        if succeeded:
            aggregate["success_count"] += 1
        else:
            aggregate["failure_count"] += 1
        if usage_tokens is not None:
            aggregate["usage_tokens_total"] += usage_tokens
        if call_meta["cache_hit"]:
            aggregate["cache_hit_count"] += 1
        if call_meta["error_type"] == "balance_error":
            aggregate["balance_error_count"] += 1
        if call_meta["error_type"] == "budget_error":
            aggregate["budget_error_count"] += 1

        log_payload = {
            "url": call_meta["url"],
            "status": call_meta["http_status_code"],
            "usage_tokens": call_meta["usage_tokens"],
            "cache_hit": call_meta["cache_hit"],
            "succeeded": call_meta["succeeded"],
            "error_type": call_meta["error_type"],
            "error_name": call_meta["error_name"],
            "error_status": call_meta["error_status"],
            "budget_intended_charge_tokens": call_meta["budget_intended_charge_tokens"],
            "jina_token_budget": call_meta["jina_token_budget"],
        }
        if succeeded:
            _logger.info(
                f"[visit] success {json.dumps(log_payload, ensure_ascii=False)}"
            )
        else:
            _logger.warning(
                f"[visit] failure {json.dumps(log_payload, ensure_ascii=False)}"
            )

    @staticmethod
    def _parse_int(value: Any) -> Optional[int]:
        try:
            if value is None or value == "":
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_intended_charge(error_message: Any) -> Optional[int]:
        if not isinstance(error_message, str):
            return None
        match = re.search(r"intended charge amount (\d+)", error_message)
        if not match:
            return None
        return int(match.group(1))
