# Derived from Tongyi DeepResearch (Apache-2.0); modified for this project.
# See NOTICE.md and licenses/DeepResearch-APACHE-2.0.txt.
import json
import json5
import logging
import os
import random
import re
import time
import asyncio
import datetime
from typing import Dict, List, Optional, Tuple, Union

from openai import APIError, APIConnectionError, APITimeoutError

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Import LLM unified interface
from breadcrumbing.models import unified_client
from breadcrumbing.runtime_trace import trace_span

# Import tools
from breadcrumbing.agent.tools.tool_search import *
from breadcrumbing.agent.tools.tool_visit import *
from breadcrumbing.agent.prompt import *

# Import attacker
from breadcrumbing.attacks import build_attacker

OBS_START = "<tool_response>"
OBS_END = "\n</tool_response>"

# Think token pattern
THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_think(content: str) -> Tuple[str, str]:
    """
    Extract think block from content.

    Returns:
        Tuple of (content_without_think, think_content)
    """
    think_matches = THINK_PATTERN.findall(content)
    think = "".join(think_matches) if think_matches else ""
    content_without_think = THINK_PATTERN.sub("", content)
    return content_without_think, think


# ================= TOKEN LIMIT CONFIG =================
MAX_CONTEXT_TOKENS = 115000  # 最大上下文窗口
MIN_REMAINING_TOKENS = 5000  # 最小剩余窗口阈值
MAX_TOOL_RESPONSE_TOKENS = 15000  # 单次工具返回最大token
SAFETY_COMPLETION_BUFFER = 50
MIN_FINAL_ANSWER_TOKENS = 256
# 主动收敛阈值：剩余调用次数低于此值时提醒模型
LLM_CALLS_WARN_THRESHOLD = int(os.getenv("LLM_CALLS_WARN_THRESHOLD", 5))
# ======================================================


def parse_context_limit_error(error_str: str) -> Optional[Tuple[int, int]]:
    """Return (max_context, input_tokens) from provider context-limit errors."""
    patterns = [
        r"maximum context length is (\d+) tokens and your request has (\d+) input tokens",
        r"maximum context length is (\d+) tokens\..*?prompt contains at least (\d+) input tokens",
        r"maximum context length is (\d+) tokens\..*?requested about \d+ tokens \((\d+) of text input, \d+ in the output\)",
    ]
    for pattern in patterns:
        match = re.search(pattern, error_str, re.DOTALL)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


TOOL_NAMES = {"search", "visit"}


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")


class MultiTurnReactAgent:
    def __init__(
        self,
        function_list: Optional[List[Union[str, Dict]]] = None,
        llm: Optional[Dict] = None,
        attacker: Optional[str] = "no_attack",
        attacker_kwargs: Optional[Dict] = None,
        max_llm_calls_per_run: int = 10,
        researcher_adversarial_enhancement: bool = False,
        **kwargs,
    ):
        """
        Initialize the MultiTurnReactAgent.

        Args:
            function_list: Optional list of functions/tools
            llm: LLM configuration dict with keys:
                - model: Model path/name
                - generate_cfg: Generation config (temperature, top_p, presence_penalty, etc.)
            attacker: Public method name (e.g., "no_attack", "ach", "agentdojo")
            attacker_kwargs: Arguments passed to attacker constructor
                (e.g., {"csv_path": "./data.csv", "model_name": "openrouter:moonshotai/kimi-k2.5"})
            researcher_adversarial_enhancement: If True, append an explicit
                verification-oriented robustness block to the researcher system prompt.
            **kwargs: Additional keyword arguments
        """
        if llm is None:
            llm = {}
        self.llm_generate_cfg = llm.get("generate_cfg", {})
        self.llm_local_path = llm.get("model", "")

        # Initialize and inject attacker
        if attacker:
            attacker_kwargs = attacker_kwargs or {}
            self.attacker = build_attacker(attacker, **attacker_kwargs)
        else:
            self.attacker = None
        self.tool_map = {"search": Search(), "visit": Visit()}
        for tool in self.tool_map.values():
            tool.set_attacker(self.attacker)

        # Store latest token count from chat_response
        self.last_token_count = 0
        # Cache for tokenizer to avoid repeated loading
        self._tokenizer_cache = None
        self.next_max_tokens = None
        self.max_llm_calls_per_run = max_llm_calls_per_run
        self.researcher_adversarial_enhancement = researcher_adversarial_enhancement
        logger.info(f"Setting max_llm_calls_per_run = {max_llm_calls_per_run}")

    def sanity_check_output(self, content):
        return "<think>" in content and "</think>" in content

    def count_tokens(self, messages):
        """Count tokens in messages using the configured tokenizer."""
        if not self.llm_local_path:
            # If no local path configured, estimate tokens (rough approximation)
            return sum(len(str(msg.get("content", ""))) // 4 for msg in messages)

        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.llm_local_path)
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
            tokens = tokenizer(full_prompt, return_tensors="pt")
            token_count = len(tokens["input_ids"][0])
            return token_count
        except Exception as e:
            logger.warning(f"Could not count tokens with tokenizer: {e}")
            # Fallback to character-based estimation
            return sum(len(str(msg.get("content", ""))) // 4 for msg in messages)

    def count_text_tokens(self, text: str) -> int:
        """Count tokens in a single text string."""
        if not self.llm_local_path:
            # Rough estimation: 4 characters per token
            return len(text) // 4

        try:
            # Lazy load and cache tokenizer
            if self._tokenizer_cache is None:
                from transformers import AutoTokenizer

                self._tokenizer_cache = AutoTokenizer.from_pretrained(
                    self.llm_local_path
                )

            tokens = self._tokenizer_cache(text, return_tensors="pt")
            return len(tokens["input_ids"][0])
        except Exception as e:
            logger.warning(f"Could not count text tokens: {e}")
            return len(text) // 4

    def truncate_text(self, text: str, max_tokens: int, add_marker: bool = True) -> str:
        """
        Truncate text to fit within max_tokens limit.

        Strategy: Keep first 70% and last 30% to preserve context structure.

        Args:
            text: The text to truncate
            max_tokens: Maximum number of tokens allowed
            add_marker: Whether to add truncation marker in the middle

        Returns:
            Truncated text
        """
        current_tokens = self.count_text_tokens(text)

        if current_tokens <= max_tokens:
            return text

        # Calculate how much to keep (70% start, 30% end)
        keep_ratio_start = 0.7
        keep_ratio_end = 0.3

        # Calculate approximate character counts
        total_chars = len(text)
        start_chars = int(
            total_chars * keep_ratio_start * (max_tokens / current_tokens)
        )
        end_chars = int(total_chars * keep_ratio_end * (max_tokens / current_tokens))

        truncated = text[:start_chars]
        if add_marker:
            truncated += "\n\n...[Content truncated due to length]...\n\n"
        truncated += text[-end_chars:]

        while True:
            final_tokens = self.count_text_tokens(truncated)
            if final_tokens <= max_tokens:
                break

            overflow = final_tokens - max_tokens

            # 估算当前 token 密度
            avg_chars_per_token = len(truncated) / final_tokens

            # 需要裁掉的字符
            cut_chars = int(4 * overflow * avg_chars_per_token)

            if cut_chars <= 0:
                cut_chars = 1000  # 保底

            truncated = truncated[:-cut_chars]

            if not truncated:
                break

        logger.debug(
            f"[Token Management] Truncated text from {current_tokens} to ~{max_tokens} tokens"
        )
        return truncated

    @staticmethod
    def _extract_first_tool_call(text: str):
        start = text.find("<tool_call>")
        if start != -1:
            end = text.find("</tool_call>", start)
            if end == -1:
                return None, None

            block = text[start + len("<tool_call>") : end].strip()
            return block, end + len("</tool_call>")

        return MultiTurnReactAgent._extract_first_raw_json_tool_call(text)

    @staticmethod
    def _strip_json_fence(text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped

        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
        return stripped

    @staticmethod
    def _find_json_object_end(text: str, start: int) -> Optional[int]:
        depth = 0
        in_string = False
        quote_char = ""
        escaped = False

        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote_char:
                    in_string = False
                continue

            if ch in {'"', "'"}:
                in_string = True
                quote_char = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return idx + 1

        return None

    @staticmethod
    def _extract_first_raw_json_tool_call(text: str):
        stripped = MultiTurnReactAgent._strip_json_fence(text)
        search_text = stripped if stripped else text

        for match in re.finditer(r"\{", search_text):
            end = MultiTurnReactAgent._find_json_object_end(search_text, match.start())
            if end is None:
                continue

            block = search_text[match.start() : end].strip()
            try:
                tool_call_json = json5.loads(block)
            except Exception:
                continue

            if not isinstance(tool_call_json, dict):
                continue

            tool_name = tool_call_json.get("name", "")
            tool_args = tool_call_json.get("arguments")
            if tool_name in TOOL_NAMES and isinstance(tool_args, dict):
                return block, None

        return None, None

    @staticmethod
    def _extract_answer_content(text: str) -> Optional[str]:
        content_clean, _ = extract_think(text)
        for start_tag in ("<answer>", "answer>"):
            start = content_clean.find(start_tag)
            if start == -1:
                continue
            end = content_clean.find("</answer>", start + len(start_tag))
            if end == -1:
                continue
            return content_clean[start + len(start_tag) : end]
        return None

    def call_server(
        self,
        messages: List[Dict],
        model_name: str,
        max_tries: int = 10,
        stop: Optional[List[str]] = None,
        max_tokens: int = 10000,
    ) -> str:
        """
        Unified LLM calling method supporting multiple backends.

        This method provides a unified interface for calling different LLM backends
        through the backend:model naming convention (e.g., ollama:qwen2.5:7b).

        Supported backends:
        - ollama: Local Ollama models (e.g., ollama:qwen2.5:7b)
        - oai: OpenAI and OpenAI-compatible APIs (e.g., oai:gpt-4)
        - openrouter: Public model gateway (e.g., openrouter:qwen/qwen3-235b-a22b-2507)

        Args:
            messages: Message list in OpenAI format
                Example: [{"role": "user", "content": "Hello"}]
            model_name: Model name in "backend:model" format
                Examples: "ollama:qwen2.5:7b", "oai:gpt-4", "openrouter:qwen/qwen3-235b-a22b-2507"
            max_tries: Maximum number of retry attempts (default: 10)
            stop: Stop sequences for generation (default: ["\\n<tool_response>", "<tool_response>"])
            max_tokens: Maximum tokens in response (default: 10000)

        Returns:
            str: Generated content or error message

        Raises:
            ValueError: If model_name format is invalid
        """
        # if self.next_max_tokens is None:
        #         max_tokens = max_tokens  # 首轮默认
        # else:
        #     max_tokens = max(0, min(max_tokens, self.next_max_tokens))

        # if max_tokens==0:
        #     raise RuntimeError("Context window exhausted — cannot generate further.")

        if stop is None:
            stop = ["\n<tool_response>", "<tool_response>"]

        # Build kwargs for the API call
        max_tokens = self.llm_generate_cfg.get("max_tokens", max_tokens)
        kwargs = {
            "messages": messages,
            "temperature": self.llm_generate_cfg.get("temperature", 0.0),
            "top_p": self.llm_generate_cfg.get("top_p", 0.95),
            "max_tokens": max_tokens,
            "presence_penalty": self.llm_generate_cfg.get("presence_penalty", 1.1),
            # "logprobs": True,
            # "stop": stop,
        }

        base_sleep_time = 1
        adjusted_max_tokens = None  # Track if we've adjusted max_tokens
        last_exception = None  # Record last exception

        for attempt in range(max_tries):
            # Use adjusted max_tokens if we previously calculated it
            current_max_tokens = (
                adjusted_max_tokens if adjusted_max_tokens is not None else max_tokens
            )

            try:
                logger.info(
                    f"Attempting to call LLM ({model_name}) with max_tokens={current_max_tokens}, try {attempt + 1}/{max_tries}"
                )

                # Update kwargs with current max_tokens
                kwargs["max_tokens"] = current_max_tokens

                # Use unified client to get response
                with trace_span(
                    "llm.logical",
                    role="researcher",
                    method="deepresearch",
                    phase="researcher_turn",
                    logical_attempt=attempt + 1,
                    max_logical_attempts=max_tries,
                ) as logical_span:
                    chat_response = unified_client.chat_completions_create(
                        model_name=model_name, **kwargs
                    )
                    usage = getattr(chat_response, "usage", None)
                    logical_span.update(
                        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                        completion_tokens=int(
                            getattr(usage, "completion_tokens", 0) or 0
                        ),
                    )

                choices = getattr(chat_response, "choices", None)
                if not choices:
                    content = ""
                else:
                    message = getattr(choices[0], "message", None)
                    content = getattr(message, "content", None) or ""

                # Record token usage from chat_response
                usage = chat_response.usage if hasattr(chat_response, "usage") else None
                if usage:
                    total_tokens = (
                        usage.total_tokens if hasattr(usage, "total_tokens") else 0
                    )
                    self.last_token_count = total_tokens
                    self.llm_call_records.append(
                        {
                            "prompt_tokens": usage.prompt_tokens
                            if hasattr(usage, "prompt_tokens")
                            else 0,
                            "completion_tokens": usage.completion_tokens
                            if hasattr(usage, "completion_tokens")
                            else 0,
                            "total_tokens": total_tokens,
                        }
                    )
                    logger.info(
                        f"prompt_tokens={self.llm_call_records[-1]['prompt_tokens']}, completion_tokens={self.llm_call_records[-1]['completion_tokens']}"
                    )
                    logger.debug(
                        f"total_tokens={total_tokens}, count_tokens={self.count_tokens(messages)}, count_text_tokens={self.count_text_tokens(content)}"
                    )
                    logger.debug(
                        f"sum={self.count_tokens(messages) + self.count_text_tokens(content)}"
                    )

                if content and content.strip():
                    logger.info(
                        f"LLM call successful ({model_name}), received a valid response"
                    )
                    return content.strip()
                else:
                    logger.warning(f"Attempt {attempt + 1} received an empty response.")

            except ValueError as e:
                raise ValueError(f"Invalid model configuration: {e}") from e
            except Exception as e:
                last_exception = e  # record the last exception
                error_str = str(e)

                # Special handling: max_tokens too large - auto-adjust and retry.
                context_limit = parse_context_limit_error(error_str)
                if context_limit and adjusted_max_tokens is None:
                    max_context, input_tokens = context_limit
                    adjusted_max_tokens = (
                        max_context - input_tokens - SAFETY_COMPLETION_BUFFER
                    )
                    if adjusted_max_tokens <= 0:
                        raise RuntimeError(
                            f"Context overflow: input_tokens={input_tokens} exceeds or equals "
                            f"model limit={max_context}."
                        )

                    if adjusted_max_tokens < MIN_FINAL_ANSWER_TOKENS:
                        raise RuntimeError(
                            f"Insufficient space for completion: only {adjusted_max_tokens} tokens available, "
                            f"minimum required is {MIN_FINAL_ANSWER_TOKENS}."
                        )

                    logger.warning(
                        f"max_tokens={current_max_tokens} is too large, adjusted to {adjusted_max_tokens} (context: {max_context}, input: {input_tokens})"
                    )
                    logger.info("Retrying immediately with adjusted max_tokens...")
                    continue

                # All other errors
                error_type = (
                    "API or network error"
                    if isinstance(e, (APIError, APIConnectionError, APITimeoutError))
                    else "unexpected error"
                )
                logger.error(f"Attempt {attempt + 1} failed with {error_type}: {e}")

            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2**attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30)

                logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                with trace_span(
                    "llm.logical_retry_backoff",
                    role="researcher",
                    method="deepresearch",
                    after_logical_attempt=attempt + 1,
                ):
                    time.sleep(sleep_time)
            else:
                logger.error(
                    "All retry attempts have been exhausted. The call has failed."
                )

        if last_exception is not None:
            raise RuntimeError(
                f"LLM call failed for {model_name} after {max_tries} attempts."
            ) from last_exception
        else:
            raise RuntimeError(
                f"LLM call failed for {model_name} "
                f"after {max_tries} attempts "
                f"(no exception captured)."
            )

    def _run(self, data: str, model_name: str, **kwargs) -> Dict:
        """
        Run a researcher trajectory using the <tool_call>/<tool_response> protocol.

        Tool-call handling:
        - Enforces ONE tool call per round
        - Rejects answer mixed with tool call
        - Handles malformed tool blocks safely
        """

        try:
            question = data["item"]["question"]
        except:
            raw_msg = data["item"]["messages"][1]["content"]
            question = (
                raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg
            )

        start_time = time.time()
        answer = data["item"].get("answer", "")

        self.user_prompt = question

        # Set the task intent before the first model call.
        if self.attacker and hasattr(self.attacker, "set_intent"):
            self.attacker.set_intent(question)
        # -------------------------

        system_prompt = SYSTEM_PROMPT
        cur_date = kwargs.get("prompt_date") or today_date()
        system_prompt = system_prompt + str(cur_date)
        if self.researcher_adversarial_enhancement:
            system_prompt = (
                system_prompt + "\n\n" + RESEARCHER_ADVERSARIAL_ENHANCEMENT_PROMPT
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        # Initialize records for metadata
        self.llm_call_records = []
        self.tool_call_records = []

        num_llm_calls_available = self.max_llm_calls_per_run
        round = 0
        final_answer = None

        while num_llm_calls_available > 0:
            if time.time() - start_time > 150 * 60:
                return {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": "No answer found after 2h30mins",
                    "termination": "No answer found after 2h30mins",
                    "metadata": {
                        "llm_calls": self.llm_call_records,
                        "tool_calls": self.tool_call_records,
                    },
                }

            round += 1
            num_llm_calls_available -= 1
            logger.info(
                f"[Call {round}/{self.max_llm_calls_per_run}] Sending request (remaining: {num_llm_calls_available})"
            )

            content = self.call_server(messages=messages, model_name=model_name)

            content = content.strip()

            tool_call_block, tool_call_end = self._extract_first_tool_call(content)
            has_tool_call = "<tool_call>" in content or tool_call_block is not None
            tool_call_count = content.count("<tool_call>") or (
                1 if tool_call_block is not None else 0
            )
            answer_content = self._extract_answer_content(content)
            has_answer = answer_content is not None

            current_tokens = self.last_token_count  # using api result to minimize error
            forced_answer_flag = False
            # =========================================================
            # CASE 1: TOOL CALL
            # =========================================================
            if has_tool_call:
                if tool_call_block is None:
                    messages.append({"role": "assistant", "content": content})
                    logger.error(
                        "Tool call block is malformed (missing </tool_call>). "
                    )
                    logger.debug(f"Incomplete tool call detected:\n{content}...")
                    result = (
                        "Error: Tool call block is malformed (missing </tool_call>). "
                        "Please retry with exactly one complete <tool_call>...</tool_call> block."
                    )
                else:
                    trimmed_content = (
                        content[:tool_call_end]
                        if tool_call_end is not None
                        else content
                    )
                    messages.append({"role": "assistant", "content": trimmed_content})
                    try:
                        # python tool
                        if "python" in tool_call_block.lower():
                            try:
                                code_raw = (
                                    tool_call_block.split("<code>")[1]
                                    .split("</code>")[0]
                                    .strip()
                                )
                                result = self.tool_map["PythonInterpreter"].call(
                                    code_raw
                                )
                            except Exception:
                                result = "[Python Interpreter Error]: Formatting error."

                        else:
                            tool_call_json = json5.loads(tool_call_block)
                            if not isinstance(tool_call_json, dict):
                                raise ValueError("Tool call JSON must be an object")

                            tool_name = tool_call_json.get("name", "")
                            tool_args = tool_call_json.get("arguments") or {}
                            if not isinstance(tool_args, dict):
                                raise ValueError(
                                    "Tool call arguments must be an object"
                                )

                            # Copy tool_args BEFORE custom_call_tool modifies it
                            tool_args_copy = dict(tool_args)

                            logger.info(f"Tool call [{tool_name}]: {tool_args}")

                            result, tool_metadata = self.custom_call_tool(
                                tool_name, tool_args, model_name=model_name
                            )
                            # Record tool call
                            tool_record = {
                                "tool_name": tool_name,
                                "tool_args": tool_args_copy,
                                "result_length": len(str(result)) if result else 0,
                            }
                            if tool_metadata is not None:
                                tool_record["tool_metadata"] = tool_metadata
                            self.tool_call_records.append(tool_record)

                    except Exception as e:
                        logger.error(f"Tool call parsing error: {e}")
                        result = (
                            "Error: Tool call is not a valid JSON. "
                            'Tool call must contain a valid "name" and "arguments" field.'
                        )
                notes = []

                if tool_call_count > 1:
                    notes.append(
                        f"Note: Multiple tool calls were detected ({tool_call_count}). "
                        "Only the first was executed. Please issue at most one <tool_call> per turn."
                    )

                if has_answer:
                    notes.append(
                        "Note: An <answer> was detected in the same message as a tool call. "
                        "Ignore that draft and wait for the <tool_response>; only output <answer> when no more tool calls are needed."
                    )

                if notes:
                    logger.warning("Notes: " + "\n".join(notes))
                    result = f"{result}\n\n" + "\n".join(notes)

                # ================= Token Management: Truncate tool result =================
                # Stage 1: Truncate if tool result exceeds MAX_TOOL_RESPONSE_TOKENS
                result_str = str(result) if result else ""
                result_tokens = self.count_text_tokens(result_str)

                if result_tokens > MAX_TOOL_RESPONSE_TOKENS:
                    logger.warning(
                        f"Tool result too large ({result_tokens} > {MAX_TOOL_RESPONSE_TOKENS}), truncating..."
                    )
                    result_str = self.truncate_text(
                        result_str, MAX_TOOL_RESPONSE_TOKENS
                    )

                # Stage 2: Predictive check - truncate if adding result would exceed MAX_CONTEXT_TOKENS
                # Calculate what the total would be after adding the result message
                wrapper_overhead = 200  # Rough estimate for "</think>\n... \n</think>" wrapper and other tokens
                result_tokens = self.count_text_tokens(result_str)
                predicted_total = current_tokens + result_tokens + wrapper_overhead
                limit = MAX_CONTEXT_TOKENS - MIN_REMAINING_TOKENS
                truncated = False

                logger.debug(f"[Token Management] Result tokens={result_tokens}")

                if predicted_total > limit:
                    truncated = True
                    # 允许加入的最大 tool result token 数
                    allowed_result_tokens = limit - current_tokens - wrapper_overhead

                    if allowed_result_tokens > 0:
                        result_str = self.truncate_text(
                            result_str, allowed_result_tokens
                        )
                    else:
                        # 连 tool result 都放不下，只能强制终止 tool 流程，并强制生成答案。
                        result_str = ""
                        forced_answer_flag = True
                if truncated:
                    notice = (
                        "\n\n[SYSTEM NOTICE]: The context window is nearly full. "
                        "Tool output has been truncated to preserve space. "
                        "You must provide your final answer immediately WITHOUT making additional tool calls."
                    )
                    result_str += notice

                # ================================================================================

                result = "<tool_response>\n" + result_str + "\n</tool_response>"
                messages.append({"role": "user", "content": result})

                updated_tokens = (
                    current_tokens
                    + self.count_text_tokens(str(result))
                    + SAFETY_COMPLETION_BUFFER
                )
                self.next_max_tokens = MAX_CONTEXT_TOKENS - updated_tokens

            # =========================================================
            # CASE 2: NORMAL ANSWER
            # =========================================================
            else:
                messages.append({"role": "assistant", "content": content})

                if has_answer:
                    final_answer = answer_content
                    termination = "answer"
                    break
                else:
                    logger.warning(f"No answer tags detected: {content[:500]}")
                    messages.append(
                        {
                            "role": "user",
                            "content": "Please wrap your final answer using <answer>...</answer> format. "
                            "Do not output any plain text outside <answer> tags.",
                        }
                    )

            # =========================================================
            # 主动收敛预警：当剩余次数不多时提醒模型
            # =========================================================
            if 0 < num_llm_calls_available <= LLM_CALLS_WARN_THRESHOLD:
                warn_msg = (
                    f"[SYSTEM WARNING] Only {num_llm_calls_available} LLM call(s) remaining. "
                    "Please wrap up your reasoning and provide your final answer soon."
                )
                messages.append({"role": "user", "content": warn_msg})
                logger.warning(f"[Proactive Convergence] {warn_msg}")

            # =========================================================
            # Final safety check: token limit exceeded OR last LLM call
            # =========================================================
            # This is a final safety check after all the proactive measures above

            if (
                current_tokens > MAX_CONTEXT_TOKENS
                or forced_answer_flag
                or num_llm_calls_available == 0
            ):
                # 区分终止原因
                if num_llm_calls_available == 0:
                    reason = f"reached max LLM calls ({self.max_llm_calls_per_run})"
                    notice_msg = (
                        "[SYSTEM NOTICE]: You have reached the maximum number of LLM calls. "
                        "Stop all tool calls and provide your final answer immediately.\n"
                        "<answer>your answer here</answer>"
                    )
                else:
                    reason = f"context limit exceeded: {current_tokens} > {MAX_CONTEXT_TOKENS} or forced_answer_flag = {forced_answer_flag}"
                    notice_msg = (
                        "[SYSTEM NOTICE]: You have reached the maximum context length. "
                        "Stop all tool calls and provide your final answer immediately.\n"
                        "<answer>your answer here</answer>"
                    )

                logger.warning(f"[Token Management] CRITICAL: {reason}")
                logger.warning("[Token Management] Forcing final answer generation...")

                # 对于max LLM calls的情况，append新消息保留tool result；对于token limit，替换以节省空间
                if num_llm_calls_available == 0:
                    messages.append({"role": "user", "content": notice_msg})
                else:
                    messages[-1]["content"] = notice_msg

                content = self.call_server(messages, model_name)
                messages.append({"role": "assistant", "content": content.strip()})

                forced_answer_content = self._extract_answer_content(content)
                if forced_answer_content is not None:
                    prediction = forced_answer_content
                    termination = (
                        "max LLM calls - forced answer"
                        if num_llm_calls_available == 0
                        else "token limit exceeded - forced answer"
                    )
                else:
                    prediction = content
                    termination = (
                        "format error: max LLM calls - forced answer"
                        if num_llm_calls_available == 0
                        else "format error: token limit exceeded - forced answer"
                    )

                # Extract think from last assistant message
                think = ""
                for msg in reversed(messages):
                    if msg.get("role") == "assistant":
                        _, think = extract_think(msg.get("content", ""))
                        break

                return {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination,
                    "metadata": {
                        "llm_calls": self.llm_call_records,
                        "tool_calls": self.tool_call_records,
                        "think": think,
                    },
                }

        if final_answer is not None:
            prediction = final_answer
            termination = "answer"
        else:
            prediction = None
            for msg in reversed(messages):
                if msg.get("role") != "assistant":
                    continue
                msg_content = msg.get("content", "")
                answer_content = self._extract_answer_content(msg_content)
                if answer_content is not None:
                    prediction = answer_content
                    termination = "answer"
                    break

        if prediction is None:
            prediction = "No answer found."
            termination = (
                "exceed available llm calls"
                if num_llm_calls_available == 0
                else "answer not found"
            )

        # Extract think from last assistant message
        think = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                _, think = extract_think(msg.get("content", ""))
                break

        return {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination,
            "metadata": {
                "llm_calls": self.llm_call_records,
                "tool_calls": self.tool_call_records,
                "think": think,
            },
        }

    def custom_call_tool(
        self, tool_name: str, tool_args: dict, model_name=None, **kwargs
    ):
        """Execute a tool call with the given name and arguments."""
        if not isinstance(tool_args, dict):
            tool_args = {}

        with trace_span("tool.call", tool_name=tool_name):
            if tool_name in self.tool_map:
                tool = self.tool_map[tool_name]
                if model_name:
                    kwargs["model_name"] = model_name
                tool_args["params"] = tool_args
                if "python" in tool_name.lower():
                    result = self.tool_map["PythonInterpreter"].call(tool_args)
                elif tool_name == "parse_file":
                    params = {"files": tool_args["files"]}

                    raw_result = asyncio.run(
                        tool.call(params, file_root_path="./eval_data/file_corpus")
                    )
                    result = raw_result

                    if not isinstance(raw_result, str):
                        result = str(raw_result)
                else:
                    raw_result = tool.call(tool_args, **kwargs)
                    result = raw_result
                tool_metadata = getattr(tool, "last_call_metadata", None)
                if isinstance(tool_metadata, dict):
                    try:
                        tool_metadata = json.loads(
                            json.dumps(tool_metadata, ensure_ascii=False)
                        )
                    except Exception:
                        tool_metadata = {"raw_tool_metadata": str(tool_metadata)}
                else:
                    tool_metadata = None
                return result, tool_metadata

            return f"Error: Tool {tool_name} not found", None
