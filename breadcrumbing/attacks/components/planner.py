"""
attacker的核心planner，负责search和visit阶段的计划。

决策流程:
- plan_search: 接受原始query和response，基于intent和memory的总结做出攻击决策
- 使用unified_client进行LLM调用，支持多轮对话
- 输出Decision: {attack: bool, payload: Payload | None, reflection: str}
"""

from ..schema import Decision, Payload
from openai import APIError, APIConnectionError, APITimeoutError
from typing import Any, Dict, Optional, List, Tuple
import ast
import hashlib
import random
import re
import time
import json5
import logging

from breadcrumbing.models import unified_client
from breadcrumbing.runtime_trace import trace_span
from .prompts import (
    get_search_plan_prompts,
    get_visit_plan_prompts,
    get_observe_visit_prompts,
)

# Special tokens for LLM interaction (defined at file top)
OBS_START = "<observe>"
OBS_END = "</observe>"
ANSWER_START = "<answer>"
ANSWER_END = "</answer>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
TOOL_RESPONSE_START = "<tool_response>"
TOOL_RESPONSE_END = "</tool_response>"

# Think token pattern
THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
SAFETY_COMPLETION_BUFFER = 50
MIN_FINAL_ANSWER_TOKENS = 256


def expected_answer_json(include_reflection: bool = True) -> str:
    if include_reflection:
        return (
            f"{ANSWER_START}\n"
            "{'attack': true/false, 'payload': {...} or null, 'reflection': '...'}\n"
            f"{ANSWER_END}"
        )
    return (
        f"{ANSWER_START}\n"
        "{'attack': true/false, 'payload': {...} or null}\n"
        f"{ANSWER_END}"
    )


def expected_answer_fields(include_reflection: bool = True) -> str:
    if include_reflection:
        return "The JSON should contain 'attack', 'payload' (or null), and 'reflection' fields."
    return "The JSON should contain 'attack' and 'payload' (or null) fields."


def parse_context_limit_error(error_str: str) -> Optional[Tuple[int, int]]:
    """Return (max_context, input_tokens) from provider context-limit errors."""
    patterns = [
        r"maximum context length is (\d+) tokens and your request has (\d+) input tokens",
        r"maximum context length is (\d+) tokens\..*?prompt contains at least (\d+) input tokens",
    ]
    for pattern in patterns:
        match = re.search(pattern, error_str, re.DOTALL)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


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


def escape_multiline_string_literals(text: str) -> str:
    """
    Escape raw control characters that appear inside quoted JSON-like strings.

    Some models emit literal newlines in a JSON string value such as reflection,
    which breaks both JSON and JSON5 parsing. This helper preserves structure
    while replacing those control characters with escaped sequences.
    """
    escaped_chars: List[str] = []
    in_string = False
    quote_char = ""
    escape_next = False

    for ch in text:
        if escape_next:
            escaped_chars.append(ch)
            escape_next = False
            continue

        if ch == "\\":
            escaped_chars.append(ch)
            if in_string:
                escape_next = True
            continue

        if ch in ('"', "'"):
            escaped_chars.append(ch)
            if in_string:
                if ch == quote_char:
                    in_string = False
                    quote_char = ""
            else:
                in_string = True
                quote_char = ch
            continue

        if in_string:
            if ch == "\n":
                escaped_chars.append("\\n")
                continue
            if ch == "\r":
                escaped_chars.append("\\r")
                continue
            if ch == "\t":
                escaped_chars.append("\\t")
                continue

        escaped_chars.append(ch)

    return "".join(escaped_chars)


class Planner:
    """
    LLM-driven attack planner.

    Responsibilities:
    - decide whether to attack search results
    - generate payload if needed
    - provide reflection for next stages

    Stateless. Pure decision maker.
    """

    def __init__(
        self,
        attacker,
        model_name: str,
        log_level: str = "INFO",
        temperature: float = 0.5,
    ):
        self.attacker = attacker
        self.model_name = model_name
        self.temperature = temperature

        # Setup logging (handler configured by calling script)
        self.logger = logging.getLogger("Planner")
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    def _run_with_prompt_metadata(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        include_reflection: bool = True,
    ) -> tuple[Decision, dict]:
        decision, meta = self._run_planning_loop(
            system_prompt,
            user_prompt,
            include_reflection=include_reflection,
        )
        meta["prompt_render"] = {
            "profile": getattr(self.attacker, "prompt_profile", "rich"),
            "system_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            "user_sha256": hashlib.sha256(user_prompt.encode("utf-8")).hexdigest(),
            "system_chars": len(system_prompt),
            "user_chars": len(user_prompt),
        }
        return decision, meta

    def call_llm(
        self,
        messages: List[Dict],
        model_name: str,
        max_tries: int = 10,
        stop: Optional[List[str]] = None,
        max_tokens: int = 10000,
        meta_list: Optional[List[Dict]] = None,
    ) -> str:
        """
        Unified LLM calling method for the planner.

        Args:
            messages: Message list in OpenAI format
            model_name: Model name in "backend:model" format
            max_tries: Maximum number of retry attempts
            stop: Stop sequences for generation
            max_tokens: Maximum tokens in response
            meta_list: Optional list to append LLM call metadata

        Returns:
            str: Generated content or error message
        """
        if stop is None:
            stop = ["\n<observe>", "<observe>"]

        # Build kwargs for the API call
        kwargs = {
            "messages": messages,
            "temperature": self.temperature,
            "top_p": 0.95,
            "max_tokens": max_tokens,
            "presence_penalty": 1.1,
            # "stop": stop,
        }

        base_sleep_time = 1
        adjusted_max_tokens = (
            None  # Track if we've adjusted max_tokens due to context limits
        )
        last_exception = None  # Record last exception

        for attempt in range(max_tries):
            # Use adjusted max_tokens if we previously calculated it
            current_max_tokens = (
                adjusted_max_tokens if adjusted_max_tokens is not None else max_tokens
            )

            # Update kwargs with current max_tokens
            kwargs["max_tokens"] = current_max_tokens
            start = time.perf_counter()
            try:
                self.logger.info(
                    f"Calling LLM ({model_name}), attempt {attempt + 1}/{max_tries}"
                )

                # Use unified client to get response
                with trace_span(
                    "llm.logical",
                    role="attacker",
                    method=getattr(self.attacker, "mode", "ach"),
                    phase="planner",
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
                duration = (time.perf_counter() - start) * 1000

                # === Record LLM call to meta list ===
                if meta_list is not None:
                    usage = getattr(chat_response, "usage", None)
                    meta_list.append(
                        {
                            "type": "llm",
                            "tokens": {
                                "prompt": getattr(usage, "prompt_tokens", 0)
                                if usage
                                else 0,
                                "completion": getattr(usage, "completion_tokens", 0)
                                if usage
                                else 0,
                                "total": getattr(usage, "total_tokens", 0)
                                if usage
                                else 0,
                            },
                            "duration_ms": duration,
                            "success": True,
                        }
                    )

                if content and content.strip():
                    self.logger.info("LLM call successful, received response")
                    return content.strip()
                else:
                    self.logger.warning(
                        f"Warning: Attempt {attempt + 1} received empty response"
                    )

            except ValueError as e:
                self.logger.error(f"Invalid model configuration: {e}")
                return f"Invalid model configuration: {e}"
            except Exception as e:
                last_exception = e  # record the last exception
                error_str = str(e)

                # === Special handling: max_tokens too large - auto-adjust and retry ===
                context_limit = parse_context_limit_error(error_str)
                if context_limit and adjusted_max_tokens is None:
                    max_context, input_tokens = context_limit
                    adjusted_max_tokens = (
                        max_context - input_tokens - SAFETY_COMPLETION_BUFFER
                    )

                    if adjusted_max_tokens <= 0:
                        self.logger.error(
                            f"Context overflow: input_tokens={input_tokens} exceeds or equals model limit={max_context}"
                        )
                        return f"Context overflow: input_tokens={input_tokens} exceeds model limit={max_context}"

                    if adjusted_max_tokens < MIN_FINAL_ANSWER_TOKENS:
                        self.logger.error(
                            f"Insufficient space for completion: only {adjusted_max_tokens} tokens available, minimum required is {MIN_FINAL_ANSWER_TOKENS}"
                        )
                        return f"Insufficient space for completion: only {adjusted_max_tokens} tokens available"

                    self.logger.info(
                        f"max_tokens={current_max_tokens} is too large. Adjusted to {adjusted_max_tokens} (context: {max_context}, input: {input_tokens})"
                    )
                    self.logger.info("Retrying immediately with adjusted max_tokens...")
                    continue

                # === Handle other API/network errors ===
                error_type = (
                    "API or network error"
                    if isinstance(e, (APIError, APIConnectionError, APITimeoutError))
                    else "unexpected error"
                )
                self.logger.error(f"{error_type} on attempt {attempt + 1}: {e}")

                # === Record failed call ===
                if meta_list is not None:
                    meta_list.append(
                        {
                            "type": "llm",
                            "tokens": {"prompt": 0, "completion": 0, "total": 0},
                            "duration_ms": (time.perf_counter() - start) * 1000,
                            "success": False,
                        }
                    )

            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2**attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30)
                self.logger.info(f"Retrying in {sleep_time:.2f}s...")
                with trace_span(
                    "llm.logical_retry_backoff",
                    role="attacker",
                    method=getattr(self.attacker, "mode", "ach"),
                    after_logical_attempt=attempt + 1,
                ):
                    time.sleep(sleep_time)
            else:
                self.logger.error("All retry attempts exhausted")

        if last_exception is not None:
            return f"[Planner] LLM call failed for {model_name} after {max_tries} attempts: {last_exception}"
        else:
            return (
                f"[Planner] LLM call failed for {model_name} after {max_tries} attempts"
            )

    def _extract_answer(self, content: str) -> Tuple[Optional[str], str]:
        """
        Extract content from <answer>...</answer> tags, after removing think tokens.

        Args:
            content: The raw content from LLM response

        Returns:
            Tuple of (extracted_answer, think_content)
        """
        content_clean, think = extract_think(content)
        if ANSWER_START in content_clean and ANSWER_END in content_clean:
            answer = content_clean.split(ANSWER_START)[1].split(ANSWER_END)[0].strip()
            return answer, think
        return None, think

    def _extract_raw_decision_json(self, content: str) -> Optional[str]:
        """
        Accept bare JSON decisions only for no-strategy runs.

        The no-strategy prompt intentionally removes the tool/state protocol. Some
        models then answer with raw JSON despite the <answer> instruction. This keeps
        the ablation from being dominated by wrapper-format failures while leaving the
        normal strategy workflow unchanged.
        """
        content_clean, _ = extract_think(content)
        stripped = content_clean.strip()
        if not stripped:
            return None

        if stripped.startswith("```"):
            stripped = re.sub(
                r"^```(?:json|json5)?\s*", "", stripped, flags=re.IGNORECASE
            )
            stripped = re.sub(r"\s*```$", "", stripped).strip()

        candidates = [stripped]
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end > start:
            candidates.append(stripped[start : end + 1])

        for candidate in candidates:
            try:
                parsed = self._parse_decision_json(
                    candidate,
                    allow_python_literals=True,
                )
            except Exception:
                continue
            if isinstance(parsed, dict) and "attack" in parsed:
                return candidate

        return None

    def _extract_first_tool_call(self, text: str):
        """
        Extract a tagged tool call or a complete bare/fenced tool object.

        Args:
            text: The raw content from LLM response

        Returns:
            Tuple of (tool_call_block, end_position) or (None, None) if not found
        """
        start = text.find(TOOL_CALL_START)
        if start == -1:
            block = text.strip()
            if block.startswith("```") and block.endswith("```"):
                first_line, separator, body = block.partition("\n")
                if not separator or first_line.strip() not in {
                    "```",
                    "```json",
                    "```json5",
                }:
                    return None, None
                block = body[:-3].strip()
            try:
                tool_call = json5.loads(block)
            except (ValueError, TypeError):
                return None, None
            if (
                isinstance(tool_call, dict)
                and isinstance(tool_call.get("name"), str)
                and tool_call.get("name")
                in {"get_strategies", "set_strategy", "set_runtime_strategy"}
                and isinstance(tool_call.get("arguments"), dict)
            ):
                return block, None
            return None, None

        end = text.find(TOOL_CALL_END, start)
        if end == -1:
            return None, None

        block = text[start + len(TOOL_CALL_START) : end].strip()
        return block, end + len(TOOL_CALL_END)

    def _parse_decision_json(
        self,
        answer_content: str,
        allow_python_literals: bool = False,
    ) -> Dict[str, Any]:
        """
        Parse planner decision JSON with a narrow repair fallback.

        The fallback only repairs raw control characters inside quoted string
        literals, which is a frequent LLM formatting error for `reflection`.
        """
        try:
            return json5.loads(answer_content)
        except Exception as original_error:
            if allow_python_literals:
                try:
                    parsed = ast.literal_eval(answer_content)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass

            repaired_content = escape_multiline_string_literals(answer_content)
            if repaired_content == answer_content:
                raise original_error

            try:
                parsed = json5.loads(repaired_content)
                self.logger.warning(
                    "Decision JSON required multiline-string repair before parsing"
                )
                return parsed
            except Exception:
                if allow_python_literals:
                    try:
                        parsed = ast.literal_eval(repaired_content)
                        if isinstance(parsed, dict):
                            self.logger.warning(
                                "Decision JSON required Python-literal repair before parsing"
                            )
                            return parsed
                    except Exception:
                        pass
                raise original_error

    def _normalize_disable_strategy_decision(
        self, decision_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not getattr(self.attacker, "disable_strategy", False):
            return decision_dict

        payload = decision_dict.get("payload")
        if not isinstance(payload, dict):
            return decision_dict

        normalized_payload = dict(payload)
        renamed_fields = []
        field_aliases = {
            "url": "link",
            "published_time": "date",
            "markdown_content": "page_content",
            "visit_content": "page_content",
            "visit_payload": "page_content",
        }
        for source_field, target_field in field_aliases.items():
            if (
                source_field in normalized_payload
                and target_field not in normalized_payload
            ):
                normalized_payload[target_field] = normalized_payload[source_field]
                renamed_fields.append(f"{source_field}->{target_field}")

        allowed_fields = set(Payload.__dataclass_fields__)
        filtered_payload = {
            key: value
            for key, value in normalized_payload.items()
            if key in allowed_fields
        }

        if filtered_payload != payload:
            decision_dict = dict(decision_dict)
            decision_dict["payload"] = filtered_payload
            dropped_fields = sorted(set(normalized_payload) - allowed_fields)
            self.logger.warning(
                "Normalized no-strategy payload fields"
                + (f" renamed={renamed_fields}" if renamed_fields else "")
                + (f" dropped={dropped_fields}" if dropped_fields else "")
            )

        return decision_dict

    def custom_call_tool(
        self, tool_name: str, tool_args: dict, meta_list: Optional[List[Dict]] = None
    ) -> str:
        """
        Execute a tool call with the given name and arguments.

        This method delegates to the attacker's tool methods:
        - get_strategies: Get all available attack strategies
        - set_strategy: Set the current attack strategy by name
        - set_runtime_strategy: Temporarily set a strategy at runtime (not persisted)
        Args:
            tool_name: The name of the tool to call
            tool_args: Dictionary of arguments for the tool
            meta_list: Optional list to append tool call metadata

        Returns:
            str: The result of the tool execution
        """
        start = time.perf_counter()
        result = ""
        if not isinstance(tool_args, dict):
            tool_args = {}

        def record_strategy_meta(action: str, new: str):
            old_name = (
                self.attacker.current_strategy.name
                if self.attacker.current_strategy
                else None
            )
            if meta_list is not None:
                meta_list.append(
                    {
                        "type": "strategy",
                        "action": action,
                        "from": old_name,
                        "to": new,
                    }
                )

        try:
            if getattr(self.attacker, "disable_strategy", False) and tool_name in {
                "get_strategies",
                "set_strategy",
                "set_runtime_strategy",
            }:
                result = "Error: selection tools are disabled for this run. Output <answer> directly."

            elif getattr(self.attacker, "lock_strategy", False) and tool_name in {
                "get_strategies",
                "set_strategy",
            }:
                result = (
                    "Error: strategy selection tools are disabled for this fixed-strategy run. "
                    "The active strategy is already locked by experiment configuration; output <answer> directly."
                )

            elif tool_name == "get_strategies":
                result = str(self.attacker.get_strategies())

            elif tool_name == "set_strategy":
                strategy_name = tool_args.get("name", "")
                if not strategy_name:
                    result = "Error: 'name' argument is required for set_strategy tool"
                else:
                    record_strategy_meta("set", strategy_name)
                    self.attacker.set_strategy(strategy_name)
                    result = f"Strategy '{strategy_name}' has been set as the current strategy"
            elif tool_name == "set_runtime_strategy":
                strategy_dict = tool_args.get("strategy_dict", {})
                if not strategy_dict:
                    result = "Error: 'strategy_dict' argument is required for set_runtime_strategy tool"
                else:
                    try:
                        new_name = strategy_dict.get("name", "")
                        record_strategy_meta("set_runtime", new_name)
                        self.attacker.set_runtime_strategy(strategy_dict)
                        result = "Runtime strategy has been updated (not persisted)"
                    except Exception as e:
                        result = f"Error setting runtime strategy: {e}"

            # =========================
            # unknown tool
            # =========================
            else:
                result = (
                    f"Error: Tool '{tool_name}' not found. "
                    "Available tools: get_strategies, set_strategy, "
                    "set_runtime_strategy"
                )

        except Exception as e:
            result = f"Error executing tool '{tool_name}': {e}"

        # =========================
        # Unified tool meta recording
        # =========================
        duration_ms = (time.perf_counter() - start) * 1000

        if meta_list is not None:
            meta_list.append(
                {
                    "type": "tool",
                    "name": tool_name,
                    "args": tool_args,
                    "duration_ms": duration_ms,
                    "success": not result.startswith("Error"),
                    "result": result[:100],
                }
            )

        return result

    def _run_planning_loop(
        self,
        system_prompt: str,
        user_prompt: str,
        include_reflection: bool = True,
    ) -> tuple[Decision, dict]:
        """
        Core planning execution engine - handles multi-round LLM interaction.

        This method orchestrates the decision-making process by:
        1. Building initial messages from prompts
        2. Conducting multi-turn dialogue with the LLM
        3. Handling tool calls for strategy management
        4. Parsing the LLM response to extract the decision
        5. Handling invalid responses with appropriate feedback
        6. Collecting behavior metadata (LLM calls, tool calls, etc.)

        Args:
            system_prompt: The system prompt for the LLM
            user_prompt: The initial user prompt for the LLM

        Returns:
            Tuple of (Decision, behavior_meta) where:
            - Decision: {attack: bool, payload: Payload | None, reflection: str}
            - behavior_meta: Dict with llm_calls, tool_calls, strategy_changes, decision, time
        """
        # Build initial messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Collect behavior metadata
        meta_list: List[Dict] = []

        max_rounds = 12
        max_tool_rounds = 5
        max_same_tool_repeats = 2
        tool_rounds = 0
        last_tool_signature = None
        same_tool_repeat_count = 0
        strategies_inspected = False

        for round_num in range(1, max_rounds + 1):
            self.logger.info(f"Round {round_num}: Calling LLM...")

            # Call LLM with meta tracking
            content = self.call_llm(messages, self.model_name, meta_list=meta_list)

            if not content:
                self.logger.error("Error: Received empty response from LLM")
                break

            content = content.strip()

            # Check for tool calls
            tool_call_block, tool_call_end = self._extract_first_tool_call(content)
            has_tool_call = TOOL_CALL_START in content or tool_call_block is not None
            has_answer = ANSWER_START in content and ANSWER_END in content

            # =========================================================
            # CASE 1: TOOL CALL
            # =========================================================
            if has_tool_call:
                tool_rounds += 1
                if tool_call_block is None:
                    messages.append({"role": "assistant", "content": content})
                    result = (
                        f"Error: Tool call block is malformed (missing {TOOL_CALL_END}). "
                        f"Please retry with exactly one complete {TOOL_CALL_START}...{TOOL_CALL_END} block."
                    )
                else:
                    trimmed_content = (
                        content[:tool_call_end]
                        if tool_call_end is not None
                        else content
                    )
                    messages.append({"role": "assistant", "content": trimmed_content})
                    try:
                        tool_call_json = json5.loads(tool_call_block)
                        if not isinstance(tool_call_json, dict):
                            raise ValueError("Tool call JSON must be an object")

                        tool_name = tool_call_json.get("name", "")
                        tool_args = tool_call_json.get("arguments") or {}
                        if not isinstance(tool_args, dict):
                            raise ValueError("Tool call arguments must be an object")

                        self.logger.info(f"Tool call: {tool_name}")
                        self.logger.info(f"Tool args: {tool_args}")

                        tool_signature = (
                            tool_name,
                            json5.dumps(tool_args, sort_keys=True),
                        )
                        if tool_signature == last_tool_signature:
                            same_tool_repeat_count += 1
                        else:
                            last_tool_signature = tool_signature
                            same_tool_repeat_count = 1

                        if tool_name == "get_strategies" and strategies_inspected:
                            result = (
                                "Error: `get_strategies` has already been called in this planning session. "
                                "Do not call it again. You MUST now call `set_strategy` with an existing "
                                "strategy name before returning <answer>."
                            )
                        elif same_tool_repeat_count >= max_same_tool_repeats:
                            result = (
                                f"Error: Repeated identical tool call detected for `{tool_name}`. "
                                "Do not repeat the same tool call. "
                                "If no strategy is active, you MUST call `set_strategy` next. "
                                "If a strategy is already active, output <answer> instead of another tool call."
                            )
                        else:
                            result = self.custom_call_tool(
                                tool_name, tool_args, meta_list=meta_list
                            )
                            if tool_name == "get_strategies" and not result.startswith(
                                "Error"
                            ):
                                strategies_inspected = True

                    except ValueError as e:
                        result = (
                            f"Error: Tool call is not valid JSON. {e}. "
                            'Tool call must contain valid JSON with "name" and "arguments" fields.'
                        )
                    except Exception as e:
                        result = f"Error executing tool: {e}"

                notes = []

                # Check for multiple tool calls
                tool_call_count = content.count(TOOL_CALL_START)
                if tool_call_count > 1:
                    notes.append(
                        f"Note: Multiple tool calls were detected ({tool_call_count}). "
                        f"Only the first was executed. Please issue at most one {TOOL_CALL_START} per turn."
                    )

                # Check for answer mixed with tool call
                if has_answer:
                    notes.append(
                        f"Note: An {ANSWER_START} was detected in the same message as a tool call. "
                        f"Ignore that draft and wait for the {TOOL_RESPONSE_START}; "
                        f"only output {ANSWER_START} when no more tool calls are needed."
                    )

                if notes:
                    result = f"{result}\n\n" + "\n".join(notes)

                result = f"{TOOL_RESPONSE_START}\n{result}\n{TOOL_RESPONSE_END}"
                messages.append({"role": "user", "content": result})

                if tool_rounds >= max_tool_rounds:
                    self.logger.warning(
                        "Max tool rounds reached, returning default no-attack decision"
                    )
                    break

                continue

            # =========================================================
            # CASE 2: NORMAL ANSWER (Final decision)
            # =========================================================
            # Strategy: try raw JSON first (most reliable across models),
            # fall back to extracting JSON from an <answer> block.
            answer_content = self._extract_raw_decision_json(content)
            raw_json_without_answer_tag = answer_content is not None

            if answer_content is None:
                answer_content, _ = self._extract_answer(content)

            if answer_content is not None:
                messages.append({"role": "assistant", "content": content})
                self.logger.info("Received answer, parsing decision...")
                if raw_json_without_answer_tag:
                    self.logger.warning(
                        "Accepted raw JSON decision without <answer> tags"
                    )

                # === Guard: Check for active strategy ===
                if self.attacker.current_strategy is None and not getattr(
                    self.attacker, "disable_strategy", False
                ):
                    self.logger.error(
                        "Error: No active strategy detected, rejecting answer"
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "ERROR: No active strategy is set.\n\n"
                                "You are currently in STATE 1 — No Active Strategy.\n"
                                "Returning <answer> in this state is STRICTLY FORBIDDEN.\n\n"
                                "You MUST first call `get_strategies` to inspect available strategies.\n"
                                "Then call `set_strategy` with an existing strategy name.\n\n"
                                "Do NOT output <answer> until an active strategy has been established.\n"
                                "Output EXACTLY ONE <tool_call>."
                            ),
                        }
                    )
                    continue

                # Parse the JSON decision
                try:
                    decision_dict = self._parse_decision_json(
                        answer_content,
                        allow_python_literals=(
                            getattr(self.attacker, "disable_strategy", False)
                            or getattr(self.attacker, "lock_strategy", False)
                        ),
                    )
                    decision_dict = self._normalize_disable_strategy_decision(
                        decision_dict
                    )
                    decision = Decision.from_dict(decision_dict)
                    if not include_reflection:
                        decision.reflection = ""
                    self.logger.info(f"Decision parsed: attack={decision.attack}")

                    # === Build behavior meta ===
                    llm_calls = [m for m in meta_list if m.get("type") == "llm"]
                    tool_calls = [m for m in meta_list if m.get("type") == "tool"]
                    strategy_changes = [
                        m for m in meta_list if m.get("type") == "strategy"
                    ]

                    # Only include non-empty fields
                    meta_dict = {}
                    if llm_calls:
                        meta_dict["llm_calls"] = llm_calls
                    if tool_calls:
                        meta_dict["tool_calls"] = tool_calls
                    if strategy_changes:
                        meta_dict["strategy_changes"] = strategy_changes

                    return decision, meta_dict

                except Exception as e:
                    self.logger.error(f"Error: Failed to parse decision JSON: {e}")
                    self.logger.error(f"Raw answer content: {answer_content}")

                    # Provide feedback and retry
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Error: Your response was not valid JSON. "
                                f"Please format your response as:\n"
                                f"{expected_answer_json(include_reflection)}"
                            ),
                        }
                    )
                    continue

            # =========================================================
            # CASE 3: No valid decision JSON found
            # =========================================================
            messages.append({"role": "assistant", "content": content})
            self.logger.warning("No valid decision JSON found, requesting retry...")

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Please provide your decision as a JSON object. "
                        f"{expected_answer_fields(include_reflection)}"
                    ),
                }
            )

        # If we exhausted all rounds, return a default no-attack decision
        self.logger.warning("Max rounds reached, returning default no-attack decision")

        llm_calls = [m for m in meta_list if m.get("type") == "llm"]
        tool_calls = [m for m in meta_list if m.get("type") == "tool"]
        strategy_changes = [m for m in meta_list if m.get("type") == "strategy"]

        # Only include non-empty fields
        meta_dict = {}
        if llm_calls:
            meta_dict["llm_calls"] = llm_calls
        if tool_calls:
            meta_dict["tool_calls"] = tool_calls
        if strategy_changes:
            meta_dict["strategy_changes"] = strategy_changes

        return Decision(
            attack=False,
            payload=None,
            reflection=""
            if not include_reflection
            else "Max rounds reached without valid decision",
        ), meta_dict

    def plan_search(
        self,
        *,
        query: str,
        response: Dict[str, Any],
        intent: Dict[str, Any],
        memory_summary: str,
        current_strategy: str,
    ) -> tuple[Decision, dict]:
        """
        Decide whether to inject malicious search result.

        This method builds prompts for the LLM and delegates to the
        unified planning loop for multi-turn dialogue and decision parsing.

        Args:
            query: The original search query
            response: The raw search response from the search tool
            intent: The recognized user intent
            memory_summary: Summary of previous events from memory
            current_strategy: The current active strategy

        Returns:
            Tuple of (Decision, behavior_meta)
        """
        self.logger.info("Starting plan_search decision process...")

        include_reflection = not getattr(
            self.attacker, "disable_reflection_generation", False
        )

        # Get prompts
        system_prompt, user_prompt = get_search_plan_prompts(
            query=query,
            response=response,
            intent=intent,
            memory_summary=memory_summary,
            current_strategy=current_strategy,
            include_reflection=include_reflection,
            domain_constraint_mode=getattr(
                self.attacker, "domain_constraint_mode", "real_reuse"
            ),
            controlled_domain_root=getattr(self.attacker, "controlled_domain_root", "")
            or "",
            controlled_visit_mode=getattr(
                self.attacker, "controlled_visit_mode", "generated"
            ),
            prompt_profile=getattr(self.attacker, "prompt_profile", "rich"),
        )

        # Delegate to unified planning loop
        return self._run_with_prompt_metadata(
            system_prompt,
            user_prompt,
            include_reflection=include_reflection,
        )

    def plan_visit(
        self,
        *,
        url: str,
        response: str,
        original_query: str,
        injected_payload: Payload,
        search_reflection: str,
        intent: Dict[str, Any],
        memory_summary: str,
        current_strategy: str,
    ) -> tuple[Decision, dict]:
        """
        Generate malicious webpage content for an injected URL visit.

        This method builds prompts for the visit phase and delegates to the
        unified planning loop for multi-turn dialogue and decision parsing.

        Args:
            url: The URL being visited (injected URL from search phase).
            response: Original page content from the visit tool.
            original_query: The user query that triggered the injection.
            injected_payload: The payload created during search phase.
            search_reflection: Reflection produced during search planning.
            intent: The recognized user intent (dict with user_query, user_expectation, etc.).
            memory_summary: Summary of prior attack events.
            current_strategy: Active attack strategy prompt text.

        Returns:
            Tuple of (Decision, behavior_meta)
        """
        self.logger.info("Starting plan_visit decision process...")

        include_reflection = not getattr(
            self.attacker, "disable_reflection_generation", False
        )

        # Get prompts
        system_prompt, user_prompt = get_visit_plan_prompts(
            url=url,
            response=response,
            original_query=original_query,
            injected_payload=injected_payload,
            search_reflection=search_reflection,
            intent=intent,
            memory_summary=memory_summary,
            current_strategy=current_strategy,
            include_reflection=include_reflection,
            domain_constraint_mode=getattr(
                self.attacker, "domain_constraint_mode", "real_reuse"
            ),
            controlled_domain_root=getattr(self.attacker, "controlled_domain_root", "")
            or "",
            prompt_profile=getattr(self.attacker, "prompt_profile", "rich"),
        )

        # Delegate to unified planning loop
        return self._run_with_prompt_metadata(
            system_prompt,
            user_prompt,
            include_reflection=include_reflection,
        )

    def plan_observe_visit(
        self,
        *,
        url: str,
        response: str,
        original_query: str,
        intent: Dict[str, Any],
        memory_summary: str,
        current_strategy: str,
    ) -> tuple[Decision, dict]:
        self.logger.info("Starting plan_observe_visit decision process...")
        if getattr(self.attacker, "disable_reflection_generation", False):
            return Decision(attack=False, payload=None, reflection=""), {
                "reflection_generation_disabled": True
            }

        system_prompt, user_prompt = get_observe_visit_prompts(
            url=url,
            response=response,
            original_query=original_query,
            intent=intent,
            memory_summary=memory_summary,
            current_strategy=current_strategy,
            prompt_profile=getattr(self.attacker, "prompt_profile", "rich"),
        )

        decision, meta = self._run_with_prompt_metadata(system_prompt, user_prompt)

        # 强制约束
        decision.attack = False
        decision.payload = None

        return decision, meta
