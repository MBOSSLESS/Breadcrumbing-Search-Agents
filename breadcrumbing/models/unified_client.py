"""
Unified LLM Client for DeepResearch

Provides a unified interface for multiple LLM providers (OpenAI, Ollama, Qwen, OpenRouter).
All backends are OpenAI-compatible - only api_key and base_url differ.

Key Features:
- Single interface for multiple LLM backends
- Model name format: "backend:model" (e.g., "ollama:qwen2.5:7b", "oai:gpt-4")
- Support for structured output with Pydantic models
- Easy to extend with new backends (just add config entry)

Usage:
    from breadcrumbing.models import unified_client

    # Standard chat completion
    response = unified_client.chat_completions_create(
        model_name="ollama:qwen2.5:7b",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7
    )

    # Structured output (requires Pydantic model)
    from pydantic import BaseModel

    class Response(BaseModel):
        answer: str
        confidence: float

    result = unified_client.beta_chat_completions_parse(
        model_name="ollama:qwen2.5:7b",
        messages=[{"role": "user", "content": "What is AI?"}],
        response_format=Response,
        temperature=0.5
    )
    print(result.answer, result.confidence)
"""

import copy
import json
import logging
import os
import re
import time
from openai import OpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError

from breadcrumbing.runtime_trace import trace_span

logger = logging.getLogger(__name__)

OPENROUTER_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def _openrouter_env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in OPENROUTER_TRUTHY_ENV_VALUES


def _openai_client_timeout_seconds() -> float | None:
    raw_value = os.getenv("OPENAI_CLIENT_TIMEOUT_SECONDS") or os.getenv(
        "LLM_CLIENT_TIMEOUT_SECONDS"
    )
    if raw_value is None or raw_value.strip() == "":
        return 180.0
    clean = raw_value.strip().lower()
    if clean in {"0", "none", "off", "false"}:
        return None
    try:
        return float(clean)
    except ValueError:
        logger.warning(
            "Ignoring invalid OpenAI client timeout=%r; using 180s", raw_value
        )
        return 180.0


# Public routing table; shared by every model role.
from .config import (
    MODEL_MAP,
    BACKEND_CONFIG as _BACKEND_CONFIG,
    REASONING_POLICIES as OPENROUTER_REASONING_POLICIES,
)

# =============================================================================
# Client Factory
# =============================================================================
_clients: dict[str, OpenAI] = {}


def get_client_by_model(model_name: str) -> OpenAI:
    """
    Get the OpenAI client for a given model name.

    Args:
        model_name: Model name in "backend:model" format
            Examples:
                - "ollama:qwen2.5:7b" -> returns OpenAI client for ollama
                - "oai:gpt-4" -> returns OpenAI client for OpenAI
                - "openrouter:qwen/qwen3-235b-a22b-2507" -> returns the OpenRouter client

    Returns:
        OpenAI client instance configured for the backend

    Raises:
        ValueError: If model_name doesn't contain ":" or backend is not supported

    Example:
        >>> client = get_client_by_model("ollama:qwen2.5:7b")
        >>> response = client.chat.completions.create(
        ...     model="qwen2.5:7b",
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
    """
    backend, _ = classify_model_backend(model_name)
    return _get_openai_client(backend)


def _get_openai_client(backend: str) -> OpenAI:
    """
    Get or create an OpenAI client for the given backend.

    Args:
        backend: Backend name (e.g., "oai", "qwen", "ollama", "openrouter")

    Returns:
        OpenAI client instance configured for the backend

    Raises:
        ValueError: If backend is not supported or API key is missing
    """
    if backend not in _BACKEND_CONFIG:
        raise ValueError(
            f"Unsupported backend '{backend}'. "
            f"Valid backends: {', '.join(_BACKEND_CONFIG.keys())}"
        )

    if backend not in _clients:
        config = _BACKEND_CONFIG[backend]

        # Get base URL (from env or default)
        base_url = (
            os.getenv(config["base_url_env"]) if config.get("base_url_env") else None
        ) or config.get("default_base_url")

        # Get API key (from env or dummy)
        if config.get("api_key_env"):
            api_key = os.getenv(config["api_key_env"])
            if api_key is None:
                raise ValueError(
                    f"API key not found for backend '{backend}'. "
                    f"Please set environment variable: {config['api_key_env']}"
                )
        else:
            api_key = config.get("dummy_api_key", "dummy")

        client_kwargs = {"api_key": api_key, "base_url": base_url}
        timeout_seconds = _openai_client_timeout_seconds()
        if timeout_seconds is not None:
            client_kwargs["timeout"] = timeout_seconds
        _clients[backend] = OpenAI(**client_kwargs)

    return _clients[backend]


def normalize_model_name(model_name: str) -> str:
    """
    Normalize model name:
    - If alias -> map to full name
    - If already contains backend prefix -> return as is
    """
    if not model_name:
        raise ValueError("model_name cannot be empty")

    # already full format
    if ":" in model_name:
        prefix = model_name.split(":", 1)[0]
        if prefix in _BACKEND_CONFIG:
            return model_name
    # Alias overrides use the same routing layer for every role.
    if model_name in MODEL_MAP:
        env_name = (
            "BREADCRUMBING_MODEL_" + re.sub(r"[^A-Za-z0-9]", "_", model_name).upper()
        )
        return os.getenv(env_name, MODEL_MAP[model_name])
    if "/" in model_name or model_name.startswith("."):
        return "vllm:" + model_name

    raise ValueError(
        f"Unknown model alias '{model_name}'. "
        f"Available aliases: {', '.join(MODEL_MAP.keys())}"
    )


# =============================================================================
# Public Interface
# =============================================================================
def classify_model_backend(model_name: str) -> tuple[str, str]:
    """
    Parse model name to extract backend and model identifier.

    Args:
        model_name: Model name in format "backend:model"
            Examples:
                - "ollama:qwen2.5:7b" -> ("ollama", "qwen2.5:7b")
                - "oai:gpt-4" -> ("oai", "gpt-4")
                - "vllm:local-model" -> ("vllm", "local-model")

    Returns:
        Tuple of (backend, model_name)

    Raises:
        ValueError: If model_name doesn't contain ":" or backend is not supported

    Supported Backends:
        - oai: OpenAI and OpenAI-compatible APIs
        - vllm: Local OpenAI-compatible model server
        - ollama: Local Ollama models
        - openrouter: OpenRouter API
    """
    model_name = normalize_model_name(model_name)

    backend, model = model_name.split(":", 1)
    backend = backend.lower()

    if backend not in _BACKEND_CONFIG:
        raise ValueError(
            f"Backend '{backend}' is not supported. "
            f"Supported backends: {', '.join(_BACKEND_CONFIG.keys())}"
        )

    return backend, model


class UnifiedLLMClient:
    """
    Unified LLM client supporting multiple backends.

    This class provides a consistent interface for working with different
    LLM providers through the OpenAI SDK. It abstracts away the differences
    between providers while supporting advanced features like structured output.

    Example:
        >>> client = UnifiedLLMClient()
        >>> response = client.chat_completions_create(
        ...     model_name="ollama:qwen2.5:7b",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
        >>> print(response.choices[0].message.content)
    """

    def _wrap_think(self, response):
        """
        如果 content 中有 <think>...</think> 且没有 reasoning 字段，
        提取 think 块作为 reasoning，同时清理 content。
        """
        choices = getattr(response, "choices", None)
        if not choices:
            return response

        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            return response

        content = getattr(message, "content", None) or ""

        # 如果已经有 reasoning 字段，不处理
        if hasattr(message, "reasoning") and message.reasoning:
            return response

        reasoning, cleaned_content = self._extract_think(content)
        message.reasoning = reasoning
        message.content = cleaned_content
        return response

    def _extract_think(self, content: str) -> tuple[str, str]:
        """
        提取 <think>...</think> 中的内容作为 reasoning。
        返回 (reasoning, content_without_think)
        """
        if not content:
            return "", ""

        pattern = r"<think>(.*?)</think>"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            reasoning = match.group(1).strip()
            content_without_think = re.sub(
                pattern, "", content, flags=re.DOTALL | re.IGNORECASE
            ).strip()
            return reasoning, content_without_think
        return "", content

    def _chat_completion_with_retry(self, client, model: str, backend: str, **kwargs):
        """
        Retry transient SDK/provider failures for standard chat completions.

        This specifically guards against cases where the provider returns HTTP 200
        but the body is empty/truncated/non-JSON, which surfaces as JSONDecodeError
        inside the OpenAI SDK response parser.
        """
        max_attempts = int(os.getenv("LLM_RETRY_MAX_ATTEMPTS", "3"))
        retryable_errors = (json.JSONDecodeError, APIConnectionError, APITimeoutError)
        retryable_status_codes = {408, 409, 429, 500, 502, 503, 504}
        last_error = None

        for attempt in range(1, max_attempts + 1):
            try:
                with trace_span(
                    "llm.transport",
                    backend=backend,
                    resolved_model=model,
                    transport_attempt=attempt,
                    max_transport_attempts=max_attempts,
                ) as transport_span:
                    response = client.chat.completions.create(model=model, **kwargs)
                    usage = getattr(response, "usage", None)
                    transport_span.update(
                        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                        completion_tokens=int(
                            getattr(usage, "completion_tokens", 0) or 0
                        ),
                        finish_reason=(
                            getattr(response.choices[0], "finish_reason", None)
                            if getattr(response, "choices", None)
                            else None
                        ),
                    )
                    return response
            except APIStatusError as exc:
                if exc.status_code not in retryable_status_codes:
                    raise
                last_error = exc
                if attempt >= max_attempts:
                    break
                base_sleep = 5 if exc.status_code in {429, 503, 504} else 2
                sleep_seconds = min(base_sleep * (2 ** (attempt - 1)), 30)
                logger.warning(
                    "Chat completion failed for backend=%s model=%s on attempt %s/%s "
                    "with retryable HTTP status %s: %s. Retrying in %ss.",
                    backend,
                    model,
                    attempt,
                    max_attempts,
                    exc.status_code,
                    exc,
                    sleep_seconds,
                )
                with trace_span(
                    "llm.retry_backoff",
                    backend=backend,
                    resolved_model=model,
                    retry_after_transport_attempt=attempt,
                ):
                    time.sleep(sleep_seconds)
            except retryable_errors as exc:
                last_error = exc
                if attempt >= max_attempts:
                    break
                sleep_seconds = min(2 ** (attempt - 1), 4)
                logger.warning(
                    "Chat completion failed for backend=%s model=%s on attempt %s/%s "
                    "with retryable error %s: %s. Retrying in %ss.",
                    backend,
                    model,
                    attempt,
                    max_attempts,
                    type(exc).__name__,
                    exc,
                    sleep_seconds,
                )
                with trace_span(
                    "llm.retry_backoff",
                    backend=backend,
                    resolved_model=model,
                    retry_after_transport_attempt=attempt,
                ):
                    time.sleep(sleep_seconds)

        assert last_error is not None
        raise last_error

    def _normalize_backend_kwargs(self, backend: str, model: str, kwargs: dict) -> dict:
        """
        Apply narrow backend/model compatibility shims before sending requests.
        """
        kwargs = dict(kwargs)

        if backend == "openrouter":
            kwargs = self._apply_openrouter_reasoning_policy(model, kwargs)
        return kwargs

    def _apply_openrouter_reasoning_policy(self, model: str, kwargs: dict) -> dict:
        if _openrouter_env_flag("OPENROUTER_DISABLE_REASONING_POLICY"):
            return kwargs

        policy = OPENROUTER_REASONING_POLICIES.get(model)
        effort = os.getenv("BREADCRUMBING_REASONING_EFFORT")
        if effort:
            policy = {"effort": effort}
        if not policy:
            return kwargs

        extra_body = copy.deepcopy(kwargs.get("extra_body") or {})
        if "reasoning" in extra_body:
            return kwargs

        extra_body["reasoning"] = copy.deepcopy(policy)
        kwargs["extra_body"] = extra_body
        return kwargs

    @staticmethod
    def _normalize_text_tool_calls(response, request_kwargs):
        """Adapt unsolicited native function calls to the existing text protocol."""
        extra_body = request_kwargs.get("extra_body") or {}
        if any(
            key in request_kwargs or key in extra_body for key in ("tools", "functions")
        ):
            return response
        if not getattr(response, "choices", None):
            return response
        message = response.choices[0].message
        content = message.content or ""
        calls = getattr(message, "tool_calls", None)
        if not calls or "<tool_call>" in content:
            return response

        blocks = []
        for call in calls:
            function = getattr(call, "function", None)
            name = getattr(function, "name", None)
            if (
                getattr(call, "type", None) != "function"
                or not isinstance(name, str)
                or not name
            ):
                raise ValueError("Native tool call requires a function name")
            arguments = getattr(function, "arguments", None)
            if not isinstance(arguments, str):
                raise ValueError(
                    "Native tool call arguments must be a JSON object string"
                )
            try:
                arguments = json.loads(arguments)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    "Native tool call arguments must be valid JSON"
                ) from exc
            if not isinstance(arguments, dict):
                raise ValueError("Native tool call arguments must be a JSON object")
            payload = json.dumps(
                {"name": name, "arguments": arguments}, ensure_ascii=False
            )
            # A literal protocol delimiter inside an argument must remain JSON data.
            payload = payload.replace("<", "\\u003c").replace(">", "\\u003e")
            blocks.append(f"<tool_call>\n{payload}\n</tool_call>")

        # Validate the entire response before exposing any executable text.
        message.content = "\n".join(([content] if content else []) + blocks)
        return response

    def chat_completions_create(self, model_name: str, **kwargs):
        """
        Create a standard chat completion.

        Args:
            model_name: Model name in "backend:model" format
            **kwargs: Additional arguments passed to OpenAI SDK
                - messages: List of message dicts (required)
                - temperature: Sampling temperature (0.0 to 2.0)
                - max_tokens: Maximum tokens in response
                - top_p: Nucleus sampling parameter
                - And other provider-specific parameters

        Returns:
            OpenAI-style completion object with choices, usage, etc.

        Raises:
            ValueError: If backend is not supported
            Exception: If the API call fails

        Example:
            >>> response = client.chat_completions_create(
            ...     model_name="ollama:qwen2.5:7b",
            ...     messages=[
            ...         {"role": "system", "content": "You are a helpful assistant."},
            ...         {"role": "user", "content": "Explain quantum computing."}
            ...     ],
            ...     temperature=0.7,
            ...     max_tokens=500
            ... )
        """
        backend, model = classify_model_backend(model_name)
        client = _get_openai_client(backend)
        kwargs = self._normalize_backend_kwargs(backend, model, kwargs)
        with trace_span(
            "llm.client",
            backend=backend,
            requested_model=model_name,
            resolved_model=model,
            max_tokens=kwargs.get("max_tokens"),
            temperature=kwargs.get("temperature"),
        ):
            response = self._chat_completion_with_retry(
                client=client,
                model=model,
                backend=backend,
                **kwargs,
            )

        # ---------- 统一处理 think 块 ----------
        response = self._wrap_think(response)
        response = self._normalize_text_tool_calls(response, kwargs)
        return response

    def beta_chat_completions_parse(self, model_name: str, **kwargs):
        """
        Create a chat completion with structured output parsing.
        For backends that do not support structured parsing, fallback to manual JSON + think extraction.

        Returns unified response with:
            response.choices[0].message.parsed
            response.choices[0].message.reasoning
            response.choices[0].message.content
        """

        backend, model = classify_model_backend(model_name)
        client = _get_openai_client(backend)
        kwargs = self._normalize_backend_kwargs(backend, model, kwargs)

        response_format = kwargs.get("response_format")
        if not response_format:
            raise ValueError(
                "response_format is required for beta_chat_completions_parse"
            )

        supports_structured = _BACKEND_CONFIG[backend].get(
            "supports_structured_output", False
        )

        if supports_structured:
            return client.beta.chat.completions.parse(model=model, **kwargs)

        else:
            # fallback: 普通 chat completion + JSON / think 解析
            fallback_kwargs = {
                k: v for k, v in kwargs.items() if k != "response_format"
            }
            with trace_span(
                "llm.client",
                backend=backend,
                requested_model=model_name,
                resolved_model=model,
                structured_fallback=True,
                response_schema=getattr(
                    response_format, "__name__", str(response_format)
                ),
                max_tokens=fallback_kwargs.get("max_tokens"),
                temperature=fallback_kwargs.get("temperature"),
            ):
                response = self._chat_completion_with_retry(
                    client=client,
                    model=model,
                    backend=backend,
                    **fallback_kwargs,
                )
                content = response.choices[0].message.content

                with trace_span(
                    "llm.parse_validate",
                    response_schema=getattr(
                        response_format, "__name__", str(response_format)
                    ),
                ):
                    # 提取 think + JSON
                    parsed_json, reasoning, cleaned_content = (
                        self._extract_json_and_think(content)
                    )

                    # 转为 Pydantic 模型. Some backends occasionally wrap the real
                    # structured payload as {"result": "<json string>"}. Validate the
                    # original payload first, then try conservative unwrap candidates.
                    parsed_obj = self._parse_structured_payload(
                        parsed_json, response_format
                    )

            # 创建 MockResponse 保持统一接口
            class MockMessage:
                def __init__(self, parsed, reasoning, content):
                    self.parsed = parsed
                    self.reasoning = reasoning
                    self.content = content

            class MockChoice:
                def __init__(self, parsed, reasoning, content):
                    self.message = MockMessage(parsed, reasoning, content)
                    self.finish_reason = response.choices[0].finish_reason
                    self.logprobs = getattr(response.choices[0], "logprobs", None)

            class MockResponse:
                def __init__(self, original_response, parsed, reasoning, content):
                    self.choices = [MockChoice(parsed, reasoning, content)]
                    self.id = getattr(original_response, "id", None)
                    self.created = getattr(original_response, "created", None)
                    self.model = getattr(original_response, "model", None)
                    self.object = "chat.completion"
                    self.usage = getattr(original_response, "usage", None)

            return MockResponse(response, parsed_obj, reasoning, cleaned_content)

    def _parse_structured_payload(self, parsed_json: dict, response_format):
        if not parsed_json:
            return response_format()

        errors = []
        for candidate in self._structured_payload_candidates(parsed_json):
            try:
                return response_format(**candidate)
            except Exception as exc:
                errors.append(exc)

        if errors:
            raise errors[0]
        return response_format()

    def _structured_payload_candidates(self, parsed_json: dict) -> list[dict]:
        candidates: list[dict] = []
        seen: set[str] = set()

        def add_candidate(payload):
            if not isinstance(payload, dict):
                return
            for candidate in (payload, self._sanitize_structured_payload(payload)):
                marker = json.dumps(
                    candidate, sort_keys=True, ensure_ascii=False, default=str
                )
                if marker not in seen:
                    seen.add(marker)
                    candidates.append(candidate)

        def parse_json_object(text: str):
            current = str(text or "").strip()
            for _ in range(3):
                if not current:
                    return None
                current = self._extract_json_from_markdown(current).strip()
                try:
                    loaded = json.loads(current)
                except json.JSONDecodeError:
                    loaded = self._load_json_object_from_text(current)
                    if loaded is None:
                        return None
                if isinstance(loaded, dict):
                    return loaded
                if isinstance(loaded, str):
                    current = loaded.strip()
                    continue
                return None
            return None

        add_candidate(parsed_json)
        if isinstance(parsed_json, dict) and "result" in parsed_json:
            result = parsed_json.get("result")
            if isinstance(result, dict):
                add_candidate(result)
            elif isinstance(result, str):
                add_candidate(parse_json_object(result))
                extracted, _, _ = self._extract_json_and_think(result)
                add_candidate(extracted)
                add_candidate(self._load_json_object_from_text(result))
                add_candidate(self._recover_structured_fields_from_text(result))

        return candidates

    def _recover_structured_fields_from_text(self, text: str):
        """Recover common structured judge fields from malformed JSON fragments."""
        current = str(text or "").strip()
        if not current:
            return None

        payload = {}
        success_match = re.search(
            r'"?success"?\s*:\s*(true|false)',
            current,
            flags=re.IGNORECASE,
        )
        if success_match:
            payload["success"] = success_match.group(1).lower() == "true"

        reasoning_match = re.search(
            r'"?judge_reasoning"?\s*:\s*"(?P<reason>(?:\\.|[^"\\])*)"',
            current,
            flags=re.DOTALL,
        )
        if reasoning_match:
            try:
                payload["judge_reasoning"] = json.loads(
                    f'"{reasoning_match.group("reason")}"'
                )
            except json.JSONDecodeError:
                payload["judge_reasoning"] = reasoning_match.group("reason")
        elif "success" in payload:
            payload["judge_reasoning"] = (
                "Recovered from malformed structured judge output."
            )

        return payload or None

    def _load_json_object_from_text(self, text: str):
        current = str(text or "").strip()
        if not current:
            return None

        decoder = json.JSONDecoder()
        starts = [idx for idx, char in enumerate(current) if char == "{"]
        for start in starts:
            try:
                loaded, _ = decoder.raw_decode(current[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(loaded, dict):
                return loaded

        first = current.find("{")
        last = current.rfind("}")
        if first >= 0 and last > first:
            try:
                loaded = json.loads(current[first : last + 1])
            except json.JSONDecodeError:
                return None
            if isinstance(loaded, dict):
                return loaded
        return None

    def _sanitize_structured_payload(self, payload):
        if isinstance(payload, dict):
            sanitized = {}
            for key, value in payload.items():
                clean_key = key
                if isinstance(clean_key, str):
                    clean_key = re.sub(r"(?:游戏副本|副本)+$", "", clean_key)
                sanitized[clean_key] = self._sanitize_structured_payload(value)
            return sanitized
        if isinstance(payload, list):
            return [self._sanitize_structured_payload(item) for item in payload]
        return payload

    def _extract_json_from_markdown(self, content: str) -> str:
        """
        Extract JSON content from markdown code blocks.

        Handles formats like:
        ```json
        {...}
        ```
        or
        ```
        {...}
        ```

        Args:
            content: String potentially containing markdown-wrapped JSON

        Returns:
            Clean JSON string
        """
        if not content:
            return content

        content = content.strip()

        # Pattern to match ```json or ``` code blocks
        patterns = [
            r"```json\s*\n(.*?)\n```",  # ```json ... ```
            r"```\s*\n(.*?)\n```",  # ``` ... ```
            r"```json(.*?)```",  # ```json...``` (single line)
            r"```(.*?)```",  # ```...``` (single line)
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                # Verify it looks like JSON
                if extracted.startswith(("{", "[")):
                    return extracted

        # If no markdown wrapper found, return original
        return content

    def _extract_json_and_think(self, content: str) -> tuple[dict, str, str]:

        reasoning = ""
        if not content:
            return {}, reasoning, ""

        # 1. 提取 think
        think_pattern = r"<think>(.*?)</think>"
        match = re.search(think_pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            reasoning = match.group(1).strip()
            content = re.sub(
                think_pattern, "", content, flags=re.DOTALL | re.IGNORECASE
            )

        # 2. 提取 JSON
        json_str = self._extract_json_from_markdown(content)
        parsed_data = {}
        if json_str:
            try:
                parsed_data = json.loads(json_str)
            except json.JSONDecodeError:
                parsed_data = {}

        # fallback: 如果没有 JSON，至少返回 content 到 result
        if not parsed_data:
            parsed_data = {"result": content.strip()}

        # 3. cleaned content 去掉 markdown
        cleaned_content = re.sub(r"```.*?```", "", content, flags=re.DOTALL).strip()

        return parsed_data, reasoning, cleaned_content


# =============================================================================
# Singleton Instance
# =============================================================================
_unified_client: UnifiedLLMClient | None = None


def get_unified_client() -> UnifiedLLMClient:
    """
    Get or create the unified client instance (lazy initialization).

    Returns:
        UnifiedLLMClient instance

    Note:
        Client is created on first access to avoid requiring API keys at import time.
    """
    global _unified_client
    if _unified_client is None:
        _unified_client = UnifiedLLMClient()
    return _unified_client


class _UnifiedClientProxy:
    """Initialize the shared client lazily on its first use."""

    def __call__(self) -> UnifiedLLMClient:
        return get_unified_client()

    def __getattr__(self, name: str):
        """Proxy all attribute and method access to the actual client."""
        return getattr(get_unified_client(), name)


# Global singleton instance
unified_client = _UnifiedClientProxy()
