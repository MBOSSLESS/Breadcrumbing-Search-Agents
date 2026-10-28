"""Public model aliases and endpoint configuration (shared across roles).

Override an alias with BREADCRUMBING_MODEL_QWEN3, BREADCRUMBING_MODEL_KIMI,
or BREADCRUMBING_MODEL_QWEN3_6_PLUS. Values use backend:model syntax.
Reasoning can be overridden through extra_body.reasoning per call, or globally
with BREADCRUMBING_REASONING_EFFORT; OPENROUTER_DISABLE_REASONING_POLICY=1
omits defaults. These examples do not guarantee provider availability.
"""

MODEL_MAP = {
    "qwen3": "openrouter:qwen/qwen3-235b-a22b-2507",
    "kimi": "openrouter:moonshotai/kimi-k2.5",
    "qwen3.6-plus": "openrouter:qwen/qwen3.6-plus",
}
BACKEND_CONFIG = {
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url_env": "OPENROUTER_BASE_URL",
        "default_base_url": "https://openrouter.ai/api/v1",
        "supports_structured_output": False,
    },
    "oai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "default_base_url": "https://api.openai.com/v1",
        "supports_structured_output": True,
    },
    "vllm": {
        "api_key_env": None,
        "base_url_env": "VLLM_BASE_URL",
        "default_base_url": "http://localhost:8000/v1",
        "dummy_api_key": "unused",
        "supports_structured_output": False,
    },
    "ollama": {
        "api_key_env": None,
        "base_url_env": "OLLAMA_BASE_URL",
        "default_base_url": "http://localhost:11434/v1",
        "dummy_api_key": "unused",
        "supports_structured_output": False,
    },
}
REASONING_POLICIES = {
    "qwen/qwen3-235b-a22b-2507": {"effort": "none"},
}
