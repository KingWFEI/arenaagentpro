from __future__ import annotations

from typing import Any

from arenaagent.preliminary_baseline_agent.aux_client import build_aux_client_from_env


DEFAULT_RAVEN_TEXT_MODEL = "deepseek-v4-pro"
DEFAULT_RAVEN_TEXT_API_BASE = "https://api.deepseek.com"
DEFAULT_RAVEN_TEXT_TIMEOUT_SECONDS = 35.0


def build_raven_text_client_from_env() -> Any | None:
    """Build an independent DeepSeek verifier without reusing visual credentials."""
    return build_aux_client_from_env(
        enable_env="RAVEN_ENABLE_TEXT_VERIFIER",
        prefix="RAVEN_TEXT",
        label="independent Raven text verifier",
        default_model=DEFAULT_RAVEN_TEXT_MODEL,
        default_api_base=DEFAULT_RAVEN_TEXT_API_BASE,
        default_timeout_seconds=DEFAULT_RAVEN_TEXT_TIMEOUT_SECONDS,
        api_key_fallbacks=("DEEPSEEK_API_KEY",),
    )
