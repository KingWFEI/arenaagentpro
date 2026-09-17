from __future__ import annotations

from typing import Any

from arenaagent.preliminary_baseline_agent.text_client import build_text_client_from_env


DEFAULT_NPC_TEXT_MODEL = "deepseek-flash"
DEFAULT_NPC_TEXT_API_BASE = "https://api.deepseek.com"
DEFAULT_NPC_TEXT_TIMEOUT_SECONDS = 35.0


def build_npc_text_client_from_env() -> Any | None:
    """Build a fast text-only decider so the final verdict skips the slow visual model."""
    return build_text_client_from_env(
        enable_env="NPC_ENABLE_TEXT_DECIDER",
        prefix="NPC_TEXT",
        label="independent NPC text decider",
        default_model=DEFAULT_NPC_TEXT_MODEL,
        default_api_base=DEFAULT_NPC_TEXT_API_BASE,
        default_timeout_seconds=DEFAULT_NPC_TEXT_TIMEOUT_SECONDS,
        api_key_fallbacks=("DEEPSEEK_API_KEY",),
    )
