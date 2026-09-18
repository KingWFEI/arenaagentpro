from __future__ import annotations

from typing import Any

from arenaagent.preliminary_baseline_agent.aux_client import build_aux_client_from_env


DEFAULT_COUNTING_REVIEW_MODEL = "kimi-k2.7-code-highspeed"
DEFAULT_COUNTING_REVIEW_API_BASE = "https://api.moonshot.cn/v1"
DEFAULT_COUNTING_REVIEW_TIMEOUT_SECONDS = 90.0


def build_counting_review_client_from_env() -> Any | None:
    """Build a fast vision reviewer so candidate verification skips the slow main model."""
    return build_aux_client_from_env(
        enable_env="COUNTING_ENABLE_REVIEW_MODEL",
        prefix="COUNTING_REVIEW",
        label="independent counting review model",
        default_model=DEFAULT_COUNTING_REVIEW_MODEL,
        default_api_base=DEFAULT_COUNTING_REVIEW_API_BASE,
        default_timeout_seconds=DEFAULT_COUNTING_REVIEW_TIMEOUT_SECONDS,
        api_key_fallbacks=("VLM_CLIENT_CFG_API_KEY",),
    )
