from __future__ import annotations

from typing import Any

from arenaagent.preliminary_baseline_agent.aux_client import build_aux_client_from_env

DEFAULT_RAVEN_VLM_MODEL = "kimi-k3"
DEFAULT_RAVEN_VLM_API_BASE = "https://api.moonshot.cn/v1"
DEFAULT_RAVEN_VLM_TIMEOUT_SECONDS = 30.0


def build_raven_vision_client_from_env() -> Any | None:
    """Build the Raven-specific K3 whole-image client without changing other tasks."""
    return build_aux_client_from_env(
        enable_env="RAVEN_ENABLE_VLM",
        prefix="RAVEN_VLM",
        label="Raven Kimi K3 non-thinking whole-image vision model",
        default_model=DEFAULT_RAVEN_VLM_MODEL,
        default_api_base=DEFAULT_RAVEN_VLM_API_BASE,
        default_timeout_seconds=DEFAULT_RAVEN_VLM_TIMEOUT_SECONDS,
        api_key_fallbacks=("VLM_CLIENT_CFG_API_KEY",),
        preferred_api_key_env="VLM_CLIENT_CFG_API_KEY",
        chat_completion_kwargs={
            "extra_body": {"thinking": {"type": "disabled"}},
            "max_tokens": 256,
        },
    )
