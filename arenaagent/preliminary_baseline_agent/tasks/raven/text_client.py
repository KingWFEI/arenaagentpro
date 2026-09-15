from __future__ import annotations

import os
from typing import Any

from loguru import logger

from arenaagent.vlm_agent.client import ClientFactory
from arenaagent.vlm_agent.vlm_config import VLMClientCfg


DEFAULT_RAVEN_TEXT_MODEL = "deepseek-v4-pro"
DEFAULT_RAVEN_TEXT_API_BASE = "https://api.deepseek.com"
DEFAULT_RAVEN_TEXT_TIMEOUT_SECONDS = 35.0


def build_raven_text_client_from_env() -> Any | None:
    """Build an independent DeepSeek verifier without reusing visual credentials."""
    enabled = os.getenv("RAVEN_ENABLE_TEXT_VERIFIER", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        logger.info("Raven independent text verifier is disabled by environment")
        return None

    api_key = os.getenv("RAVEN_TEXT_API_KEY", "").strip() or os.getenv(
        "DEEPSEEK_API_KEY", ""
    ).strip()
    if not api_key:
        logger.info(
            "Raven text verifier is not configured; set RAVEN_TEXT_API_KEY or DEEPSEEK_API_KEY"
        )
        return None

    model = os.getenv("RAVEN_TEXT_MODEL", DEFAULT_RAVEN_TEXT_MODEL).strip()
    api_base = os.getenv("RAVEN_TEXT_API_BASE", DEFAULT_RAVEN_TEXT_API_BASE).strip()
    try:
        timeout_seconds = float(
            os.getenv("RAVEN_TEXT_TIMEOUT_SECONDS", str(DEFAULT_RAVEN_TEXT_TIMEOUT_SECONDS))
        )
    except ValueError:
        timeout_seconds = DEFAULT_RAVEN_TEXT_TIMEOUT_SECONDS
    timeout_seconds = min(max(timeout_seconds, 5.0), 90.0)
    cfg = VLMClientCfg()
    # Assign after construction so generic visual-model environment overrides
    # cannot redirect the independent verifier to the visual endpoint.
    cfg.name = model or DEFAULT_RAVEN_TEXT_MODEL
    cfg.api_base = api_base or DEFAULT_RAVEN_TEXT_API_BASE
    cfg.api_key = api_key
    cfg.message_role = "user"
    cfg.request_timeout_seconds = timeout_seconds
    cfg.native_max_retries = 0
    client = ClientFactory().build("openai", cfg)
    logger.info(
        "Configured independent Raven text verifier model={} api_base={} timeout={}s",
        cfg.name,
        cfg.api_base,
        timeout_seconds,
    )
    return client
