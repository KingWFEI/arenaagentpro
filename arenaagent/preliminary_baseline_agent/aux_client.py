from __future__ import annotations

import os
from typing import Any

from loguru import logger

from arenaagent.vlm_agent.client import ClientFactory
from arenaagent.vlm_agent.vlm_config import VLMClientCfg


def build_aux_client_from_env(
    *,
    enable_env: str,
    prefix: str,
    label: str,
    default_model: str,
    default_api_base: str,
    default_timeout_seconds: float = 35.0,
    api_key_fallbacks: tuple[str, ...] = (),
) -> Any | None:
    """Build an independent OpenAI-compatible client from <prefix>_* env vars.

    文本复核与任务专属视觉模型共用这一套构造；两者只是模型和 endpoint 不同。
    """
    enabled = os.getenv(enable_env, "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        logger.info("{} is disabled by environment", label)
        return None

    api_key = os.getenv(f"{prefix}_API_KEY", "").strip()
    for fallback in api_key_fallbacks:
        if api_key:
            break
        api_key = os.getenv(fallback, "").strip()
    if not api_key:
        hint = ", ".join((f"{prefix}_API_KEY", *api_key_fallbacks))
        logger.info("{} is not configured; set {}", label, hint)
        return None

    model = os.getenv(f"{prefix}_MODEL", "").strip() or default_model
    api_base = os.getenv(f"{prefix}_API_BASE", "").strip() or default_api_base
    try:
        timeout_seconds = float(
            os.getenv(f"{prefix}_TIMEOUT_SECONDS", str(default_timeout_seconds))
        )
    except ValueError:
        timeout_seconds = default_timeout_seconds
    timeout_seconds = min(max(timeout_seconds, 5.0), 90.0)

    cfg = VLMClientCfg()
    # Assign after construction so generic visual-model environment overrides
    # cannot redirect the independent text client to the visual endpoint.
    cfg.name = model
    cfg.api_base = api_base
    cfg.api_key = api_key
    cfg.message_role = "user"
    cfg.request_timeout_seconds = timeout_seconds
    cfg.native_max_retries = 0
    client = ClientFactory().build("openai", cfg)
    logger.info(
        "Configured {} model={} api_base={} timeout={}s",
        label,
        cfg.name,
        cfg.api_base,
        timeout_seconds,
    )
    return client
