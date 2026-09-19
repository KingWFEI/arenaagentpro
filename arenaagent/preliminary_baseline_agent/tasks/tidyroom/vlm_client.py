from __future__ import annotations

from typing import Any

from arenaagent.preliminary_baseline_agent.aux_client import build_aux_client_from_env

DEFAULT_TIDYROOM_VLM_MODEL = "kimi-k2.6"
DEFAULT_TIDYROOM_VLM_API_BASE = "https://api.moonshot.cn/v1"
DEFAULT_TIDYROOM_VLM_TIMEOUT_SECONDS = 45.0


def build_tidyroom_vision_client_from_env() -> Any | None:
    """整理房间的专属视觉模型。

    912 不提供目标清单，初始稳定帧的语义盘点必须由视觉模型补充。Kimi K3
    默认进行较重推理，在比赛时限内延迟过高；这里独立使用支持图片输入且
    可以关闭思考模式的 Kimi K2.6，避免改变其他赛题的主模型。
    """
    return build_aux_client_from_env(
        enable_env="TIDYROOM_ENABLE_VLM",
        prefix="TIDYROOM_VLM",
        label="tidy-room Kimi K2.6 non-thinking vision model",
        default_model=DEFAULT_TIDYROOM_VLM_MODEL,
        default_api_base=DEFAULT_TIDYROOM_VLM_API_BASE,
        default_timeout_seconds=DEFAULT_TIDYROOM_VLM_TIMEOUT_SECONDS,
        api_key_fallbacks=("VLM_CLIENT_CFG_API_KEY",),
        preferred_api_key_env="VLM_CLIENT_CFG_API_KEY",
        chat_completion_kwargs={
            "extra_body": {"thinking": {"type": "disabled"}},
            "max_tokens": 2048,
        },
    )
