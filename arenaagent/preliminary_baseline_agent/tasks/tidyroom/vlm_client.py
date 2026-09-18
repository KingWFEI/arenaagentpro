from __future__ import annotations

from typing import Any

from arenaagent.preliminary_baseline_agent.aux_client import build_aux_client_from_env

DEFAULT_TIDYROOM_VLM_MODEL = "glm-4.6v-flashx"
DEFAULT_TIDYROOM_VLM_API_BASE = "https://open.bigmodel.cn/api/paas/v4/"
DEFAULT_TIDYROOM_VLM_TIMEOUT_SECONDS = 90.0


def build_tidyroom_vision_client_from_env() -> Any | None:
    """整理房间的专属视觉模型。

    每发现一个待整理物品就要问一次模型（912 不提供目标清单，语义只能由模型
    补充），而主视觉模型是推理模型，单次 70 秒以上，400 秒的题目跑不完几个。
    这里换成快得多的视觉模型；其余任务仍用主模型。
    """
    return build_aux_client_from_env(
        enable_env="TIDYROOM_ENABLE_VLM",
        prefix="TIDYROOM_VLM",
        label="tidy-room dedicated vision model",
        default_model=DEFAULT_TIDYROOM_VLM_MODEL,
        default_api_base=DEFAULT_TIDYROOM_VLM_API_BASE,
        default_timeout_seconds=DEFAULT_TIDYROOM_VLM_TIMEOUT_SECONDS,
        api_key_fallbacks=("ZHIPU_API_KEY",),
    )
