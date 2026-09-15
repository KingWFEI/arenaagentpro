from __future__ import annotations

from copy import deepcopy
from typing import Any


class RecoveryPolicy:
    """对失败进行分类和计数；本地重试耗尽后才升级给视觉模型。"""

    def __init__(self, vlm_threshold: int = 3) -> None:
        self.vlm_threshold = vlm_threshold
        self.consecutive_failures: dict[str, int] = {}
        self.total_failures: dict[str, int] = {}
        self.last_failure: dict[str, dict[str, Any]] = {}
        # 同一失败次数只咨询一次 VLM。咨询后不清空失败计数，否则会形成
        # “本地失败三次 -> VLM 转身 -> 计数归零 -> 原方案再失败三次”的循环。
        self.last_vlm_failure_count: dict[str, int] = {}

    def record_failure(
        self,
        raw_id: str,
        reason: str,
        plan: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.consecutive_failures[raw_id] = self.consecutive_failures.get(raw_id, 0) + 1
        self.total_failures[raw_id] = self.total_failures.get(raw_id, 0) + 1
        failure = {"reason": reason, **deepcopy(details or {})}
        self.last_failure[raw_id] = failure
        if plan is not None:
            plan["attempt"] = int(plan.get("attempt", 0)) + 1
            plan["last_failure"] = failure

    def record_success(self, raw_id: str) -> None:
        self.consecutive_failures[raw_id] = 0
        self.last_vlm_failure_count.pop(raw_id, None)

    def needs_vlm(self, raw_id: str | None) -> bool:
        if raw_id is None:
            return False
        failure_count = self.consecutive_failures.get(raw_id, 0)
        last_consulted = self.last_vlm_failure_count.get(raw_id, 0)
        return (
            failure_count >= self.vlm_threshold
            and failure_count - last_consulted >= self.vlm_threshold
        )

    def mark_vlm_consulted(self, raw_id: str | None) -> None:
        if raw_id is not None:
            self.last_vlm_failure_count[raw_id] = self.consecutive_failures.get(raw_id, 0)

    def state(self) -> dict[str, Any]:
        return {
            "vlm_threshold": self.vlm_threshold,
            "consecutive_failures": dict(self.consecutive_failures),
            "total_failures": dict(self.total_failures),
            "last_failure": deepcopy(self.last_failure),
            "last_vlm_failure_count": dict(self.last_vlm_failure_count),
        }
