from __future__ import annotations

from typing import Any

from arenaagent.preliminary_baseline_agent.tasks.base import action_succeeded


class RoomScanner:
    """无可执行计划时用 45 度小步探索；信息齐全时允许提前结束。"""

    def __init__(self, turns_required: int = 8, turn_degrees: int = 45) -> None:
        self.turns_required = turns_required
        self.turn_degrees = turn_degrees
        self.turns_completed = 0
        self.failed_turns = 0
        self.finished_early = False
        self.early_stop_reason: str | None = None

    @property
    def complete(self) -> bool:
        return self.finished_early or self.turns_completed >= self.turns_required

    def next_action(self) -> dict[str, Any] | None:
        if self.complete:
            return None
        return {
            "action": "turn_in_degree",
            "parameters": {"degree": self.turn_degrees},
            "output": 0,
            "think": (
                f"当前视野暂无可执行的目标与目的地组合，机会式扫描 "
                f"{self.turns_completed + 1}/{self.turns_required}。"
            ),
        }

    def after_action(self, result: Any) -> None:
        # 即使一次转向返回失败也继续推进，避免扫描阶段永久卡死。
        if not action_succeeded(result):
            self.failed_turns += 1
        self.turns_completed += 1

    def finish_early(self, reason: str) -> None:
        """目标和必需家具均已建图时，不再为了凑满一圈继续旋转。"""
        if self.complete:
            return
        self.finished_early = True
        self.early_stop_reason = reason

    def state(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "turns_completed": self.turns_completed,
            "turns_required": self.turns_required,
            "failed_turns": self.failed_turns,
            "turn_degrees": self.turn_degrees,
            "finished_early": self.finished_early,
            "early_stop_reason": self.early_stop_reason,
        }
