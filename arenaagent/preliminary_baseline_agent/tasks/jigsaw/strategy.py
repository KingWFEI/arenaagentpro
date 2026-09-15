from __future__ import annotations

from typing import Any

from arenaagent.preliminary_baseline_agent.task_runtime import TaskContext
from arenaagent.preliminary_baseline_agent.tasks.base import TaskStrategy, action_succeeded

_BOUND_VALUE_COUNT = 4


class JigsawStrategy(TaskStrategy):
    """Jigsaw-specific geometry prompt and placement progress."""

    task_type = "jigsaw"

    def reset(self, subject: dict[str, Any]) -> None:
        super().reset(subject)
        self.successful_placements = 0
        self.reference_bounding = list(subject.get("reference_bounding") or [])

    def build_task_prompt(self, context: TaskContext) -> str:
        prompt = self.load_prompt()
        bounds = context.subject.get("reference_bounding") or self.reference_bounding
        if len(bounds) >= _BOUND_VALUE_COUNT:
            region = f"[Y: {bounds[0]:.1f} ~ {bounds[2]:.1f}, Z: {bounds[3]:.1f} ~ {bounds[1]:.1f}]"
            prompt = f"{prompt}\n本题拼图目标区域为 {region}。"
        return prompt

    def after_action(self, action: dict[str, Any], result: Any, context: TaskContext | None) -> None:
        del context
        name = str(action.get("action") or "").lower()
        if name in {"move_and_put_down", "put_down_to_location"} and action_succeeded(result):
            self.successful_placements += 1

    def state_for_prompt(self) -> dict[str, Any]:
        return {"step_index": self.step_index, "successful_placements": self.successful_placements}
