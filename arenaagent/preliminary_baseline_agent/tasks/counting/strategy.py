from __future__ import annotations

from typing import Any

from arenaagent.preliminary_baseline_agent.task_runtime import TaskContext
from arenaagent.preliminary_baseline_agent.tasks.base import TaskStrategy


class CountingStrategy(TaskStrategy):
    """Counting-specific observation bookkeeping."""

    task_type = "counting"

    def reset(self, subject: dict[str, Any]) -> None:
        super().reset(subject)
        self.max_visible_count = 0

    def observe(self, context: TaskContext) -> None:
        super().observe(context)
        self.max_visible_count = max(self.max_visible_count, len(context.visible_objects))

    def state_for_prompt(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "current_visible_count": len(self.last_context.visible_objects) if self.last_context else 0,
            "max_visible_count": self.max_visible_count,
        }

