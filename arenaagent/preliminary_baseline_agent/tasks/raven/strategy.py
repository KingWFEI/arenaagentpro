from __future__ import annotations

from typing import Any

from arenaagent.preliminary_baseline_agent.task_runtime import TaskContext
from arenaagent.preliminary_baseline_agent.tasks.base import TaskStrategy


class RavenStrategy(TaskStrategy):
    """Raven policy: dispatch the isolated hybrid solver without a generic-agent turn."""

    task_type = "raven"
    history_message_limit = 0

    def reset(self, subject: dict[str, Any]) -> None:
        super().reset(subject)
        self.solve_attempts = 0

    def next_local_action(self, context: TaskContext) -> dict[str, Any] | None:
        del context
        self.solve_attempts += 1
        return {
            "action": "solve_raven",
            "parameters": {},
            "output": 0,
            "think": "使用瑞文专用混合求解器，完成候选评分、置信融合与答案编号校验。",
        }

    def validate_action(self, action: dict[str, Any], context: TaskContext | None) -> dict[str, Any]:
        del context
        # Do not rewrite malformed output silently; keeping it visible makes model/prompt regressions debuggable.
        return action

    def state_for_prompt(self) -> dict[str, Any]:
        return {"step_index": self.step_index, "solve_attempts": self.solve_attempts}
