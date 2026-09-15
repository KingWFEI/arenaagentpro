from __future__ import annotations

from typing import Any

from arenaagent.preliminary_baseline_agent.task_runtime import TaskContext
from arenaagent.preliminary_baseline_agent.tasks.base import TaskStrategy, action_succeeded


class NpcStrategy(TaskStrategy):
    """Conversation memory isolated from physical manipulation tasks."""

    task_type = "npc"

    def reset(self, subject: dict[str, Any]) -> None:
        super().reset(subject)
        self.contacted_npcs: list[str] = []
        self.successful_dialogues = 0

    def after_action(self, action: dict[str, Any], result: Any, context: TaskContext | None) -> None:
        del context
        if str(action.get("action") or "").lower() != "speak_to_npc":
            return
        has_dialogue_result = isinstance(result, dict) and not result.get("error") and bool(
            result.get("npc_reply") or result.get("reply") or result.get("npc_name")
        )
        if not action_succeeded(result) and not has_dialogue_result:
            return
        params = action.get("parameters") or {}
        npc_name = str(params.get("npc_name") or params.get("npc") or params.get("target") or "").strip()
        if npc_name and npc_name not in self.contacted_npcs:
            self.contacted_npcs.append(npc_name)
        self.successful_dialogues += 1

    def state_for_prompt(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "contacted_npcs": self.contacted_npcs,
            "successful_dialogues": self.successful_dialogues,
        }
