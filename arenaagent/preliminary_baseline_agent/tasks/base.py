from __future__ import annotations

import json
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import Any

from arenaagent.preliminary_baseline_agent.task_runtime import TaskContext


class TaskStrategy:
    """Stable extension contract for one independently developed task."""

    task_type = "unknown"
    history_message_limit: int | None = None

    def __init__(self) -> None:
        self.subject: dict[str, Any] = {}
        self.step_index = 0
        self.last_context: TaskContext | None = None
        self._prompt_cache: str | None = None

    def reset(self, subject: dict[str, Any]) -> None:
        """Reset all per-subject state. Override and call super()."""
        self.subject = deepcopy(subject)
        self.step_index = 0
        self.last_context = None

    def before_step(self, subject: dict[str, Any], task_response: dict[str, Any]) -> None:
        """Called once before perception/model/action for every step."""
        del subject, task_response
        self.step_index += 1

    def observe(self, context: TaskContext) -> None:
        """Receive the newest perception before the model is invoked."""
        self.last_context = context

    def enrich_prompt(self, variables: dict[str, Any], context: TaskContext) -> dict[str, Any]:
        """Add task-owned instructions and state to shared prompt variables."""
        variables = dict(variables)
        state = self.state_for_prompt()
        prompt = self.build_task_prompt(context)
        if state:
            prompt = f"{prompt}\n当前任务策略状态：{json.dumps(state, ensure_ascii=False)}".strip()
        variables["task_prompt"] = prompt
        variables["task_strategy_state"] = state
        return variables

    def validate_action(self, action: dict[str, Any], context: TaskContext | None) -> dict[str, Any]:
        """Validate or rewrite one parsed action before TongSim executes it."""
        del context
        return action

    def next_local_action(self, context: TaskContext) -> dict[str, Any] | None:
        """返回可由本地策略确定的动作；返回 None 时才请求视觉模型。"""
        del context
        return None

    def tracked_raw_ids(self) -> tuple[str, ...]:
        """Raw object IDs whose global AABBs are needed for the next observation."""
        return ()

    def refresh_raw_ids(self) -> tuple[str, ...]:
        """本轮必须重新查询 AABB 的对象；默认保持旧行为。"""
        return self.tracked_raw_ids()

    def needs_scene_perception(self) -> bool:
        """是否仍需刷新完整的当前视野；默认任务保持每步感知。"""
        return True

    def note_forced_vlm(self, reason: str) -> None:
        """Record a VLM escalation initiated by the shared perception runtime."""
        del reason

    def after_action(
        self,
        action: dict[str, Any],
        result: Any,
        context: TaskContext | None,
    ) -> None:
        """Update task state after TongSim executed an action."""
        del action, result, context

    def state_for_prompt(self) -> dict[str, Any]:
        return {"step_index": self.step_index}

    def build_task_prompt(self, context: TaskContext) -> str:
        del context
        return self.load_prompt()

    def load_prompt(self) -> str:
        if self._prompt_cache is not None:
            return self._prompt_cache

        module_path = Path(import_module(type(self).__module__).__file__ or "")
        file_path = module_path.with_name("prompt.txt")
        self._prompt_cache = file_path.read_text(encoding="utf-8").strip() if file_path.is_file() else ""
        return self._prompt_cache


def action_succeeded(result: Any) -> bool:
    """Best-effort normalization for the several result shapes returned by TongSim."""
    if result is True:
        return True
    if not isinstance(result, dict):
        return False
    if result.get("success") is True:
        return True
    status = str(result.get("result") or result.get("status") or "").strip().lower()
    return status in {"success", "succeeded", "ok", "true", "1"}


class GenericTaskStrategy(TaskStrategy):
    """Safe fallback so an unknown task type does not crash the agent."""

    def __init__(self, task_type: str = "unknown") -> None:
        super().__init__()
        self.task_type = task_type
