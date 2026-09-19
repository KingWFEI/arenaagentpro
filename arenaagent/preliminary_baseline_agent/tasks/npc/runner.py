from __future__ import annotations

from typing import Any

from loguru import logger

from arenaagent.preliminary_baseline_agent.task_runtime import TaskContext
from arenaagent.preliminary_baseline_agent.tasks.npc.strategy import NpcStrategy
from arenaagent.preliminary_baseline_agent.tasks.npc.text_client import (
    build_npc_text_client_from_env,
)
from arenaagent.vlm_agent.json_parsor import extract_last_json_from_text


def run_npc_fast_step(
    agent: Any,
    subject: dict[str, Any],
    task_response: dict[str, Any],
    strategy: NpcStrategy,
) -> dict[str, Any]:
    """Run one NPC step without camera or held-object RPCs."""
    asset_names = subject.get("npc_asset_name")
    agent._npc_name_to_asset_name = dict(asset_names) if isinstance(asset_names, dict) else {}

    context = TaskContext(
        task_type="npc",
        subject=dict(subject),
        task_response=dict(task_response or {}),
        visible_objects=[],
        object_in_hand=None,
        movable_objects=[],
        action_histories=list(agent._action_histories),
        last_action_result=agent._last_action_res,
    )
    agent._task_context = context
    strategy.observe(context)

    interview_action = strategy.next_local_action(context)
    if interview_action is not None:
        logger.info("NPC fast path executes interview action: {}", interview_action)
        return _execute_and_update(agent, strategy, context, interview_action)

    messages = strategy.decision_messages()
    agent._save_prompt_messages(messages)
    text_client = getattr(agent, "npc_text_client", None)
    if text_client is None and not getattr(agent, "_npc_text_client_initialized", False):
        text_client = build_npc_text_client_from_env()
        agent.npc_text_client = text_client
        agent._npc_text_client_initialized = True
    if text_client is None:
        text_client = agent.vlm_client
    response = text_client.invoke(messages) if text_client else None
    response_text = getattr(response, "text", None) or ""
    parsed = extract_last_json_from_text(response_text)
    agent.last_json_parse_message = parsed
    logger.info("NPC text decision response {}", response_text)

    final_action = agent._parse_action_from_response(parsed)
    return _execute_and_update(agent, strategy, context, final_action)


def _execute_and_update(
    agent: Any,
    strategy: NpcStrategy,
    context: TaskContext,
    action: dict[str, Any],
) -> dict[str, Any]:
    result = agent._execute_action_and_record(action)
    strategy.after_action(action, result, context)
    return result
