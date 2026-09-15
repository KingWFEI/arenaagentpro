from __future__ import annotations

import math
from typing import Any

from loguru import logger


def _submitted_values(payload: dict[str, Any], answer_key: str) -> set[int]:
    value = payload.get(answer_key)
    if isinstance(value, bool):
        return set()
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return set()
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < 0:
        return set()
    return {int(numeric)}


def _emergency_result(agent: Any, subject: dict[str, Any], strategy: Any) -> dict[str, Any]:
    """ZRQ's public-option fallback when local counting cannot submit."""
    ranked: list[int] = []
    ranker = getattr(strategy, "ranked_recovery_counts", None)
    if callable(ranker):
        try:
            ranked = list(ranker(subject, set()))
        except Exception as exc:
            logger.warning("Could not rank emergency counting options: {}", exc)
    if not ranked:
        options = subject.get("options")
        if isinstance(options, dict):
            for value in options.values():
                if isinstance(value, bool):
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(numeric) and numeric.is_integer() and numeric >= 0:
                    count = int(numeric)
                    if count not in ranked:
                        ranked.append(count)
    if not ranked:
        raise RuntimeError("Counting subject has no valid numeric option to submit")
    answer_key = str(agent.action_space.get("key") or "answer")
    logger.warning("Counting emergency submission selected count={}", ranked[0])
    return {answer_key: ranked[0]}


def run_counting_subject(agent: Any, subject: dict[str, Any]) -> None:
    """ZRQ's one-shot counting flow, scoped to the current Agent's counting branch."""
    task_response: dict[str, Any] = {}
    strategy = agent._ensure_task_strategy(subject)
    if not hasattr(strategy, "run_fast"):
        raise RuntimeError("Counting strategy has no run_fast implementation")
    agent.subject_finished = False
    active_index = agent._raven_current_subject_index()
    try:
        result = strategy.run_fast(agent, dict(subject), task_response)
    except Exception as exc:
        logger.opt(exception=True).warning(
            "Counting decision failed before submission; using an evidence-ranked public option "
            "so the agent cannot leave the server in ANSWERING: {}",
            exc,
        )
        result = _emergency_result(agent, subject, strategy)
    if agent._raven_current_subject_index() != active_index:
        raise RuntimeError("Question changed during counting; refusing to submit a stale answer")
    apply_response = agent._apply_action(result)
    logger.info("Counting answer submitted payload={} response={}", result, apply_response)
    if isinstance(apply_response, dict) and (
        apply_response.get("success") is False
        or str(apply_response.get("result") or "").lower() in {"failed", "error"}
        or apply_response.get("error")
    ):
        raise RuntimeError(f"Counting answer was rejected by task service: {apply_response}")

    if isinstance(apply_response, dict) and apply_response.get("answer_right") is False:
        answer_key = str(agent.action_space.get("key") or "answer")
        attempted = _submitted_values(result, answer_key)
        retry_values = strategy.ranked_recovery_counts(subject, attempted)
        retry_limit = max(
            0,
            min(int(getattr(agent.cfg, "counting_max_recovery_submissions", 7)), len(retry_values)),
        )
        logger.warning(
            "Counting answer explicitly rejected; retrying same subject with "
            "evidence-ranked remaining counts={} (limit={})",
            retry_values,
            retry_limit,
        )
        for retry_value in retry_values[:retry_limit]:
            if agent._raven_current_subject_index() != active_index:
                raise RuntimeError("Question changed during counting recovery")
            retry_payload = {answer_key: retry_value}
            apply_response = agent._apply_action(retry_payload)
            attempted.add(retry_value)
            logger.info("Counting recovery submitted payload={} response={}", retry_payload, apply_response)
            if not isinstance(apply_response, dict):
                break
            if apply_response.get("answer_right") is True:
                break
            if apply_response.get("answer_right") is not False:
                break
        if isinstance(apply_response, dict) and apply_response.get("answer_right") is False:
            raise RuntimeError(
                "Counting service rejected every evidence-ranked public option; "
                "refusing to wait for the 400-second timeout"
            )
    agent.subject_finished = True
    evaluation = agent._evaluate_subject()
    logger.info("Evaluation accepted (returned score is cumulative before this subject): {}", evaluation)
