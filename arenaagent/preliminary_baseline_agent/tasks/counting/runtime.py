from __future__ import annotations

import json
import math
import time
from typing import Any

from loguru import logger


def counting_subject_signature(subject: dict[str, Any]) -> str:
    """Stable identity used to reject Arena's previous-subject transition cache."""
    options = subject.get("options")
    normalized_options = options if isinstance(options, dict) else {}
    payload = {
        "task_type": subject.get("task_type"),
        "counting_type": subject.get("counting_type"),
        "question": subject.get("question"),
        "options": normalized_options,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


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


def _expected_option(subject: dict[str, Any], payload: dict[str, Any], answer_key: str) -> str | None:
    """Return the local option key for a numeric submission, if unambiguous."""
    submitted = _submitted_values(payload, answer_key)
    if len(submitted) != 1:
        return None
    target = next(iter(submitted))
    options = subject.get("options")
    if not isinstance(options, dict):
        return None
    matches: list[str] = []
    for key, value in options.items():
        if isinstance(value, bool):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric) and numeric.is_integer() and int(numeric) == target:
            matches.append(str(key))
    return matches[0] if len(matches) == 1 else None


def _ensure_option_mapping_is_current(
    subject: dict[str, Any],
    payload: dict[str, Any],
    answer_key: str,
    response: Any,
) -> None:
    """Detect the Arena transition window where subject data is one round stale."""
    if not isinstance(response, dict):
        return
    returned = response.get("selected_option")
    expected = _expected_option(subject, payload, answer_key)
    if expected is not None and returned is not None and str(returned) != expected:
        raise RuntimeError(
            "Counting subject/options changed during submission: "
            f"local option={expected}, server option={returned}. "
            "Abort this agent before it penalizes the new subject."
        )


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


def _fresh_counting_subject(
    agent: Any,
    subject: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Refresh a prefetched subject in-place without creating a second Agent.

    The 912 server can admit the next Agent while ``get_subject`` still exposes
    the completed round cached during ``init``.  Disconnecting at that point
    leaves the registered Agent in the controller and makes the replacement
    Agent produce ``not in EVALUATING``.  Keep this connection alive and poll
    until the public subject signature changes instead.
    """
    signature = counting_subject_signature(subject)
    previous = getattr(agent, "_previous_completed_counting_signature", None)
    if not isinstance(previous, str) or signature != previous:
        return subject, signature

    cfg = getattr(agent, "cfg", None)
    try:
        timeout = max(float(getattr(cfg, "counting_subject_refresh_timeout_seconds", 20.0)), 0.0)
    except (TypeError, ValueError):
        timeout = 20.0
    try:
        interval = max(float(getattr(cfg, "counting_subject_refresh_interval_seconds", 0.5)), 0.05)
    except (TypeError, ValueError):
        interval = 0.5

    deadline = time.monotonic() + timeout
    attempts = 0
    while True:
        attempts += 1
        refreshed = agent._get_subject_from_task()
        if isinstance(refreshed, dict) and refreshed:
            refreshed_signature = counting_subject_signature(refreshed)
            if refreshed_signature != previous:
                logger.info(
                    "Arena counting subject refreshed in the existing connection after {} polls",
                    attempts,
                )
                return refreshed, refreshed_signature
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Arena kept returning the previously completed counting subject for "
                f"{timeout:.1f}s on the existing connection"
            )
        time.sleep(interval)


def run_counting_subject(agent: Any, subject: dict[str, Any]) -> None:
    """ZRQ's one-shot counting flow, scoped to the current Agent's counting branch."""
    subject, subject_signature = _fresh_counting_subject(agent, subject)
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
    answer_key = str(agent.action_space.get("key") or "answer")
    _ensure_option_mapping_is_current(subject, result, answer_key, apply_response)
    if isinstance(apply_response, dict) and (
        apply_response.get("success") is False
        or str(apply_response.get("result") or "").lower() in {"failed", "error"}
        or apply_response.get("error")
    ):
        raise RuntimeError(f"Counting answer was rejected by task service: {apply_response}")

    if isinstance(apply_response, dict) and apply_response.get("answer_right") is False:
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
            _ensure_option_mapping_is_current(subject, retry_payload, answer_key, apply_response)
            if not isinstance(apply_response, dict):
                break
            if apply_response.get("answer_right") is True:
                break
            if apply_response.get("answer_right") is not False:
                break
        if isinstance(apply_response, dict) and apply_response.get("answer_right") is False:
            # A submitted Agent cannot safely disconnect: the 912 task
            # controller keeps waiting for it forever.  Exhausting every public
            # option should be impossible because the correct value is public,
            # but request evaluation as a final no-timeout guard if the service
            # still rejects all of them.
            logger.error(
                "Counting service rejected every bounded public recovery; "
                "requesting evaluation to avoid leaving the controller in ANSWERING"
            )
    agent.subject_finished = True
    evaluation = agent._evaluate_subject()
    agent._completed_counting_subject_signature = subject_signature
    logger.info("Evaluation accepted (returned score is cumulative before this subject): {}", evaluation)
