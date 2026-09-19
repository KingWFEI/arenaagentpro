from __future__ import annotations

import base64
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image

from arenaagent.preliminary_baseline_agent.tasks.raven.verifier import (
    ModelVote,
    parse_direct_model_votes,
    parse_single_model_vote,
)


def _request_concurrency(env_name: str) -> int:
    """Respect provider account limits; the 912 Moonshot account allows one."""
    try:
        return max(1, int(os.getenv(env_name, "1")))
    except ValueError:
        return 1


def _data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


@lru_cache(maxsize=1)
def _whole_image_prompt() -> str:
    return Path(__file__).with_name("vision_prompt.txt").read_text(encoding="utf-8").strip()


def ask_visual_reasoner(
    client: Any,
    image_groups: list[list[Image.Image]],
    deterministic_summaries: list[dict[str, Any]],
    correction_context: dict[str, Any] | None = None,
    target_questions: list[int] | None = None,
    whole_image: Image.Image | None = None,
) -> tuple[list[ModelVote], str]:
    """Ask K3 once about the untouched canvas containing all three questions."""
    if client is None:
        return [], ""
    del image_groups, deterministic_summaries, target_questions
    if whole_image is None:
        logger.warning("Raven original canvas is unavailable; split-image VLM fallback is disabled")
        return [], ""

    prompt = _whole_image_prompt()
    if correction_context:
        rejected = correction_context.get("rejected_triples", [])
        prior_answers = [
            attempt.get("answer")
            for attempt in correction_context.get("previous_attempts", [])[-4:]
            if isinstance(attempt, dict) and isinstance(attempt.get("answer"), list)
        ]
        prompt += (
            "\n\n复核提示：服务端只确认以下完整三位答案不正确，并未指出具体错题："
            f"{json.dumps(rejected, ensure_ascii=False)}。"
            f"历史作答：{json.dumps(prior_answers, ensure_ascii=False)}。"
            "请仍然对原始整图中的三道题全部重新独立求解，并只输出新的三位答案。"
        )

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": _data_url(whole_image)}},
    ]
    response = client.invoke([{"role": "user", "content": content}], max_retries=1)
    response_text = str(getattr(response, "text", "") or "")
    votes = parse_direct_model_votes(response_text, expected_questions=3)
    if not votes:
        logger.warning(
            "Whole-image Raven response was unparseable (chars={}, preview={!r})",
            len(response_text),
            response_text[:180],
        )
    return votes, response_text


def ask_text_verifier(
    client: Any,
    deterministic_summaries: list[dict[str, Any]],
    visual_votes: list[ModelVote],
    correction_context: dict[str, Any] | None = None,
    image_groups: list[list[Image.Image]] | None = None,
    target_questions: list[int] | None = None,
) -> tuple[list[ModelVote], str]:
    if client is None:
        return [], ""
    del image_groups  # The independent DeepSeek verifier is intentionally text-only.
    targets = sorted(set(target_questions or [1, 2, 3]))
    targets = [question for question in targets if 1 <= question <= len(deterministic_summaries)]
    visual_by_question = {vote.question: vote for vote in visual_votes}

    def invoke_one(question: int) -> tuple[int, ModelVote | None, str]:
        visual_vote = visual_by_question.get(question)
        visual = (
            {
                "answer": visual_vote.answer,
                "confidence": visual_vote.confidence,
                "candidate_confidences": visual_vote.candidate_confidences,
                "rule": visual_vote.rule,
                "evidence": visual_vote.evidence,
                "predicted_attributes": visual_vote.predicted_attributes,
            }
            if visual_vote is not None
            else None
        )
        rejection = {
            "rejected_triples": (correction_context or {}).get("rejected_triples", []),
            "previous_answers_for_question": [
                attempt.get("answer", [None, None, None])[question - 1]
                for attempt in (correction_context or {}).get("previous_attempts", [])[-4:]
                if isinstance(attempt, dict)
                and isinstance(attempt.get("answer"), list)
                and len(attempt["answer"]) >= question
            ],
        }
        prompt = (
            f"You are an independent text-only verifier for Raven question {question}. Reconcile the visual proposal "
            "with OpenCV measurements, scene-graph relations and induced rules. Check that the selected candidate has "
            "every predicted attribute and that the rule explains complete rows and columns. Treat semantic shape "
            "family and explicit monotonic-size constraints as gates: polygon vertex arithmetic is invalid for "
            "circle-like contours. The server only rejects "
            "whole triples, so do not claim to know which digit was wrong. Return one compact JSON object with fields "
            "question, answer, confidence, candidates (all IDs 1..8 with probabilities summing to 1), rule, evidence, "
            "predicted_attributes and critique. Put answer and candidates first. No images are available.\n"
            f"VISUAL_PROPOSAL={json.dumps(visual, ensure_ascii=False)}\n"
            f"STRUCTURED_EVIDENCE={json.dumps(deterministic_summaries[question - 1], ensure_ascii=False)}\n"
            f"REJECTION_CONTEXT={json.dumps(rejection, ensure_ascii=False)}"
        )
        # One bounded attempt: a verifier timeout must not consume the entire
        # subject budget or outlive the task server.
        response = client.invoke([{"role": "user", "content": prompt}], max_retries=1)
        response_text = str(getattr(response, "text", "") or "")
        return question, parse_single_model_vote(response_text, question), response_text

    votes: dict[int, ModelVote] = {}
    raw: dict[int, str] = {}
    max_workers = min(max(1, len(targets)), _request_concurrency("RAVEN_TEXT_MAX_CONCURRENCY"))
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="raven-text") as executor:
        future_map = {executor.submit(invoke_one, question): question for question in targets}
        for future in as_completed(future_map):
            question = future_map[future]
            try:
                _, vote, response_text = future.result()
            except Exception as exc:
                raw[question] = f"ERROR: {exc}"
                logger.warning("Raven text verifier failed for question {}: {}", question, exc)
                continue
            raw[question] = response_text
            if vote is not None:
                votes[question] = vote
            else:
                logger.warning("Raven text verifier returned invalid JSON for question {}", question)
    raw_text = json.dumps({str(key): raw[key] for key in sorted(raw)}, ensure_ascii=False)
    return [votes[key] for key in sorted(votes)], raw_text
