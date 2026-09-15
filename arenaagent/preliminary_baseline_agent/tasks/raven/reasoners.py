from __future__ import annotations

import base64
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from loguru import logger
from PIL import Image, ImageDraw, ImageOps

from arenaagent.preliminary_baseline_agent.tasks.raven.verifier import (
    ModelVote,
    parse_single_model_vote,
)


def _data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def create_labeled_question_sheet(images: list[Image.Image], question_number: int) -> Image.Image:
    if len(images) != 16:
        raise ValueError(f"Expected 16 panels, got {len(images)}")
    sheet = Image.new("RGB", (780, 975), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((18, 12), f"QUESTION {question_number}: context matrix (missing bottom-right)", fill="black")
    context_size, context_gap = 165, 12
    for index in range(9):
        row, column = divmod(index, 3)
        x = 18 + column * (context_size + context_gap)
        y = 42 + row * (context_size + context_gap)
        draw.rectangle((x, y, x + context_size, y + context_size), outline="black", width=2)
        if index == 8:
            draw.text((x + 72, y + 70), "?", fill="black")
            continue
        panel = ImageOps.contain(images[index].convert("RGB"), (context_size - 8, context_size - 8))
        sheet.paste(panel, (x + (context_size - panel.width) // 2, y + (context_size - panel.height) // 2))

    draw.text((18, 590), "Candidates — IDs are exactly 1..8 in reading order:", fill="black")
    candidate_width, candidate_height = 175, 165
    for candidate in range(8):
        row, column = divmod(candidate, 4)
        x = 18 + column * (candidate_width + 12)
        y = 618 + row * (candidate_height + 10)
        draw.rectangle((x, y, x + candidate_width, y + candidate_height), outline="black", width=2)
        draw.text((x + 4, y + 3), str(candidate + 1), fill="black")
        panel = ImageOps.contain(images[8 + candidate].convert("RGB"), (candidate_width - 28, candidate_height - 12))
        sheet.paste(panel, (x + 24 + (candidate_width - 28 - panel.width) // 2, y + 8))
    return sheet


def ask_visual_reasoner(
    client: Any,
    image_groups: list[list[Image.Image]],
    deterministic_summaries: list[dict[str, Any]],
    correction_context: dict[str, Any] | None = None,
    target_questions: list[int] | None = None,
) -> tuple[list[ModelVote], str]:
    """Solve independent Raven questions concurrently with grounded evidence."""
    if client is None:
        return [], ""
    targets = sorted(set(target_questions or [1, 2, 3]))
    targets = [question for question in targets if 1 <= question <= len(image_groups)]

    def correction_for(question: int) -> str:
        if not correction_context:
            return ""
        compact_attempts: list[dict[str, Any]] = []
        for attempt in correction_context.get("previous_attempts", [])[-4:]:
            if not isinstance(attempt, dict):
                continue
            answers = attempt.get("answer") or []
            prior_vote = next(
                (
                    vote
                    for vote in attempt.get("visual_votes", [])
                    if isinstance(vote, dict) and vote.get("question") == question
                ),
                None,
            )
            if prior_vote is None:
                prior_vote = next(
                    (
                        vote
                        for vote in attempt.get("text_votes", [])
                        if isinstance(vote, dict) and vote.get("question") == question
                    ),
                    None,
                )
            compact_attempts.append(
                {
                    "attempt": attempt.get("attempt"),
                    "status": attempt.get("status"),
                    "answer_for_this_question": answers[question - 1] if len(answers) >= question else None,
                    "prior_model_vote": prior_vote,
                }
            )
        return (
            "\nCORRECTION ROUND: the server rejected these complete triples: "
            f"{json.dumps(correction_context.get('rejected_triples', []), ensure_ascii=False)}. "
            "The server does NOT reveal which digit was wrong, so do not assume this question is definitely wrong. "
            "Re-evaluate it independently, explicitly compare the previous choice against its strongest alternatives, "
            "and change it only when the visual and structured evidence justify the change. If keeping the previous "
            "choice would recreate a rejected triple while the other two digits are retained, return the strongest "
            "evidence-backed alternative rather than the rejected choice. Previous evidence for this "
            f"question: {json.dumps(compact_attempts, ensure_ascii=False)}"
        )

    def invoke_one(question: int) -> tuple[int, ModelVote | None, str]:
        summary = deterministic_summaries[question - 1]
        schema = (
            '{"question":%d,"answer":2,"confidence":0.40,'
            '"candidates":[{"id":1,"confidence":0.05,"mismatch":"count"},..., '
            '{"id":8,"confidence":0.05,"mismatch":"none"}],'
            '"rule":"concise auditable rule","evidence":["observation"],'
            '"predicted_attributes":{"count":"...","shape":"...","fill":"...",'
            '"position":"...","size":"...","rotation":"..."},'
            '"critique":"what changed from prior evidence, if anything"}'
        ) % question
        prompt = (
            f"QUESTION_NUMBER={question}. Solve only this Raven matrix. Infer rules across both complete rows and "
            "columns. Check count, position, shape, fill, size, rotation, symmetry, set operations, composition and "
            "cyclic 3x3 permutations as independent attributes. Preserve the row's shape family before comparing "
            "numeric contour features, and explicitly test any monotonic size constraint. Candidate IDs are exactly "
            "1..8 as printed. The "
            "following OpenCV, scene-graph and rule-engine measurements are fallible supporting evidence, not ground "
            "truth; resolve conflicts by checking the image. Any verified_experience section contains transferable "
            "techniques distilled from earlier server-confirmed successes, never an answer key; use a technique only "
            "when the current panels independently satisfy it.\nSTRUCTURED_EVIDENCE="
            f"{json.dumps(summary, ensure_ascii=False)}"
            + correction_for(question)
            + "\nReturn JSON only. Include all eight candidate IDs exactly once. Candidate confidences must be "
            "calibrated probabilities in [0,1] that sum approximately to 1. The answer must be the candidate with the "
            "highest confidence. Put answer, confidence and candidates before explanations. Keep the complete JSON "
            "under 1800 characters, with at most three short evidence items. Provide concise auditable evidence, not "
            f"hidden chain-of-thought. Schema: {schema}"
        )
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": _data_url(create_labeled_question_sheet(image_groups[question - 1], question))},
            },
        ]
        response = client.invoke([{"role": "user", "content": content}], max_retries=2)
        response_text = str(getattr(response, "text", "") or "")
        return question, parse_single_model_vote(response_text, question), response_text

    votes: dict[int, ModelVote] = {}
    raw: dict[int, str] = {}
    # The three matrices are independent.  Separate requests reduce visual
    # interference and wall time while preserving a global question ID.
    with ThreadPoolExecutor(max_workers=max(1, len(targets)), thread_name_prefix="raven-vlm") as executor:
        future_map = {executor.submit(invoke_one, question): question for question in targets}
        for future in as_completed(future_map):
            question = future_map[future]
            try:
                _, vote, response_text = future.result()
            except Exception as exc:
                raw[question] = f"ERROR: {exc}"
                continue
            raw[question] = response_text
            if vote is not None:
                votes[question] = vote
            else:
                logger.warning(
                    "Raven question {} returned an unparseable response (chars={}, preview={!r})",
                    question,
                    len(response_text),
                    response_text[:180],
                )
    raw_text = json.dumps({str(key): raw[key] for key in sorted(raw)}, ensure_ascii=False)
    return [votes[key] for key in sorted(votes)], raw_text


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
    with ThreadPoolExecutor(max_workers=max(1, len(targets)), thread_name_prefix="raven-text") as executor:
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
