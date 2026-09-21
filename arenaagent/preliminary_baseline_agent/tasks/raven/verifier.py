from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ModelVote:
    question: int
    answer: int
    confidence: float
    alternatives: dict[int, float] = field(default_factory=dict)
    candidate_confidences: dict[int, float] = field(default_factory=dict)
    rule: str = ""
    critique: str = ""
    predicted_attributes: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    candidate_mismatches: dict[int, list[str]] = field(default_factory=dict)

    def score_vector(self) -> list[float]:
        if self.candidate_confidences:
            return [max(float(self.candidate_confidences.get(index, 0.0)), 0.0) for index in range(1, 9)]
        scores = [0.0] * 8
        scores[self.answer - 1] = max(self.confidence, 0.05)
        for answer, confidence in self.alternatives.items():
            if 1 <= answer <= 8:
                scores[answer - 1] = max(scores[answer - 1], confidence)
        return scores


def _json_payloads(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    payloads: list[Any] = []
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        payloads.append(payload)
    return payloads


def _clamp_confidence(value: Any, default: float = 0.5) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if numeric > 1.0 and numeric <= 100.0:
        numeric /= 100.0
    return max(0.0, min(numeric, 1.0))


def _answer_number(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        answer = int(value)
    except (TypeError, ValueError):
        return None
    return answer if 1 <= answer <= 8 else None


def _normalize_candidate_confidences(values: dict[int, float]) -> dict[int, float]:
    """Convert model scores into a valid eight-way probability distribution."""
    cleaned = {
        candidate: max(0.0, float(values.get(candidate, 0.0)))
        for candidate in range(1, 9)
    }
    total = sum(cleaned.values())
    if total <= 1e-12:
        return {}
    return {candidate: score / total for candidate, score in cleaned.items()}


def parse_model_votes(text: str, expected_questions: int = 3) -> list[ModelVote]:
    """Strictly accept one-based candidate IDs, preventing 0/1 indexing drift."""
    for payload in _json_payloads(text or ""):
        entries: Any = payload.get("questions") if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            continue
        votes: dict[int, ModelVote] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                question = int(entry.get("question"))
            except (TypeError, ValueError):
                continue
            answer = _answer_number(entry.get("answer"))
            candidate_confidences: dict[int, float] = {}
            candidate_mismatches: dict[int, list[str]] = {}
            for candidate_entry in entry.get("candidates") or []:
                if not isinstance(candidate_entry, dict):
                    continue
                candidate = _answer_number(candidate_entry.get("id", candidate_entry.get("answer")))
                if candidate is None:
                    continue
                candidate_confidences[candidate] = _clamp_confidence(
                    candidate_entry.get("confidence", candidate_entry.get("probability")),
                    default=0.0,
                )
                mismatch = candidate_entry.get("mismatch", candidate_entry.get("mismatches", []))
                if isinstance(mismatch, str):
                    mismatch = [mismatch]
                if isinstance(mismatch, list):
                    candidate_mismatches[candidate] = [str(value)[:120] for value in mismatch[:8]]
            candidate_confidences = _normalize_candidate_confidences(candidate_confidences)
            # The declared selection must agree with the model's own eight-way
            # distribution.  This prevents a JSON answer/probability mismatch.
            if candidate_confidences and max(candidate_confidences.values(), default=0.0) > 0.0:
                answer = max(candidate_confidences, key=candidate_confidences.get)
            if answer is None or not 1 <= question <= expected_questions:
                continue
            alternatives: dict[int, float] = {}
            for alternative in entry.get("alternatives") or []:
                if not isinstance(alternative, dict):
                    continue
                alternative_answer = _answer_number(alternative.get("answer"))
                if alternative_answer is not None:
                    alternatives[alternative_answer] = _clamp_confidence(
                        alternative.get("confidence"), default=0.2
                    )
            votes[question] = ModelVote(
                question=question,
                answer=answer,
                confidence=(
                    candidate_confidences[answer]
                    if answer in candidate_confidences
                    else _clamp_confidence(entry.get("confidence"))
                ),
                alternatives=alternatives,
                candidate_confidences=candidate_confidences,
                rule=str(entry.get("rule") or entry.get("reason") or "")[:500],
                critique=str(entry.get("critique") or entry.get("revision") or "")[:500],
                predicted_attributes=(
                    dict(entry.get("predicted_attributes"))
                    if isinstance(entry.get("predicted_attributes"), dict)
                    else {}
                ),
                evidence=[str(value)[:200] for value in (entry.get("evidence") or [])[:12]]
                if isinstance(entry.get("evidence"), list)
                else [],
                candidate_mismatches=candidate_mismatches,
            )
        if len(votes) == expected_questions:
            return [votes[index] for index in range(1, expected_questions + 1)]
    return []


def _direct_answer_sequence(value: Any, expected_questions: int) -> list[int]:
    if isinstance(value, (list, tuple)) and len(value) == expected_questions:
        answers = [_answer_number(item) for item in value]
        return [int(answer) for answer in answers] if all(answer is not None for answer in answers) else []
    if not isinstance(value, str):
        return []
    stripped = value.strip()
    patterns = (
        rf"[1-8](?:\s*[,，、/|]\s*[1-8]){{{expected_questions - 1}}}",
        rf"[1-8](?:\s+[1-8]){{{expected_questions - 1}}}",
        rf"[1-8]{{{expected_questions}}}",
    )
    if not any(re.fullmatch(pattern, stripped) for pattern in patterns):
        return []
    answers = [int(value) for value in re.findall(r"[1-8]", stripped)]
    return answers if len(answers) == expected_questions else []


def parse_direct_model_votes(
    text: str,
    expected_questions: int = 3,
    confidence: float = 0.75,
) -> list[ModelVote]:
    """Parse K3's concise whole-canvas answer without accepting numbers from prose."""
    structured = parse_model_votes(text, expected_questions=expected_questions)
    if structured:
        return structured

    answers: list[int] = []
    for payload in _json_payloads(text or ""):
        if isinstance(payload, dict):
            for key in ("answers", "answer", "final_answer"):
                answers = _direct_answer_sequence(payload.get(key), expected_questions)
                if answers:
                    break
        else:
            answers = _direct_answer_sequence(payload, expected_questions)
        if answers:
            break

    if not answers:
        sequence = (
            rf"(?:[1-8](?:\s*[,，、/|]\s*[1-8]){{{expected_questions - 1}}}"
            rf"|[1-8](?:\s+[1-8]){{{expected_questions - 1}}}"
            rf"|[1-8]{{{expected_questions}}})"
        )
        labelled = re.findall(
            rf"(?:最终答案|答案)\s*[:：]\s*({sequence})",
            text or "",
            flags=re.IGNORECASE,
        )
        if labelled:
            answers = _direct_answer_sequence(labelled[-1], expected_questions)

    if not answers:
        plain = (text or "").strip().strip("`*_")
        answers = _direct_answer_sequence(plain, expected_questions)
    if not answers:
        return []

    reported_confidences: list[float] = []
    for payload in _json_payloads(text or ""):
        if not isinstance(payload, dict):
            continue
        raw_confidences = payload.get("confidences", payload.get("confidence"))
        if isinstance(raw_confidences, (list, tuple)) and len(raw_confidences) == expected_questions:
            reported_confidences = [
                _clamp_confidence(value, default=confidence) for value in raw_confidences
            ]
            break
    if not reported_confidences:
        confidence_match = re.search(
            r"(?:置信度|confidences?)\s*[:：=]\s*([0-9.,，、％%\s]+)",
            text or "",
            flags=re.IGNORECASE,
        )
        if confidence_match:
            raw_values = re.findall(r"[0-9]+(?:\.[0-9]+)?", confidence_match.group(1))
            if len(raw_values) >= expected_questions:
                reported_confidences = [
                    _clamp_confidence(value, default=confidence)
                    for value in raw_values[:expected_questions]
                ]
    if not reported_confidences:
        reported_confidences = [_clamp_confidence(confidence, default=0.75)] * expected_questions

    return [
        ModelVote(
            question=question,
            answer=answer,
            confidence=reported_confidences[question - 1],
            candidate_confidences={
                candidate: (
                    reported_confidences[question - 1]
                    if candidate == answer
                    else (1.0 - reported_confidences[question - 1]) / 7.0
                )
                for candidate in range(1, 9)
            },
            rule="K3 whole-image direct answer",
            evidence=["Original canvas containing all three questions"],
        )
        for question, answer in enumerate(answers, start=1)
    ]


def parse_single_model_vote(text: str, question: int) -> ModelVote | None:
    """Parse one independently requested question while preserving its global ID."""
    if not 1 <= question <= 3:
        return None
    for payload in _json_payloads(text or ""):
        if isinstance(payload, dict) and isinstance(payload.get("questions"), list):
            entries = payload["questions"]
        elif isinstance(payload, list):
            entries = payload
        elif isinstance(payload, dict):
            entries = [payload]
        else:
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            normalized = dict(entry)
            normalized["question"] = 1
            votes = parse_model_votes(
                json.dumps({"questions": [normalized]}, ensure_ascii=False),
                expected_questions=1,
            )
            if votes:
                votes[0].question = question
                return votes[0]

    # Some OpenAI-compatible vision endpoints occasionally cut off the final
    # brace of an otherwise useful JSON response. Recover only explicit public
    # fields; never invent an answer from prose.
    candidate_confidences: dict[int, float] = {}
    candidate_pattern = re.compile(
        r'"(?:id|candidate)"\s*:\s*"?([1-8])"?[^{}]{0,240}?'
        r'"(?:confidence|probability)"\s*:\s*"?([0-9]+(?:\.[0-9]+)?)',
        re.IGNORECASE,
    )
    for match in candidate_pattern.finditer(text or ""):
        candidate = int(match.group(1))
        candidate_confidences[candidate] = _clamp_confidence(match.group(2), default=0.0)
    candidate_confidences = _normalize_candidate_confidences(candidate_confidences)

    answer_match = re.search(r'"answer"\s*:\s*"?([1-8])"?', text or "", re.IGNORECASE)
    answer = _answer_number(answer_match.group(1)) if answer_match else None
    if candidate_confidences:
        answer = max(candidate_confidences, key=candidate_confidences.get)
    if answer is None:
        return None

    confidence_match = re.search(
        r'"confidence"\s*:\s*"?([0-9]+(?:\.[0-9]+)?)', text or "", re.IGNORECASE
    )
    confidence = (
        candidate_confidences.get(answer, 0.0)
        or _clamp_confidence(confidence_match.group(1), default=0.5)
        if confidence_match
        else candidate_confidences.get(answer, 0.5)
    )
    return ModelVote(
        question=question,
        answer=answer,
        confidence=confidence,
        candidate_confidences=candidate_confidences,
        critique="Recovered from an incomplete JSON response.",
    )


def sanitize_ranked_triples(values: Any) -> list[list[int]]:
    if not isinstance(values, list):
        return []
    verified: list[list[int]] = []
    seen: set[tuple[int, int, int]] = set()
    for value in values:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            continue
        triple = tuple(_answer_number(item) or 0 for item in value)
        if 0 in triple or triple in seen:
            continue
        seen.add(triple)
        verified.append(list(triple))
    return verified


def verify_selected_answers(answers: list[int]) -> list[int]:
    if len(answers) != 3 or any(_answer_number(answer) is None for answer in answers):
        raise ValueError(f"Raven answer must be three one-based candidate IDs: {answers!r}")
    return [int(answer) for answer in answers]
