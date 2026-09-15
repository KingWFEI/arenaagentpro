from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from math import log
from typing import Iterable

import numpy as np

from arenaagent.preliminary_baseline_agent.tasks.raven.verifier import ModelVote, sanitize_ranked_triples


@dataclass(slots=True)
class EvidenceSource:
    name: str
    scores: list[float]
    weight: float


@dataclass(slots=True)
class QuestionEnsemble:
    probabilities: list[float]
    answer: int
    confidence: float
    margin: float
    sources: list[EvidenceSource] = field(default_factory=list)


def _probabilities(scores: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(scores), dtype=np.float64)
    if values.size != 8 or not np.all(np.isfinite(values)):
        return np.full(8, 1.0 / 8.0)
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    return values / total if total > 1e-12 else np.full(8, 1.0 / 8.0)


def _rule_probabilities(scores: Iterable[float], confidence: float) -> list[float]:
    """Turn similarity scores into useful evidence without overstating weak rules."""
    values = np.asarray(list(scores), dtype=np.float64)
    if values.size != 8 or not np.all(np.isfinite(values)):
        return [1.0 / 8.0] * 8
    temperature = max(0.055, 0.34 * (1.0 - max(0.0, min(confidence, 1.0))))
    logits = (values - float(values.max())) / temperature
    probabilities = np.exp(np.clip(logits, -30.0, 0.0))
    probabilities /= max(float(probabilities.sum()), 1e-12)
    return probabilities.tolist()


def legacy_marginals(ranked_triples: list[list[int]], limit: int = 96) -> list[list[float]]:
    marginals = np.zeros((3, 8), dtype=np.float64)
    for rank, triple in enumerate(sanitize_ranked_triples(ranked_triples)[:limit]):
        weight = float(np.exp(-rank / 18.0))
        for question_index, answer in enumerate(triple):
            marginals[question_index, answer - 1] += weight
    return [_probabilities(row).tolist() for row in marginals]


def combine_question(
    rule_scores: list[float],
    rule_confidence: float,
    legacy_scores: list[float] | None = None,
    visual_vote: ModelVote | None = None,
    text_vote: ModelVote | None = None,
) -> QuestionEnsemble:
    sources = [
        EvidenceSource(
            name="rule_engine",
            scores=_rule_probabilities(rule_scores, rule_confidence),
            weight=0.20 + 0.45 * max(0.0, min(rule_confidence, 1.0)),
        )
    ]
    if legacy_scores is not None:
        sources.append(EvidenceSource(name="legacy_visual_prior", scores=legacy_scores, weight=0.12))
    if visual_vote is not None:
        sources.append(
            EvidenceSource(
                name="vlm",
                scores=visual_vote.score_vector(),
                # The VLM sees the actual panels and is the primary open-set reasoner.
                # A model that emits a zero/placeholder confidence is weak
                # evidence; its answer must not overrule agreeing CV + legacy.
                weight=0.10 + 1.90 * visual_vote.confidence,
            )
        )
    if text_vote is not None:
        sources.append(
            EvidenceSource(
                name="text_verifier",
                scores=text_vote.score_vector(),
                # Text verification only sees symbolic summaries, so it must not
                # overturn a confident visual choice by itself.
                weight=0.05 + 0.30 * text_vote.confidence,
            )
        )

    combined = np.zeros(8, dtype=np.float64)
    total_weight = 0.0
    for source in sources:
        combined += source.weight * _probabilities(source.scores)
        total_weight += source.weight
    probabilities = combined / max(total_weight, 1e-12)
    order = np.argsort(probabilities)[::-1]
    top_probability = float(probabilities[order[0]])
    margin = float(top_probability - probabilities[order[1]])

    source_answers = [int(np.argmax(_probabilities(source.scores))) for source in sources]
    agreement = source_answers.count(int(order[0])) / max(len(source_answers), 1)
    confidence = float(np.clip(0.55 * top_probability + 0.75 * margin + 0.25 * agreement, 0.0, 1.0))
    return QuestionEnsemble(
        probabilities=probabilities.tolist(),
        answer=int(order[0]) + 1,
        confidence=confidence,
        margin=margin,
        sources=sources,
    )


def rank_answer_triples(questions: list[QuestionEnsemble]) -> list[list[int]]:
    if len(questions) != 3:
        raise ValueError(f"Expected three Raven questions, got {len(questions)}")
    ranked: list[tuple[float, list[int]]] = []
    for zero_based in product(range(8), repeat=3):
        score = sum(
            log(max(questions[index].probabilities[candidate], 1e-12))
            for index, candidate in enumerate(zero_based)
        )
        ranked.append((score, [candidate + 1 for candidate in zero_based]))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [triple for _, triple in ranked]
