from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger
from PIL import Image

from arenaagent.preliminary_baseline_agent.tasks.raven.ensemble import (
    combine_question,
    legacy_marginals,
)
from arenaagent.preliminary_baseline_agent.tasks.raven.experience import (
    relevant_experience_hints,
)
from arenaagent.preliminary_baseline_agent.tasks.raven.reasoners import ask_text_verifier, ask_visual_reasoner
from arenaagent.preliminary_baseline_agent.tasks.raven.rules import RuleInductionResult, induce_rules
from arenaagent.preliminary_baseline_agent.tasks.raven.verifier import ModelVote, verify_selected_answers
from arenaagent.preliminary_baseline_agent.tasks.raven.vision import (
    PanelObservation,
    extract_question_observations,
)


@dataclass(slots=True)
class HybridRavenResult:
    ranked_triples: list[list[int]]
    selected_answers: list[int]
    confidences: list[float]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _enabled(env_name: str, default: bool = True) -> bool:
    raw = os.getenv(env_name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _compact_panel(observation: PanelObservation, panel: int) -> dict[str, Any]:
    features = observation.features.vector()
    keep = (
        "ink_ratio",
        "component_count",
        "hole_count",
        "centroid_x",
        "centroid_y",
        "width_ratio",
        "height_ratio",
        "polygon_vertex_sum",
        "polygon_vertex_mean",
        "fill_density",
        "mean_darkness",
        "fill_category",
        "outline_ratio",
        "object_area_mean",
        "object_area_cv",
        "object_width_mean",
        "object_width_cv",
        "top_vertex_count",
        "bottom_vertex_count",
        "top_outline",
        "bottom_outline",
        "top_darkness",
        "bottom_darkness",
        "top_width",
        "bottom_width",
        "horizontal_symmetry",
        "vertical_symmetry",
        "orientation_sin",
        "orientation_cos",
    )
    graph_summary = observation.graph.summary()
    return {
        "panel": panel,
        "features": {key: round(features[key], 3) for key in keep},
        "graph": {
            "node_count": graph_summary["node_count"],
            "relation_counts": graph_summary["relation_counts"],
        },
    }


def _deterministic_summary(
    question: int,
    observations: list[PanelObservation],
    rules: RuleInductionResult,
    legacy_scores: list[float] | None,
) -> dict[str, Any]:
    rule_order = np.argsort(np.asarray(rules.scores))[::-1]
    feature_maps = [observation.features.vector() for observation in observations]
    derived_constraints: dict[str, Any] = {
        "candidate_contrast": [
            {
                "id": index - 7,
                "width": round(feature_maps[index]["width_ratio"], 3),
                "height": round(feature_maps[index]["height_ratio"], 3),
                "darkness": round(feature_maps[index]["mean_darkness"], 3),
                "components": round(feature_maps[index]["component_count"], 1),
                "vertices": round(feature_maps[index]["polygon_vertex_mean"], 1),
            }
            for index in range(8, 16)
        ]
    }
    if any(
        features["polygon_vertex_mean"] >= 7.5
        and abs(features["width_ratio"] - features["height_ratio"]) <= 0.12
        for features in feature_maps[:8]
    ):
        derived_constraints["polygon_vertex_warning"] = (
            "Circle-like panels are present. Their polygon vertex counts are contour approximations; "
            "do not use vertex sums or arithmetic to change a circle into a polygon."
        )
    size_trends: dict[str, Any] = {}
    for feature_name in ("width_ratio", "height_ratio"):
        values = [features[feature_name] for features in feature_maps]
        steps = [
            values[1] - values[0],
            values[2] - values[1],
            values[4] - values[3],
            values[5] - values[4],
        ]
        mean_step = float(np.mean(steps))
        target_step = values[7] - values[6]
        if (
            abs(mean_step) >= 0.035
            and all(step * mean_step > 0 for step in steps)
            and target_step * mean_step > 0
        ):
            predicted = values[7] + mean_step
            size_trends[feature_name] = {
                "direction": "increase" if mean_step > 0 else "decrease",
                "target_row_observed": [round(values[6], 3), round(values[7], 3)],
                "predicted_missing": round(predicted, 3),
                "closest_candidates": [
                    index - 7
                    for index in sorted(range(8, 16), key=lambda item: abs(values[item] - predicted))[:4]
                ],
            }
    if size_trends:
        derived_constraints["monotonic_size"] = size_trends
    summary: dict[str, Any] = {
        "question": question,
        "context_panels": [_compact_panel(observations[index], index + 1) for index in range(8)],
        "candidate_panels": [_compact_panel(observations[index], index - 7) for index in range(8, 16)],
        "rule_engine": rules.summary(),
        "rule_top_candidates": [int(index) + 1 for index in rule_order[:3]],
        "derived_constraints": derived_constraints,
    }
    if legacy_scores is not None:
        legacy_order = np.argsort(np.asarray(legacy_scores))[::-1]
        summary["legacy_top_candidates"] = [int(index) + 1 for index in legacy_order[:3]]
    return summary


def _vote_for_question(votes: list[ModelVote], question: int) -> ModelVote | None:
    return next((vote for vote in votes if vote.question == question), None)


def _vote_from_summary(value: Any) -> ModelVote | None:
    if not isinstance(value, dict):
        return None
    try:
        question = int(value.get("question"))
        answer = int(value.get("answer"))
        confidence = float(value.get("confidence", 0.5))
    except (TypeError, ValueError):
        return None
    if not 1 <= question <= 3 or not 1 <= answer <= 8:
        return None

    def numeric_map(raw: Any) -> dict[int, float]:
        result: dict[int, float] = {}
        if not isinstance(raw, dict):
            return result
        for key, item in raw.items():
            try:
                candidate = int(key)
                score = float(item)
            except (TypeError, ValueError):
                continue
            if 1 <= candidate <= 8:
                result[candidate] = score
        return result

    mismatches: dict[int, list[str]] = {}
    raw_mismatches = value.get("candidate_mismatches") or {}
    if not isinstance(raw_mismatches, dict):
        raw_mismatches = {}
    for key, item in raw_mismatches.items():
        try:
            candidate = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(item, list):
            mismatches[candidate] = [str(text) for text in item]
    return ModelVote(
        question=question,
        answer=answer,
        confidence=max(0.0, min(confidence, 1.0)),
        alternatives=numeric_map(value.get("alternatives")),
        candidate_confidences=numeric_map(value.get("candidate_confidences")),
        rule=str(value.get("rule") or ""),
        critique=str(value.get("critique") or ""),
        predicted_attributes=dict(value.get("predicted_attributes") or {}),
        evidence=[str(text) for text in (value.get("evidence") or [])],
        candidate_mismatches=mismatches,
    )


def restore_prior_model_votes(previous_attempts: list[dict[str, Any]] | None) -> list[ModelVote]:
    """Merge the newest successful per-question result across multiple rounds.

    Visual results take priority inside a round. An independently produced text
    result may fill a question whose visual response was malformed or missing.
    """
    restored: dict[int, ModelVote] = {}
    for attempt in reversed(previous_attempts or []):
        if not isinstance(attempt, dict):
            continue
        for field in ("visual_votes", "text_votes"):
            values = attempt.get(field, [])
            if not isinstance(values, list):
                continue
            for value in values:
                vote = _vote_from_summary(value)
                if vote is not None and vote.question not in restored:
                    restored[vote.question] = vote
        if len(restored) == 3:
            break
    return [restored[question] for question in sorted(restored)]


def select_suspect_questions(previous_attempts: list[dict[str, Any]] | None) -> list[int]:
    """Infer which digit deserves another LLM look without claiming server feedback."""
    latest: dict[str, Any] | None = None
    for attempt in reversed(previous_attempts or []):
        if isinstance(attempt, dict) and (attempt.get("visual_votes") or attempt.get("text_votes")):
            latest = attempt
            break
    if latest is None:
        return [1, 2, 3]
    votes = restore_prior_model_votes(previous_attempts)
    rule_top = latest.get("rule_top_candidates") or latest.get("rule_top") or [[], [], []]
    legacy_top = latest.get("legacy_top_candidates") or [[], [], []]
    scores: list[tuple[float, int]] = []
    for question in range(1, 4):
        vote = _vote_for_question(votes, question)
        if vote is None:
            scores.append((10.0, question))
            continue
        rule = [int(value) for value in rule_top[question - 1]] if len(rule_top) >= question else []
        legacy = [int(value) for value in legacy_top[question - 1]] if len(legacy_top) >= question else []
        distribution = vote.candidate_confidences or {
            vote.answer: vote.confidence,
            **vote.alternatives,
        }
        challenger_strength = 0.0
        for candidate, probability in distribution.items():
            if candidate == vote.answer:
                continue
            independent_support = int(candidate in rule) + int(candidate in legacy)
            challenger_strength = max(
                challenger_strength,
                float(probability) * (1.0 + 1.5 * independent_support),
            )
        disagreement = 0.0
        if vote.answer not in rule:
            disagreement += 0.35
        if vote.answer not in legacy:
            disagreement += 0.35
        score = challenger_strength + disagreement + 0.25 * (1.0 - vote.confidence)
        scores.append((score, question))
    scores.sort(reverse=True)
    # Revisit a second question only when its evidence conflict is essentially
    # tied with the first.  Usually one focused request is both faster and more
    # informative under whole-triple-only feedback.
    selected = [scores[0][1]]
    if len(scores) > 1 and scores[1][0] >= scores[0][0] * 0.92:
        selected.append(scores[1][1])
    return sorted(selected)


class HybridRavenSolver:
    """CV + scene graph + rule induction + model ensemble Raven solver."""

    def __init__(self, vlm_client: Any = None, text_client: Any = None) -> None:
        self.vlm_client = vlm_client
        # A verifier must be genuinely independent. Reusing the visual client
        # doubled latency and amplified the same model's mistakes.
        self.text_client = text_client

    def solve(
        self,
        image_groups: list[list[Image.Image]],
        legacy_ranked: list[list[int]] | None = None,
        rejected_triples: list[list[int]] | None = None,
        previous_attempts: list[dict[str, Any]] | None = None,
        whole_image: Image.Image | None = None,
    ) -> HybridRavenResult:
        if len(image_groups) != 3:
            raise ValueError(f"Competition Raven canvas must contain three questions, got {len(image_groups)}")

        solve_started = time.perf_counter()
        observations = [extract_question_observations(group) for group in image_groups]
        rule_results = [induce_rules(question) for question in observations]
        deterministic_finished = time.perf_counter()
        legacy_scores = legacy_marginals(legacy_ranked or []) if legacy_ranked else [None, None, None]
        deterministic = [
            _deterministic_summary(index + 1, observations[index], rule_results[index], legacy_scores[index])
            for index in range(3)
        ]
        if _enabled("RAVEN_ENABLE_EXPERIENCE"):
            for index, rule_result in enumerate(rule_results):
                experience = relevant_experience_hints(rule_result)
                if experience is not None:
                    deterministic[index]["verified_experience"] = experience
        correction_context = None
        if rejected_triples:
            correction_context = {
                "rejected_triples": rejected_triples,
                "previous_attempts": (previous_attempts or [])[-4:],
            }

        prior_visual_votes: list[ModelVote] = []
        target_questions = [1, 2, 3]
        reasoning_mode = "whole_image_single_request"
        if rejected_triples and previous_attempts:
            prior_visual_votes = restore_prior_model_votes(previous_attempts)
            if prior_visual_votes:
                reasoning_mode = "whole_image_single_revision"

        visual_votes: list[ModelVote] = list(prior_visual_votes)
        visual_raw = ""
        if self.vlm_client is not None and _enabled("RAVEN_ENABLE_VLM"):
            try:
                logger.info(
                    "Raven {}: dispatching one K3 request with the original canvas for questions {}",
                    reasoning_mode,
                    target_questions,
                )
                revised_votes, visual_raw = ask_visual_reasoner(
                    self.vlm_client,
                    image_groups,
                    deterministic,
                    correction_context=correction_context,
                    target_questions=target_questions,
                    whole_image=whole_image,
                )
                revised_by_question = {vote.question: vote for vote in revised_votes}
                prior_by_question = {vote.question: vote for vote in prior_visual_votes}
                visual_votes = [
                    revised_by_question.get(question) or prior_by_question.get(question)
                    for question in range(1, 4)
                ]
                visual_votes = [vote for vote in visual_votes if vote is not None]
            except Exception as exc:
                logger.warning("Raven visual reasoner failed; keep deterministic fallback: {}", exc)
        visual_finished = time.perf_counter()

        preliminary = [
            combine_question(
                rule_scores=rule_results[index].scores,
                rule_confidence=rule_results[index].confidence,
                legacy_scores=legacy_scores[index],
                visual_vote=_vote_for_question(visual_votes, index + 1),
            )
            for index in range(3)
        ]

        # Rejection feedback is already supplied to the visual reasoner.  Do not
        # automatically make a second, equally expensive image-model call on
        # every retry: on the competition server that caused one subject to run
        # past its deadline and leak a stale answer into the next subject.
        missing_visual_questions = [
            question for question in range(1, 4) if _vote_for_question(visual_votes, question) is None
        ]
        verification_targets = missing_visual_questions
        if not verification_targets and reasoning_mode == "whole_image_single_revision":
            verification_targets = target_questions
        needs_text_verifier = bool(verification_targets)
        text_votes: list[ModelVote] = []
        text_raw = ""
        if (
            needs_text_verifier
            and self.text_client is not None
            and _enabled("RAVEN_ENABLE_TEXT_VERIFIER")
        ):
            try:
                text_votes, text_raw = ask_text_verifier(
                    self.text_client,
                    deterministic,
                    visual_votes,
                    correction_context=correction_context,
                    # DeepSeek-V4-Pro is the independent text verifier. It
                    # receives structured CV/scene/rule evidence, not images.
                    image_groups=None,
                    target_questions=verification_targets,
                )
            except Exception as exc:
                logger.warning("Raven text verifier failed; keep current ensemble: {}", exc)
        verifier_finished = time.perf_counter()

        final_questions = [
            combine_question(
                rule_scores=rule_results[index].scores,
                rule_confidence=rule_results[index].confidence,
                legacy_scores=legacy_scores[index],
                visual_vote=_vote_for_question(visual_votes, index + 1),
                text_vote=_vote_for_question(text_votes, index + 1),
            )
            for index in range(3)
        ]
        selected = verify_selected_answers([question.answer for question in final_questions])
        conservative_overrides: list[dict[str, Any]] = []
        if not rejected_triples and legacy_scores:
            try:
                override_threshold = float(os.getenv("RAVEN_VLM_OVERRIDE_CONFIDENCE", "0.85"))
            except ValueError:
                override_threshold = 0.85
            override_threshold = max(0.0, min(override_threshold, 1.0))
            for index in range(3):
                legacy_answer = int(np.argmax(np.asarray(legacy_scores[index]))) + 1
                if selected[index] == legacy_answer:
                    continue
                visual_vote = _vote_for_question(visual_votes, index + 1)
                rule_answer = int(np.argmax(np.asarray(rule_results[index].scores))) + 1
                visual_can_override = bool(
                    visual_vote is not None
                    and visual_vote.answer == selected[index]
                    and rule_answer == selected[index]
                    and visual_vote.confidence >= override_threshold
                )
                if not visual_can_override:
                    conservative_overrides.append(
                        {
                            "question": index + 1,
                            "fused_answer": selected[index],
                            "legacy_answer": legacy_answer,
                            "visual_answer": visual_vote.answer if visual_vote else None,
                            "visual_confidence": visual_vote.confidence if visual_vote else None,
                            "rule_answer": rule_answer,
                        }
                    )
                    selected[index] = legacy_answer
        visual_fallback_to_legacy = False
        if not visual_votes and not text_votes and legacy_ranked:
            # A quota, timeout or malformed response is absence of visual
            # evidence, not evidence against the legacy model.  Preserve the
            # old model's first choice instead of allowing weak CV rules to
            # silently change an otherwise valid first submission.
            selected = verify_selected_answers(list(legacy_ranked[0]))
            visual_fallback_to_legacy = True
        rejected_set = {tuple(item) for item in (rejected_triples or [])}
        selection_rejected = tuple(selected) in rejected_set
        # Do not walk a 512-combination leaderboard after rejection. A new
        # answer is submitted only when the fresh/reused evidence changes the
        # per-question argmax to a previously untried triple.
        ranked_triples = [] if selection_rejected else [selected]
        diagnostics = {
            "selected_answers": selected,
            "question_confidences": [round(question.confidence, 4) for question in final_questions],
            "question_margins": [round(question.margin, 4) for question in final_questions],
            # Keep the auditable, public answer distribution so a rejected
            # triple can be repaired locally without asking the model to solve
            # all three questions again.
            "question_probabilities": [
                [round(float(value), 6) for value in question.probabilities]
                for question in final_questions
            ],
            "rule_top_candidates": [item["rule_top_candidates"] for item in deterministic],
            "legacy_top_candidates": [
                item.get("legacy_top_candidates", []) for item in deterministic
            ],
            "rule_results": [result.summary() for result in rule_results],
            "visual_votes": [self._vote_summary(vote) for vote in visual_votes],
            "reasoning_mode": reasoning_mode,
            "revisited_questions": target_questions,
            "text_verifier_used": bool(text_votes),
            "text_verifier_configured": self.text_client is not None,
            "text_verifier_questions": verification_targets if self.text_client is not None else [],
            "text_votes": [self._vote_summary(vote) for vote in text_votes],
            "selection_rejected": selection_rejected,
            "visual_fallback_to_legacy": visual_fallback_to_legacy,
            "conservative_legacy_restores": conservative_overrides,
            "visual_raw_excerpt": visual_raw[:1000],
            "text_raw_excerpt": text_raw[:1000],
            "rejected_triples": rejected_triples or [],
            "timing_seconds": {
                "deterministic": round(deterministic_finished - solve_started, 3),
                "visual": round(visual_finished - deterministic_finished, 3),
                "text_verifier": round(verifier_finished - visual_finished, 3),
                "total": round(verifier_finished - solve_started, 3),
            },
        }
        logger.info(
            "Raven evidence rule_top={} legacy_top={} visual={} text={} timing={}",
            [item["rule_top_candidates"] for item in deterministic],
            [item.get("legacy_top_candidates", []) for item in deterministic],
            diagnostics["visual_votes"],
            diagnostics["text_votes"],
            diagnostics["timing_seconds"],
        )
        logger.info(
            "Hybrid Raven selected {} with confidence {} (text_verifier={})",
            selected,
            diagnostics["question_confidences"],
            bool(text_votes),
        )
        return HybridRavenResult(
            ranked_triples=ranked_triples,
            selected_answers=selected,
            confidences=[question.confidence for question in final_questions],
            diagnostics=diagnostics,
        )

    @staticmethod
    def _vote_summary(vote: ModelVote) -> dict[str, Any]:
        return {
            "question": vote.question,
            "answer": vote.answer,
            "confidence": vote.confidence,
            "alternatives": vote.alternatives,
            "candidate_confidences": vote.candidate_confidences,
            "rule": vote.rule,
            "critique": vote.critique,
            "predicted_attributes": vote.predicted_attributes,
            "evidence": vote.evidence,
            "candidate_mismatches": vote.candidate_mismatches,
        }
