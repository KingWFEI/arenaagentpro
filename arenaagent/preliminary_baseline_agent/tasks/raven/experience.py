from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image

from arenaagent.preliminary_baseline_agent.tasks.raven.rules import (
    RuleInductionResult,
    induce_rules,
)
from arenaagent.preliminary_baseline_agent.tasks.raven.vision import (
    extract_question_observations,
)


EXPERIENCE_VERSION = 1


def default_experience_path(log_dir: str = "logs") -> Path:
    configured = os.getenv("RAVEN_EXPERIENCE_PATH", "").strip()
    return Path(configured) if configured else Path(log_dir) / "raven_experience.json"


def _empty_store() -> dict[str, Any]:
    return {
        "version": EXPERIENCE_VERSION,
        "verified_subjects": {},
        "techniques": {},
        "lessons": [],
    }


def _load_store(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return _empty_store()
    if not isinstance(value, dict) or value.get("version") != EXPERIENCE_VERSION:
        return _empty_store()
    value.setdefault("verified_subjects", {})
    value.setdefault("techniques", {})
    value.setdefault("lessons", [])
    return value


def _atomic_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="raven_experience_",
        suffix=".json",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as writer:
            json.dump(value, writer, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.remove(temporary_name)
        except OSError:
            pass
        raise


def _rule_key(name: str, direction: str) -> str:
    return f"{name}|{direction}"


def _describe_rule(name: str, direction: str) -> str:
    feature, _, operator = name.partition(":")
    feature_label = {
        "width_ratio": "object width",
        "height_ratio": "object height",
        "component_count": "object count",
        "hole_count": "hole count",
        "fill_category": "fill level",
        "mean_darkness": "shade",
        "centroid_x": "horizontal position",
        "centroid_y": "vertical position",
        "polygon_vertex_sum": "polygon family",
        "polygon_vertex_mean": "polygon family",
        "pixel_union": "pixel union",
        "pixel_xor": "pixel XOR",
        "pixel_intersection": "pixel intersection",
        "pixel_subtract": "pixel subtraction",
    }.get(feature, feature.replace("_", " "))
    operator_label = {
        "monotonic_decrease": "continues a consistent decrease",
        "monotonic_increase": "continues a consistent increase",
        "distribute_three": "distributes three attribute values once per line",
        "cyclic": "uses a cyclic permutation",
        "copy_left": "copies the first panel",
        "copy_right": "copies the second panel",
        "sum": "combines the first two values",
        "absolute_difference": "uses the absolute difference",
        "mean": "uses the mean of the first two values",
        "arithmetic_progression": "continues an arithmetic progression",
    }.get(operator, operator.replace("_", " ") if operator else "matches the induced transform")
    return (
        f"On a verified solved matrix, {feature_label} {operator_label} along the {direction}. "
        "Test the same relation on every complete line before using it on the missing cell."
    )


def _supporting_rules(result: RuleInductionResult, answer: int) -> list[tuple[Any, float]]:
    selected_index = answer - 1
    supporting: list[tuple[Any, float]] = []
    for match in result.matches:
        if len(match.candidate_scores) != 8 or not 0 <= selected_index < 8:
            continue
        selected_score = float(match.candidate_scores[selected_index])
        best_score = max(float(value) for value in match.candidate_scores)
        if match.reliability < 0.52 or selected_score < best_score - 1e-6:
            continue
        others = [
            float(value)
            for index, value in enumerate(match.candidate_scores)
            if index != selected_index
        ]
        advantage = selected_score - (max(others) if others else 0.0)
        supporting.append((match, advantage))
    supporting.sort(key=lambda item: (item[1], item[0].reliability), reverse=True)
    return supporting[:8]


def _clean_lesson(text: str) -> str:
    compact = " ".join(str(text).split())
    compact = re.sub(r"(?i)candidate\s*[1-8]", "the matching option", compact)
    compact = re.sub(r"(?i)previous (answer|choice)\s*[1-8]?", "the earlier proposal", compact)
    return compact[:700]


def _vote_for_question(diagnostics: dict[str, Any], question: int, answer: int) -> dict[str, Any] | None:
    for field in ("visual_votes", "text_votes"):
        values = diagnostics.get(field, [])
        if not isinstance(values, list):
            continue
        for vote in values:
            if not isinstance(vote, dict):
                continue
            try:
                vote_question = int(vote.get("question"))
                vote_answer = int(vote.get("answer"))
            except (TypeError, ValueError):
                continue
            if vote_question == question and vote_answer == answer:
                return vote
    return None


def record_successful_subject(
    image_groups: list[list[Image.Image]],
    answers: list[int],
    *,
    diagnostics: dict[str, Any] | None = None,
    subject_fingerprint: str = "",
    path: Path | None = None,
) -> bool:
    """Learn only explicit, auditable techniques from a server-confirmed success.

    Exact answer IDs and images are never placed in prompts. The fingerprint is
    retained solely to prevent the same training canvas from being counted many
    times across repeated runs.
    """
    if len(image_groups) != 3 or len(answers) != 3 or any(not 1 <= value <= 8 for value in answers):
        return False
    destination = path or default_experience_path()
    store = _load_store(destination)
    fingerprint = subject_fingerprint or hashlib.sha256(
        b"".join(image.tobytes() for group in image_groups for image in group)
    ).hexdigest()
    known = store["verified_subjects"].get(fingerprint)
    if isinstance(known, dict):
        known["confirmations"] = int(known.get("confirmations", 1)) + 1
        _atomic_save(destination, store)
        return False

    rule_results = [
        induce_rules(extract_question_observations(group))
        for group in image_groups
    ]
    diagnostics = diagnostics or {}
    for question, (answer, result) in enumerate(zip(answers, rule_results), start=1):
        supporting = _supporting_rules(result, answer)
        signatures = [_rule_key(match.name, match.direction) for match, _ in supporting]
        for match, advantage in supporting:
            key = _rule_key(match.name, match.direction)
            existing = store["techniques"].setdefault(
                key,
                {
                    "name": match.name,
                    "direction": match.direction,
                    "successes": 0,
                    "mean_reliability": 0.0,
                    "mean_advantage": 0.0,
                    "tip": _describe_rule(match.name, match.direction),
                },
            )
            count = int(existing.get("successes", 0))
            existing["successes"] = count + 1
            existing["mean_reliability"] = round(
                (float(existing.get("mean_reliability", 0.0)) * count + match.reliability)
                / (count + 1),
                4,
            )
            existing["mean_advantage"] = round(
                (float(existing.get("mean_advantage", 0.0)) * count + advantage) / (count + 1),
                4,
            )

        vote = _vote_for_question(diagnostics, question, answer)
        if vote is None:
            continue
        rule_text = _clean_lesson(str(vote.get("rule") or ""))
        evidence = [
            _clean_lesson(str(item))
            for item in (vote.get("evidence") or [])[:3]
            if str(item).strip()
        ]
        lesson_text = " ".join([rule_text, *evidence]).strip()
        if not lesson_text:
            continue
        lesson_id = hashlib.sha256(lesson_text.encode("utf-8")).hexdigest()[:16]
        existing_lesson = next(
            (item for item in store["lessons"] if item.get("id") == lesson_id),
            None,
        )
        if existing_lesson is not None:
            existing_lesson["successes"] = int(existing_lesson.get("successes", 1)) + 1
        else:
            store["lessons"].append(
                {
                    "id": lesson_id,
                    "successes": 1,
                    "signatures": signatures,
                    "tip": lesson_text,
                }
            )

    store["verified_subjects"][fingerprint] = {
        "confirmations": 1,
        "questions": 3,
    }
    store["lessons"] = sorted(
        store["lessons"],
        key=lambda item: int(item.get("successes", 0)),
        reverse=True,
    )[:120]
    _atomic_save(destination, store)
    logger.info(
        "Recorded verified Raven experience fingerprint={} techniques={} lessons={} path={}",
        fingerprint[:16],
        len(store["techniques"]),
        len(store["lessons"]),
        destination,
    )
    return True


def relevant_experience_hints(
    result: RuleInductionResult,
    *,
    path: Path | None = None,
    limit: int = 6,
) -> dict[str, Any] | None:
    destination = path or default_experience_path()
    store = _load_store(destination)
    if not store["verified_subjects"]:
        return None
    current = {
        _rule_key(match.name, match.direction)
        for match in result.matches
        if match.reliability >= 0.50
    }
    techniques = [
        value
        for key, value in store["techniques"].items()
        if key in current
    ]
    techniques.sort(
        key=lambda value: (
            int(value.get("successes", 0)),
            float(value.get("mean_reliability", 0.0)),
            float(value.get("mean_advantage", 0.0)),
        ),
        reverse=True,
    )
    lessons: list[tuple[float, dict[str, Any]]] = []
    for lesson in store["lessons"]:
        signatures = set(lesson.get("signatures") or [])
        if not signatures:
            continue
        overlap = len(current & signatures)
        if overlap == 0:
            continue
        similarity = overlap / max(1, min(len(current), len(signatures)))
        lessons.append((similarity * int(lesson.get("successes", 1)), lesson))
    lessons.sort(key=lambda item: item[0], reverse=True)
    return {
        "verified_unique_subjects": len(store["verified_subjects"]),
        "matched_techniques": [
            {
                "successes": int(item.get("successes", 0)),
                "tip": item.get("tip"),
            }
            for item in techniques[:limit]
        ],
        "similar_solved_lessons": [item.get("tip") for _, item in lessons[:3]],
        "usage_guardrail": (
            "These are transferable priors from server-verified successes, not answer keys. "
            "Apply a tip only if the current image independently satisfies it."
        ),
    }
