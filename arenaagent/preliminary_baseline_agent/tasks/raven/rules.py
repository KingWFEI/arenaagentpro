from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Callable

import cv2
import numpy as np

from arenaagent.preliminary_baseline_agent.tasks.raven.vision import PanelObservation


@dataclass(slots=True, frozen=True)
class RuleMatch:
    name: str
    direction: str
    reliability: float
    candidate_scores: tuple[float, ...]


@dataclass(slots=True)
class RuleInductionResult:
    scores: list[float]
    confidence: float
    matches: list[RuleMatch]

    def summary(self, limit: int = 10) -> dict[str, object]:
        ranked = sorted(self.matches, key=lambda item: item.reliability, reverse=True)[:limit]
        return {
            "candidate_scores": [round(value, 4) for value in self.scores],
            "confidence": round(self.confidence, 4),
            "rules": [
                {
                    "name": item.name,
                    "direction": item.direction,
                    "reliability": round(item.reliability, 4),
                }
                for item in ranked
            ],
        }


ScalarOperator = Callable[[float, float], float]
MaskOperator = Callable[[np.ndarray, np.ndarray], np.ndarray]

_SCALAR_OPERATORS: tuple[tuple[str, ScalarOperator, float], ...] = (
    ("copy_left", lambda left, right: left, 0.92),
    ("copy_right", lambda left, right: right, 0.92),
    ("sum", lambda left, right: left + right, 0.86),
    ("absolute_difference", lambda left, right: abs(left - right), 0.88),
    ("arithmetic_progression", lambda left, right: 2.0 * right - left, 0.82),
    ("mean", lambda left, right: (left + right) / 2.0, 0.82),
    ("minimum", min, 0.84),
    ("maximum", max, 0.84),
)


def _as_bool(mask: np.ndarray) -> np.ndarray:
    return mask > 0


_MASK_OPERATORS: tuple[tuple[str, MaskOperator, float], ...] = (
    ("pixel_copy_left", lambda left, right: _as_bool(left), 0.94),
    ("pixel_copy_right", lambda left, right: _as_bool(right), 0.94),
    ("pixel_union", lambda left, right: np.logical_or(_as_bool(left), _as_bool(right)), 1.0),
    ("pixel_intersection", lambda left, right: np.logical_and(_as_bool(left), _as_bool(right)), 0.98),
    ("pixel_xor", lambda left, right: np.logical_xor(_as_bool(left), _as_bool(right)), 1.0),
    (
        "pixel_left_minus_right",
        lambda left, right: np.logical_and(_as_bool(left), np.logical_not(_as_bool(right))),
        0.96,
    ),
    (
        "pixel_right_minus_left",
        lambda left, right: np.logical_and(_as_bool(right), np.logical_not(_as_bool(left))),
        0.96,
    ),
    ("rotate_right_90", lambda left, right: np.rot90(_as_bool(right), 3), 0.88),
    ("rotate_right_180", lambda left, right: np.rot90(_as_bool(right), 2), 0.88),
    ("flip_right_horizontal", lambda left, right: np.flip(_as_bool(right), axis=1), 0.88),
    ("flip_right_vertical", lambda left, right: np.flip(_as_bool(right), axis=0), 0.88),
)

_FEATURE_SCALES: dict[str, float] = {
    "ink_ratio": 0.055,
    "component_count": 1.0,
    "hole_count": 1.0,
    "centroid_x": 0.12,
    "centroid_y": 0.12,
    "width_ratio": 0.12,
    "height_ratio": 0.12,
    "polygon_vertex_sum": 1.0,
    "polygon_vertex_mean": 1.0,
    "fill_density": 0.16,
    "mean_darkness": 0.16,
    "fill_category": 1.0,
    "outline_ratio": 0.25,
    "object_area_mean": 0.018,
    "object_area_cv": 0.20,
    "object_width_mean": 0.10,
    "object_width_cv": 0.18,
    "top_vertex_count": 1.0,
    "bottom_vertex_count": 1.0,
    "top_outline": 1.0,
    "bottom_outline": 1.0,
    "top_darkness": 0.16,
    "bottom_darkness": 0.16,
    "top_width": 0.10,
    "bottom_width": 0.10,
    "horizontal_symmetry": 0.18,
    "vertical_symmetry": 0.18,
    "orientation_sin": 0.28,
    "orientation_cos": 0.28,
    "quadrant_nw": 0.14,
    "quadrant_ne": 0.14,
    "quadrant_sw": 0.14,
    "quadrant_se": 0.14,
}

_FEATURE_PRIORS: dict[str, float] = {
    "component_count": 3.0,
    "polygon_vertex_sum": 2.2,
    "polygon_vertex_mean": 2.2,
    "fill_density": 1.8,
    "mean_darkness": 1.8,
    "fill_category": 3.0,
    "outline_ratio": 2.6,
    "object_area_mean": 1.8,
    "object_area_cv": 1.5,
    "object_width_mean": 1.8,
    "object_width_cv": 1.5,
    "top_vertex_count": 2.8,
    "bottom_vertex_count": 2.8,
    "top_outline": 2.2,
    "bottom_outline": 2.2,
    "top_darkness": 2.2,
    "bottom_darkness": 2.2,
    "top_width": 2.0,
    "bottom_width": 2.0,
    "relative_width_level": 2.4,
    "relative_height_level": 2.4,
    "width_ratio": 3.2,
    "height_ratio": 3.2,
    "ink_ratio": 1.2,
}

_GENERIC_VERTEX_FEATURES = {"polygon_vertex_sum", "polygon_vertex_mean"}


def _contains_circle_like_context(feature_maps: list[dict[str, float]]) -> bool:
    """Detect when polygon vertices are merely a circle approximation."""
    return any(
        features["polygon_vertex_mean"] >= 7.5
        and abs(features["width_ratio"] - features["height_ratio"]) <= 0.12
        for features in feature_maps[:8]
    )


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    left_bool, right_bool = _as_bool(left), _as_bool(right)
    union = np.logical_or(left_bool, right_bool).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(left_bool, right_bool).sum() / union)


def _soft_similarity(actual: float, predicted: float, scale: float) -> float:
    return exp(-abs(actual - predicted) / max(scale, 1e-6))


def _direction_layouts() -> tuple[tuple[str, tuple[tuple[int, int, int], ...], tuple[int, int]], ...]:
    return (
        ("row", ((0, 1, 2), (3, 4, 5)), (6, 7)),
        ("column", ((0, 3, 6), (1, 4, 7)), (2, 5)),
    )


def _induce_scalar_rules(observations: list[PanelObservation]) -> list[RuleMatch]:
    feature_maps = [observation.features.vector() for observation in observations]
    circle_like_context = _contains_circle_like_context(feature_maps)
    matches: list[RuleMatch] = []
    for direction, training, target_pair in _direction_layouts():
        for feature_name, base_scale in _FEATURE_SCALES.items():
            values = [features[feature_name] for features in feature_maps]
            observed_range = max(values[:8]) - min(values[:8])
            if observed_range < max(base_scale * 0.08, 1e-4):
                continue
            scale = max(base_scale, observed_range * 0.12)
            for operator_name, operator, complexity_prior in _SCALAR_OPERATORS:
                if (
                    circle_like_context
                    and feature_name in _GENERIC_VERTEX_FEATURES
                    and operator_name not in {"copy_left", "copy_right"}
                ):
                    continue
                training_scores = [
                    _soft_similarity(values[end], operator(values[left], values[right]), scale)
                    for left, right, end in training
                ]
                reliability = float(np.mean(training_scores)) * complexity_prior
                if reliability < 0.58:
                    continue
                predicted = operator(values[target_pair[0]], values[target_pair[1]])
                candidate_scores = tuple(
                    _soft_similarity(values[candidate_index], predicted, scale)
                    for candidate_index in range(8, 16)
                )
                matches.append(
                    RuleMatch(
                        name=f"{feature_name}:{operator_name}",
                        direction=direction,
                        reliability=reliability,
                        candidate_scores=candidate_scores,
                    )
                )
    return matches


def _induce_cyclic_rules(observations: list[PanelObservation]) -> list[RuleMatch]:
    """Detect Latin-square style shifts such as 4,5,6 / 6,4,5 / 5,6,?."""
    feature_maps = [observation.features.vector() for observation in observations]
    circle_like_context = _contains_circle_like_context(feature_maps)
    matches: list[RuleMatch] = []
    for feature_name, base_scale in _FEATURE_SCALES.items():
        if circle_like_context and feature_name in _GENERIC_VERTEX_FEATURES:
            continue
        values = [features[feature_name] for features in feature_maps]
        observed_range = max(values[:8]) - min(values[:8])
        if observed_range < max(base_scale * 0.25, 1e-4):
            continue
        scale = max(base_scale, observed_range * 0.10)

        rows = (values[0:3], values[3:6], values[6:8])
        for shift_name, expected_second, expected_third in (
            ("cyclic_right", [rows[0][2], rows[0][0], rows[0][1]], [rows[0][1], rows[0][2], rows[0][0]]),
            ("cyclic_left", [rows[0][1], rows[0][2], rows[0][0]], [rows[0][2], rows[0][0], rows[0][1]]),
        ):
            checks = [
                _soft_similarity(rows[1][index], expected_second[index], scale) for index in range(3)
            ] + [_soft_similarity(rows[2][index], expected_third[index], scale) for index in range(2)]
            reliability = float(np.mean(checks)) * 0.98
            if reliability >= 0.68:
                predicted = expected_third[2]
                matches.append(
                    RuleMatch(
                        name=f"{feature_name}:{shift_name}",
                        direction="row",
                        reliability=reliability,
                        candidate_scores=tuple(
                            _soft_similarity(values[index], predicted, scale) for index in range(8, 16)
                        ),
                    )
                )
    return matches


def _induce_distribute_three_rules(observations: list[PanelObservation]) -> list[RuleMatch]:
    """Detect RAVEN's Distribute-Three rule as equal unordered triples.

    The official generator treats Type/Size/Color/Number/Position as discrete
    attributes.  CV measurements are slightly noisy, so integer-valued
    attributes are rounded and continuous attributes use their native scale.
    """
    feature_maps = [observation.features.vector() for observation in observations]
    circle_like_context = _contains_circle_like_context(feature_maps)
    categorical = {
        "component_count",
        "hole_count",
        "polygon_vertex_sum",
        "polygon_vertex_mean",
        "fill_category",
        "top_vertex_count",
        "bottom_vertex_count",
        "top_outline",
        "bottom_outline",
    }
    matches: list[RuleMatch] = []
    for direction, training, target_pair in _direction_layouts():
        for feature_name, base_scale in _FEATURE_SCALES.items():
            if circle_like_context and feature_name in _GENERIC_VERTEX_FEATURES:
                continue
            values = [features[feature_name] for features in feature_maps]
            scale = max(base_scale, (max(values[:8]) - min(values[:8])) * 0.10)

            def canonical(value: float) -> float:
                return float(round(value)) if feature_name in categorical else float(value)

            first = sorted(canonical(values[index]) for index in training[0])
            second = sorted(canonical(values[index]) for index in training[1])
            agreement = float(
                np.mean([_soft_similarity(left, right, scale) for left, right in zip(first, second)])
            )
            # A constant triple is already covered by the simpler copy rules.
            if agreement < 0.82 or max(first) - min(first) < max(scale * 0.25, 1e-4):
                continue

            remaining = list(first)
            removal_quality = 1.0
            for index in target_pair:
                value = canonical(values[index])
                nearest = min(range(len(remaining)), key=lambda item: abs(remaining[item] - value))
                removal_quality *= _soft_similarity(value, remaining[nearest], scale)
                remaining.pop(nearest)
            reliability = agreement * removal_quality * 0.98
            if not remaining or reliability < 0.68:
                continue
            predicted = remaining[0]
            matches.append(
                RuleMatch(
                    name=f"{feature_name}:distribute_three",
                    direction=direction,
                    reliability=reliability,
                    candidate_scores=tuple(
                        _soft_similarity(canonical(values[index]), predicted, scale)
                        for index in range(8, 16)
                    ),
                )
            )
    return matches


def _induce_monotonic_size_rules(observations: list[PanelObservation]) -> list[RuleMatch]:
    """Extrapolate stable large→medium→small (or reverse) progressions."""
    feature_maps = [observation.features.vector() for observation in observations]
    matches: list[RuleMatch] = []
    for direction, training, target_pair in _direction_layouts():
        for feature_name in ("width_ratio", "height_ratio"):
            values = [features[feature_name] for features in feature_maps]
            training_steps = [
                values[middle] - values[start]
                for start, middle, _ in training
            ] + [
                values[end] - values[middle]
                for _, middle, end in training
            ]
            mean_step = float(np.mean(training_steps))
            if abs(mean_step) < 0.035:
                continue
            if any(step * mean_step <= 0 for step in training_steps):
                continue
            target_step = values[target_pair[1]] - values[target_pair[0]]
            if target_step * mean_step <= 0:
                continue
            consistency_scale = max(abs(mean_step) * 0.45, 0.018)
            consistency = float(
                np.mean(
                    [
                        _soft_similarity(step, mean_step, consistency_scale)
                        for step in training_steps + [target_step]
                    ]
                )
            )
            reliability = consistency * 0.98
            if reliability < 0.72:
                continue
            predicted = values[target_pair[1]] + mean_step
            candidate_scale = max(abs(mean_step) * 0.35, 0.022)
            trend_name = "increase" if mean_step > 0 else "decrease"
            matches.append(
                RuleMatch(
                    name=f"{feature_name}:monotonic_{trend_name}",
                    direction=direction,
                    reliability=reliability,
                    candidate_scores=tuple(
                        _soft_similarity(values[index], predicted, candidate_scale)
                        for index in range(8, 16)
                    ),
                )
            )
    return matches


def _three_level_labels(values: list[float]) -> list[int] | None:
    data = np.asarray(values, dtype=np.float64)
    if data.size < 3 or float(np.ptp(data)) < 1e-4:
        return None
    centers = np.quantile(data, [0.1, 0.5, 0.9])
    for _ in range(12):
        labels = np.argmin(np.abs(data[:, None] - centers[None, :]), axis=1)
        updated = np.asarray(
            [data[labels == index].mean() if np.any(labels == index) else centers[index] for index in range(3)]
        )
        if np.allclose(updated, centers):
            break
        centers = updated
    if len(set(int(item) for item in labels)) != 3:
        return None
    order = np.argsort(centers)
    remap = {int(original): level for level, original in enumerate(order)}
    return [remap[int(label)] for label in labels]


def _induce_relative_size_rules(observations: list[PanelObservation]) -> list[RuleMatch]:
    """Compare small/medium/large levels despite context/candidate crop-scale changes."""
    feature_maps = [observation.features.vector() for observation in observations]
    matches: list[RuleMatch] = []
    for feature_name in ("width_ratio", "height_ratio"):
        values = [features[feature_name] for features in feature_maps]
        context_labels = _three_level_labels(values[:8])
        candidate_labels = _three_level_labels(values[8:16])
        if context_labels is None or candidate_labels is None:
            continue
        rows = (context_labels[0:3], context_labels[3:6], context_labels[6:8])
        for shift_name, expected_second, expected_third in (
            ("cyclic_right", [rows[0][2], rows[0][0], rows[0][1]], [rows[0][1], rows[0][2], rows[0][0]]),
            ("cyclic_left", [rows[0][1], rows[0][2], rows[0][0]], [rows[0][2], rows[0][0], rows[0][1]]),
        ):
            checks = [rows[1][index] == expected_second[index] for index in range(3)] + [
                rows[2][index] == expected_third[index] for index in range(2)
            ]
            reliability = float(np.mean(checks)) * 0.98
            if reliability < 0.78:
                continue
            predicted = expected_third[2]
            matches.append(
                RuleMatch(
                    name=f"relative_{feature_name.removesuffix('_ratio')}_level:{shift_name}",
                    direction="row",
                    reliability=reliability,
                    candidate_scores=tuple(exp(-1.7 * abs(level - predicted)) for level in candidate_labels),
                )
            )
    return matches


def _align_mask(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    if reference.shape == candidate.shape:
        return candidate
    return cv2.resize(candidate.astype(np.uint8), (reference.shape[1], reference.shape[0])) > 0


def _induce_mask_rules(observations: list[PanelObservation]) -> list[RuleMatch]:
    masks = [observation.mask for observation in observations]
    matches: list[RuleMatch] = []
    for direction, training, target_pair in _direction_layouts():
        for operator_name, operator, complexity_prior in _MASK_OPERATORS:
            training_scores = []
            try:
                for left, right, end in training:
                    predicted = operator(masks[left], masks[right])
                    predicted = _align_mask(masks[end], predicted)
                    training_scores.append(_iou(masks[end], predicted))
            except (ValueError, cv2.error):
                continue
            reliability = float(np.mean(training_scores)) * complexity_prior
            if reliability < 0.52:
                continue
            predicted = operator(masks[target_pair[0]], masks[target_pair[1]])
            candidate_scores = tuple(_iou(mask, _align_mask(mask, predicted)) for mask in masks[8:16])
            matches.append(
                RuleMatch(
                    name=operator_name,
                    direction=direction,
                    reliability=reliability,
                    candidate_scores=candidate_scores,
                )
            )
    return matches


def induce_rules(observations: list[PanelObservation]) -> RuleInductionResult:
    if len(observations) != 16:
        raise ValueError(f"Rule induction requires 16 observations, got {len(observations)}")

    matches = (
        _induce_scalar_rules(observations)
        + _induce_cyclic_rules(observations)
        + _induce_distribute_three_rules(observations)
        + _induce_monotonic_size_rules(observations)
        + _induce_relative_size_rules(observations)
        + _induce_mask_rules(observations)
    )
    if not matches:
        return RuleInductionResult(scores=[0.125] * 8, confidence=0.0, matches=[])

    # Avoid counting ten near-identical operators for one measurement as ten
    # independent votes. Keep the strongest explanation per feature/direction.
    selected_matches: list[RuleMatch] = []
    best_scalar: dict[tuple[str, str], RuleMatch] = {}
    for match in matches:
        if ":" not in match.name:
            selected_matches.append(match)
            continue
        feature_name = match.name.split(":", 1)[0]
        key = (feature_name, match.direction)
        if key not in best_scalar or match.reliability > best_scalar[key].reliability:
            best_scalar[key] = match
    selected_matches.extend(best_scalar.values())

    weighted_scores = np.zeros(8, dtype=np.float64)
    total_weight = 0.0
    for match in selected_matches:
        weight = max(match.reliability - 0.45, 0.0) ** 2
        feature_name = match.name.split(":", 1)[0]
        weight *= _FEATURE_PRIORS.get(feature_name, 1.0)
        if match.name.startswith("pixel_") or match.name.startswith(("rotate_", "flip_")):
            weight *= 2.1
        weighted_scores += weight * np.asarray(match.candidate_scores, dtype=np.float64)
        total_weight += weight
    if total_weight <= 0:
        return RuleInductionResult(scores=[0.125] * 8, confidence=0.0, matches=matches)

    scores = weighted_scores / total_weight
    order = np.argsort(scores)[::-1]
    margin = float(scores[order[0]] - scores[order[1]])
    reliable = sorted((match.reliability for match in selected_matches), reverse=True)[:8]
    rule_quality = float(np.mean(reliable)) if reliable else 0.0
    confidence = float(np.clip(0.55 * rule_quality + 1.8 * margin, 0.0, 1.0))
    return RuleInductionResult(scores=scores.tolist(), confidence=confidence, matches=matches)
