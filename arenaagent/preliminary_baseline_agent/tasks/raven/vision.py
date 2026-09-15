from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from arenaagent.preliminary_baseline_agent.tasks.raven.scene_graph import SceneGraph, build_scene_graph


@dataclass(slots=True, frozen=True)
class PanelFeatures:
    ink_ratio: float
    component_count: float
    hole_count: float
    centroid_x: float
    centroid_y: float
    width_ratio: float
    height_ratio: float
    polygon_vertex_sum: float
    polygon_vertex_mean: float
    fill_density: float
    mean_darkness: float
    fill_category: float
    outline_ratio: float
    object_area_mean: float
    object_area_cv: float
    object_width_mean: float
    object_width_cv: float
    top_vertex_count: float
    bottom_vertex_count: float
    top_outline: float
    bottom_outline: float
    top_darkness: float
    bottom_darkness: float
    top_width: float
    bottom_width: float
    horizontal_symmetry: float
    vertical_symmetry: float
    orientation_sin: float
    orientation_cos: float
    quadrant_nw: float
    quadrant_ne: float
    quadrant_sw: float
    quadrant_se: float

    def vector(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(slots=True)
class PanelObservation:
    mask: np.ndarray
    features: PanelFeatures
    graph: SceneGraph

    def summary(self) -> dict[str, Any]:
        return {"features": self.features.vector(), "scene_graph": self.graph.summary()}


def normalize_panel(image: Image.Image, size: int = 128) -> np.ndarray:
    gray = np.asarray(image.convert("L").resize((size, size), Image.Resampling.LANCZOS), dtype=np.uint8)
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    background = float(np.median(border))
    difference = cv2.absdiff(gray, np.full_like(gray, int(round(background))))
    otsu_threshold, _ = cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(8.0, float(otsu_threshold) * 0.75)
    mask = np.where(difference >= threshold, 255, 0).astype(np.uint8)
    kernel = np.ones((2, 2), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Some competition panels contain a layout divider (not a puzzle object).
    # Remove near-full-width/height straight lines before counting components.
    binary_mask = mask > 0

    def longest_run(values: np.ndarray) -> int:
        padded = np.pad(values.astype(np.int8), (1, 1))
        changes = np.flatnonzero(np.diff(padded))
        return int(np.max(changes[1::2] - changes[::2], initial=0))

    # A sum-only test mistakes two aligned filled objects for a divider.  Layout
    # lines are continuous; require a long uninterrupted run before removal.
    horizontal_lines = [
        row for row in range(size) if longest_run(binary_mask[row, :]) >= int(size * 0.72)
    ]
    vertical_lines = [
        column for column in range(size) if longest_run(binary_mask[:, column]) >= int(size * 0.72)
    ]
    for row in horizontal_lines:
        mask[max(0, row - 1) : min(size, row + 2), :] = 0
    for column in vertical_lines:
        mask[:, max(0, column - 1) : min(size, column + 2)] = 0
    return mask


def _symmetry(mask: np.ndarray, axis: int) -> float:
    flipped = np.flip(mask, axis=axis)
    union = np.logical_or(mask > 0, flipped > 0).sum()
    if union == 0:
        return 1.0
    disagreement = np.logical_xor(mask > 0, flipped > 0).sum()
    return float(1.0 - disagreement / union)


def _quadrants(mask: np.ndarray) -> tuple[float, float, float, float]:
    binary = mask > 0
    height, width = binary.shape
    mid_y, mid_x = height // 2, width // 2
    total = max(int(binary.sum()), 1)
    return tuple(
        float(region.sum() / total)
        for region in (
            binary[:mid_y, :mid_x],
            binary[:mid_y, mid_x:],
            binary[mid_y:, :mid_x],
            binary[mid_y:, mid_x:],
        )
    )


def _shape_statistics(image: Image.Image, mask: np.ndarray) -> tuple[float, float, float, float]:
    """Measure polygon complexity and fill independently of object size."""
    gray = np.asarray(
        image.convert("L").resize((mask.shape[1], mask.shape[0]), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = mask.shape[0] * mask.shape[1] * 0.002
    contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    if not contours:
        return 0.0, 0.0, 0.0, 0.0

    vertices: list[int] = []
    fill_values: list[float] = []
    darkness_values: list[float] = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, max(1.5, 0.035 * perimeter), True)
        vertices.append(int(len(approximation)))

        interior = np.zeros_like(mask)
        cv2.drawContours(interior, [contour], -1, 255, thickness=cv2.FILLED)
        region = interior > 0
        region_area = max(int(region.sum()), 1)
        fill_values.append(float(np.logical_and(mask > 0, region).sum() / region_area))
        darkness_values.append(float(((255.0 - gray[region]) / 255.0).mean()))

    return (
        float(sum(vertices)),
        float(np.mean(vertices)),
        float(np.mean(fill_values)),
        float(np.mean(darkness_values)),
    )


def extract_panel_observation(image: Image.Image) -> PanelObservation:
    mask = normalize_panel(image)
    binary = mask > 0
    height, width = mask.shape
    image_area = max(height * width, 1)
    ys, xs = np.nonzero(binary)

    if len(xs):
        centroid_x = float(xs.mean() / max(width - 1, 1))
        centroid_y = float(ys.mean() / max(height - 1, 1))
        width_ratio = float((xs.max() - xs.min() + 1) / width)
        height_ratio = float((ys.max() - ys.min() + 1) / height)
    else:
        centroid_x = centroid_y = 0.5
        width_ratio = height_ratio = 0.0

    moments = cv2.moments(mask, binaryImage=True)
    if abs(moments["mu20"] - moments["mu02"]) + abs(moments["mu11"]) > 1e-8:
        angle = 0.5 * np.arctan2(2.0 * moments["mu11"], moments["mu20"] - moments["mu02"])
        orientation_sin = float(np.sin(2.0 * angle))
        orientation_cos = float(np.cos(2.0 * angle))
    else:
        orientation_sin, orientation_cos = 0.0, 1.0

    gray = np.asarray(
        image.convert("L").resize((width, height), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
    graph = build_scene_graph(mask, gray=gray)
    object_areas = np.asarray([node.area_ratio for node in graph.nodes], dtype=np.float64)
    object_widths = np.asarray([node.width_ratio for node in graph.nodes], dtype=np.float64)
    object_area_mean = float(object_areas.mean()) if object_areas.size else 0.0
    object_width_mean = float(object_widths.mean()) if object_widths.size else 0.0
    object_area_cv = (
        float(object_areas.std() / max(object_area_mean, 1e-6)) if object_areas.size > 1 else 0.0
    )
    object_width_cv = (
        float(object_widths.std() / max(object_width_mean, 1e-6)) if object_widths.size > 1 else 0.0
    )
    outline_ratio = (
        float(sum(node.holes > 0 for node in graph.nodes) / len(graph.nodes)) if graph.nodes else 0.0
    )
    vertical_nodes = sorted(graph.nodes, key=lambda node: node.centroid_y)
    if len(vertical_nodes) >= 2:
        top_node, bottom_node = vertical_nodes[0], vertical_nodes[-1]
        top_vertex_count = float(top_node.polygon_vertices)
        bottom_vertex_count = float(bottom_node.polygon_vertices)
        top_outline = float(top_node.holes > 0)
        bottom_outline = float(bottom_node.holes > 0)
        top_darkness = float(top_node.mean_darkness)
        bottom_darkness = float(bottom_node.mean_darkness)
        top_width = float(top_node.width_ratio)
        bottom_width = float(bottom_node.width_ratio)
    else:
        # Zero marks the attribute as not applicable to singleton layouts.
        top_vertex_count = bottom_vertex_count = 0.0
        top_outline = bottom_outline = 0.0
        top_darkness = bottom_darkness = 0.0
        top_width = bottom_width = 0.0
    quadrant_nw, quadrant_ne, quadrant_sw, quadrant_se = _quadrants(mask)
    polygon_vertex_sum, polygon_vertex_mean, fill_density, mean_darkness = _shape_statistics(image, mask)
    if fill_density < 0.78:
        fill_category = 0.0  # outline
    elif mean_darkness < 0.44:
        fill_category = 1.0  # light/medium fill
    else:
        fill_category = 2.0  # dark fill
    features = PanelFeatures(
        ink_ratio=float(binary.sum() / image_area),
        component_count=float(len(graph.nodes)),
        hole_count=float(sum(node.holes for node in graph.nodes)),
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        width_ratio=width_ratio,
        height_ratio=height_ratio,
        polygon_vertex_sum=polygon_vertex_sum,
        polygon_vertex_mean=polygon_vertex_mean,
        fill_density=fill_density,
        mean_darkness=mean_darkness,
        fill_category=fill_category,
        outline_ratio=outline_ratio,
        object_area_mean=object_area_mean,
        object_area_cv=object_area_cv,
        object_width_mean=object_width_mean,
        object_width_cv=object_width_cv,
        top_vertex_count=top_vertex_count,
        bottom_vertex_count=bottom_vertex_count,
        top_outline=top_outline,
        bottom_outline=bottom_outline,
        top_darkness=top_darkness,
        bottom_darkness=bottom_darkness,
        top_width=top_width,
        bottom_width=bottom_width,
        horizontal_symmetry=_symmetry(mask, axis=0),
        vertical_symmetry=_symmetry(mask, axis=1),
        orientation_sin=orientation_sin,
        orientation_cos=orientation_cos,
        quadrant_nw=quadrant_nw,
        quadrant_ne=quadrant_ne,
        quadrant_sw=quadrant_sw,
        quadrant_se=quadrant_se,
    )
    return PanelObservation(mask=mask, features=features, graph=graph)


def extract_question_observations(images: list[Image.Image]) -> list[PanelObservation]:
    if len(images) != 16:
        raise ValueError(f"A Raven question must contain 16 panels, got {len(images)}")
    return [extract_panel_observation(image) for image in images]
