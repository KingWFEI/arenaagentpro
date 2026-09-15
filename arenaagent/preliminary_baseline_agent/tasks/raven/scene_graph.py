from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass(slots=True, frozen=True)
class SceneNode:
    node_id: int
    area_ratio: float
    centroid_x: float
    centroid_y: float
    width_ratio: float
    height_ratio: float
    holes: int
    polygon_vertices: int
    mean_darkness: float


@dataclass(slots=True, frozen=True)
class SceneEdge:
    source: int
    target: int
    relation: str


@dataclass(slots=True)
class SceneGraph:
    """Small geometric graph used by deterministic rules and text reasoning."""

    nodes: list[SceneNode] = field(default_factory=list)
    edges: list[SceneEdge] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        relation_counts: dict[str, int] = {}
        for edge in self.edges:
            relation_counts[edge.relation] = relation_counts.get(edge.relation, 0) + 1
        return {
            "node_count": len(self.nodes),
            "nodes": [asdict(node) for node in self.nodes[:12]],
            "relation_counts": relation_counts,
        }


def build_scene_graph(
    mask: np.ndarray,
    min_area_ratio: float = 0.001,
    gray: np.ndarray | None = None,
) -> SceneGraph:
    height, width = mask.shape[:2]
    image_area = max(height * width, 1)
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    nodes: list[SceneNode] = []

    for component_id in range(1, component_count):
        _, _, box_width, box_height, area = stats[component_id]
        if area / image_area < min_area_ratio:
            continue
        component = np.where(labels == component_id, 255, 0).astype(np.uint8)
        contours, hierarchy = cv2.findContours(component, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = 0
        if hierarchy is not None:
            holes = sum(1 for item in hierarchy[0] if int(item[3]) >= 0)
        external = max(contours, key=cv2.contourArea) if contours else None
        if external is not None:
            perimeter = cv2.arcLength(external, True)
            polygon_vertices = int(len(cv2.approxPolyDP(external, max(1.5, 0.035 * perimeter), True)))
            interior = np.zeros_like(mask)
            cv2.drawContours(interior, [external], -1, 255, thickness=cv2.FILLED)
            region = interior > 0
            mean_darkness = (
                float(((255.0 - gray[region]) / 255.0).mean())
                if gray is not None and np.any(region)
                else 0.0
            )
        else:
            polygon_vertices = 0
            mean_darkness = 0.0
        nodes.append(
            SceneNode(
                node_id=len(nodes),
                area_ratio=round(float(area / image_area), 5),
                centroid_x=round(float(centroids[component_id][0] / max(width - 1, 1)), 4),
                centroid_y=round(float(centroids[component_id][1] / max(height - 1, 1)), 4),
                width_ratio=round(float(box_width / width), 4),
                height_ratio=round(float(box_height / height), 4),
                holes=holes,
                polygon_vertices=polygon_vertices,
                mean_darkness=round(mean_darkness, 5),
            )
        )

    edges: list[SceneEdge] = []
    for source in nodes:
        for target in nodes:
            if source.node_id == target.node_id:
                continue
            if source.centroid_x + 0.05 < target.centroid_x:
                edges.append(SceneEdge(source.node_id, target.node_id, "left_of"))
            if source.centroid_y + 0.05 < target.centroid_y:
                edges.append(SceneEdge(source.node_id, target.node_id, "above"))
            distance = float(
                np.hypot(source.centroid_x - target.centroid_x, source.centroid_y - target.centroid_y)
            )
            if distance < 0.28:
                edges.append(SceneEdge(source.node_id, target.node_id, "near"))
    return SceneGraph(nodes=nodes, edges=edges)
