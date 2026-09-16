from __future__ import annotations

import base64
import itertools
import json
from dataclasses import dataclass
from statistics import median
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class JigsawCell:
    name: str
    row: int
    column: int
    location: tuple[float, float, float]


@dataclass(frozen=True)
class JigsawLayout:
    candidates: tuple[dict[str, Any], ...]
    placed: tuple[dict[str, Any], ...]
    empty_cells: tuple[JigsawCell, ...]
    target_rotation: dict[str, float]


def _location(obj: dict[str, Any]) -> tuple[float, float, float] | None:
    loc = obj.get("place_location")
    if not isinstance(loc, dict):
        return None
    try:
        return (float(loc["X"]), float(loc["Y"]), float(loc["Z"]))
    except (KeyError, TypeError, ValueError):
        return None


def _is_tile_sized(obj: dict[str, Any]) -> bool:
    aabb = obj.get("world_aabb")
    if not isinstance(aabb, dict):
        return False
    low, high = aabb.get("min"), aabb.get("max")
    if not isinstance(low, dict) or not isinstance(high, dict):
        return False
    try:
        sizes = [abs(float(high[axis]) - float(low[axis])) for axis in ("x", "y", "z")]
    except (KeyError, TypeError, ValueError):
        return False
    return sizes[0] <= 3.0 and 7.0 <= sizes[1] <= 16.0 and 7.0 <= sizes[2] <= 16.0


def _cluster(values: list[float], tolerance: float = 1.5) -> list[float]:
    clusters: list[list[float]] = []
    for value in sorted(values):
        if not clusters or abs(value - sum(clusters[-1]) / len(clusters[-1])) > tolerance:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return [sum(group) / len(group) for group in clusters]


def _angular_distance(left: float, right: float) -> float:
    """Smallest distance between two Euler components, accounting for wraparound."""
    return abs((left - right + 180.0) % 360.0 - 180.0)


def _representative_rotation(rotations: list[dict[str, Any]]) -> dict[str, float]:
    """Choose an observed board-tile rotation instead of mixing Euler components."""
    axes = ("roll", "pitch", "yaw")
    observed = [
        {axis: float(rotation.get(axis, 0.0)) for axis in axes}
        for rotation in rotations
    ]
    if not observed:
        return {axis: 0.0 for axis in axes}
    return min(
        observed,
        key=lambda candidate: (
            sum(
                _angular_distance(candidate[axis], other[axis])
                for other in observed
                for axis in axes
            ),
            # Break even-sized cluster ties in favour of the configured board
            # orientation. This matters when three UE Euler decompositions use
            # pitch +/-90 while the other three use yaw +90.
            _angular_distance(candidate["roll"], 0.0)
            + _angular_distance(candidate["pitch"], 0.0)
            + _angular_distance(candidate["yaw"], 90.0),
        ),
    )


def rotation_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Circular L1 distance between two roll/pitch/yaw representations."""
    return sum(
        _angular_distance(float(left.get(axis, 0.0)), float(right.get(axis, 0.0)))
        for axis in ("roll", "pitch", "yaw")
    )


def infer_layout(objects: list[dict[str, Any]], reference_bounding: list[Any]) -> JigsawLayout:
    """Infer candidates and empty cells from world geometry, never from model-generated IDs."""
    if len(reference_bounding) < 4:
        raise ValueError("jigsaw reference_bounding is missing")
    y_min, z_max, y_max, z_min = map(float, reference_bounding[:4])

    tiles = [obj for obj in objects if _location(obj) is not None and _is_tile_sized(obj)]
    placed = []
    for obj in tiles:
        _, y, z = _location(obj)  # type: ignore[misc]
        if y_min <= y <= y_max and z_min <= z <= z_max:
            placed.append(obj)
    if len(placed) != 6:
        raise ValueError(f"expected 6 placed jigsaw tiles, found {len(placed)}")

    board_x = float(median(_location(obj)[0] for obj in placed))  # type: ignore[index]
    y_axes = _cluster([_location(obj)[1] for obj in placed])  # type: ignore[index]
    z_axes = list(reversed(_cluster([_location(obj)[2] for obj in placed])))  # type: ignore[index]
    if len(y_axes) != 3 or len(z_axes) != 3:
        raise ValueError(f"could not infer 3x3 lattice: y={y_axes}, z={z_axes}")

    occupied: set[tuple[int, int]] = set()
    for obj in placed:
        _, y, z = _location(obj)  # type: ignore[misc]
        column = min(range(3), key=lambda index: abs(y_axes[index] - y))
        row = min(range(3), key=lambda index: abs(z_axes[index] - z))
        occupied.add((row, column))

    names = (
        ("top-left", "top-middle", "top-right"),
        ("middle-left", "center", "middle-right"),
        ("bottom-left", "bottom-middle", "bottom-right"),
    )
    empty_cells = tuple(
        JigsawCell(names[row][column], row, column, (board_x, y_axes[column], z_axes[row]))
        for row in range(3)
        for column in range(3)
        if (row, column) not in occupied
    )
    if len(empty_cells) != 3:
        raise ValueError(f"expected 3 empty cells, found {len(empty_cells)}")

    candidates = []
    for obj in tiles:
        x, y, z = _location(obj)  # type: ignore[misc]
        near_board = abs(x - board_x) <= 3.0 and min(z_axes) - 8.0 <= z <= max(z_axes) + 8.0
        if near_board and y > y_max + 5.0:
            candidates.append(obj)
    candidates.sort(key=lambda obj: _location(obj)[2], reverse=True)  # type: ignore[index]
    if len(candidates) != 3:
        raise ValueError(f"expected 3 loose candidate tiles, found {len(candidates)}")

    rotations = [obj.get("rotation", {}) for obj in placed]
    target_rotation = _representative_rotation(rotations)
    return JigsawLayout(
        candidates=tuple(candidates),
        placed=tuple(placed),
        empty_cells=empty_cells,
        target_rotation=target_rotation,
    )


def _decode_combined_image(image_b64: str) -> np.ndarray:
    payload = image_b64.split(",", 1)[1] if image_b64.startswith("data:image") else image_b64
    image = cv2.imdecode(np.frombuffer(base64.b64decode(payload), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[1] < 2:
        raise ValueError("could not decode jigsaw perception image")
    return image


def _candidate_crop(clean: np.ndarray, candidate: dict[str, Any], wide: bool = False) -> np.ndarray:
    _, world_y, world_z = _location(candidate)  # type: ignore[misc]
    height, width = clean.shape[:2]
    center_x = int(round(((1.8426 * world_y - 38.58) / 640.0) * width))
    center_y = int(round(((-2.1818 * world_z + 571.0) / 720.0) * height))
    half_width = max(2, int(round((10.5 if wide else 10.0) * width / 640.0)))
    half_height = max(2, int(round((12.0 if wide else 11.0) * height / 720.0)))
    return clean[
        max(0, center_y - half_height) : min(height, center_y + half_height),
        max(0, center_x - half_width) : min(width, center_x + half_width),
    ]


def _reference_crop(clean: np.ndarray) -> np.ndarray:
    height, width = clean.shape[:2]
    x1, x2 = int(round(width * 99 / 640)), int(round(width * 221 / 640))
    y1, y2 = int(round(height * 320 / 720)), int(round(height * 432 / 720))
    crop = clean[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("reference crop is empty")
    return crop


def build_vlm_montage(image_b64: str, layout: JigsawLayout) -> str:
    """Enlarge the tiny pieces and label the 3x3 reference cells for a focused VLM call."""
    combined = _decode_combined_image(image_b64)
    clean = combined[:, : combined.shape[1] // 2]
    reference = cv2.resize(_reference_crop(clean), (488, 448), interpolation=cv2.INTER_CUBIC)
    for x in (163, 325):
        cv2.line(reference, (x, 0), (x, 448), (0, 0, 255), 3)
    for y in (149, 299):
        cv2.line(reference, (0, y), (488, y), (0, 0, 255), 3)

    canvas = np.full((720, 1000, 3), 245, dtype=np.uint8)
    canvas[80:528, 30:518] = reference
    cv2.putText(canvas, "COMPLETE REFERENCE: 3x3 CELLS", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cell_names = (
        "top-left", "top-middle", "top-right",
        "middle-left", "center", "middle-right",
        "bottom-left", "bottom-middle", "bottom-right",
    )
    for index, name in enumerate(cell_names):
        row, column = divmod(index, 3)
        cv2.putText(
            canvas,
            name,
            (35 + column * 163, 100 + row * 149),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 255),
            1,
        )

    for index, candidate in enumerate(layout.candidates):
        tile = _candidate_crop(clean, candidate)
        if tile.size == 0:
            raise ValueError("candidate crop is empty")
        tile = cv2.resize(tile, (180, 198), interpolation=cv2.INTER_CUBIC)
        y = 60 + index * 220
        canvas[y : y + 198, 650:830] = tile
        object_id = str(candidate["object_id"])
        cv2.putText(canvas, f"CANDIDATE ID {object_id}", (620, y + 218), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

    ok, encoded = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise ValueError("could not encode jigsaw montage")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def build_mapping_prompt(layout: JigsawLayout) -> str:
    candidate_ids = [str(candidate["object_id"]) for candidate in layout.candidates]
    cells = [
        {"cell": cell.name, "world_center": list(cell.location)}
        for cell in layout.empty_cells
    ]
    return (
        "Solve the 3x3 image jigsaw shown in the diagnostic image. The complete reference is enlarged on the left "
        "and divided by red lines; the three loose candidate tiles are enlarged and labelled on the right. "
        "Match visual content, contours, colors, and continuation across cell boundaries. Do not assume candidate "
        "vertical order equals grid row order. Assign each candidate and each empty cell exactly once.\n"
        f"Allowed candidate IDs: {json.dumps(candidate_ids)}\n"
        f"Allowed empty cells: {json.dumps(cells)}\n"
        "Return JSON only, without a markdown fence, in this schema: "
        '{"placements":[{"object_id":"<allowed-id>","cell":"<allowed-cell>"}]}'
    )


def parse_mapping(response_text: str, layout: JigsawLayout) -> list[dict[str, str]]:
    text = response_text.strip()
    if text.startswith("```"):
        text = text.strip("` \n")
        if "\n" in text and not text.startswith(("{", "[")):
            text = text.split("\n", 1)[1]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("VLM mapping response contains no JSON object")
    parsed = json.loads(text[start : end + 1])
    placements = parsed.get("placements")
    if not isinstance(placements, list) or len(placements) != len(layout.candidates):
        raise ValueError("VLM mapping response has the wrong number of placements")

    allowed_ids = {str(candidate["object_id"]) for candidate in layout.candidates}
    allowed_cells = {cell.name for cell in layout.empty_cells}
    normalized = [
        {"object_id": str(placement.get("object_id")), "cell": str(placement.get("cell"))}
        for placement in placements
        if isinstance(placement, dict)
    ]
    if {item["object_id"] for item in normalized} != allowed_ids:
        raise ValueError("VLM mapping did not use every allowed candidate exactly once")
    if {item["cell"] for item in normalized} != allowed_cells:
        raise ValueError("VLM mapping did not use every empty cell exactly once")
    return normalized


def solve_by_image_cost(image_b64: str, layout: JigsawLayout) -> list[dict[str, str]]:
    """Offline fallback: globally minimize LAB/edge mismatch over all one-to-one assignments."""
    combined = _decode_combined_image(image_b64)
    clean = combined[:, : combined.shape[1] // 2]
    reference = _reference_crop(clean)
    x_edges = np.rint(np.linspace(0, reference.shape[1], 4)).astype(int)
    y_edges = np.rint(np.linspace(0, reference.shape[0], 4)).astype(int)
    candidates = list(layout.candidates)
    cells = list(layout.empty_cells)
    costs = np.zeros((len(candidates), len(cells)), dtype=np.float64)

    for candidate_index, candidate in enumerate(candidates):
        tile = _candidate_crop(clean, candidate, wide=True)
        for cell_index, cell in enumerate(cells):
            target = reference[
                y_edges[cell.row] : y_edges[cell.row + 1],
                x_edges[cell.column] : x_edges[cell.column + 1],
            ]
            resized = cv2.resize(tile, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_CUBIC)
            tile_lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB).astype(np.float32)
            target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
            color_cost = np.mean(
                (cv2.GaussianBlur(tile_lab, (5, 5), 0) - cv2.GaussianBlur(target_lab, (5, 5), 0)) ** 2
            )
            tile_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            edge_cost = np.mean(
                (cv2.Sobel(tile_gray, cv2.CV_32F, 1, 1) - cv2.Sobel(target_gray, cv2.CV_32F, 1, 1)) ** 2
            )
            costs[candidate_index, cell_index] = float(color_cost + 0.2 * edge_cost)

    assignment = min(
        itertools.permutations(range(len(cells))),
        key=lambda permutation: sum(costs[index, permutation[index]] for index in range(len(candidates))),
    )
    return [
        {"object_id": str(candidate["object_id"]), "cell": cells[assignment[index]].name}
        for index, candidate in enumerate(candidates)
    ]

