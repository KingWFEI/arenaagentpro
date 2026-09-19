from __future__ import annotations

import base64
import unittest

import cv2
import numpy as np

from arenaagent.preliminary_baseline_agent.tasks.jigsaw.solver import (
    build_vlm_montage,
    infer_layout,
    parse_mapping,
    solve_by_image_cost,
)


def _tile(object_id: str, y: float, z: float) -> dict:
    return {
        "object_id": object_id,
        "place_location": {"X": 837.0, "Y": y, "Z": z},
        "rotation": {"roll": 0.0, "pitch": 0.0, "yaw": 131.0},
        "world_aabb": {
            "min": {"x": 836.6, "y": y - 5.2, "z": z - 5.2},
            "max": {"x": 837.4, "y": y + 5.2, "z": z + 5.2},
        },
    }


def _layout():
    objects = [
        _tile("7", 155, 110),
        _tile("9", 177, 110),
        _tile("11", 166, 99),
        _tile("12", 177, 99),
        _tile("13", 155, 88),
        _tile("15", 177, 88),
        _tile("14", 209, 110),
        _tile("10", 209, 96),
        _tile("8", 209, 82),
    ]
    return infer_layout(objects, [152, 113, 180, 85])


class JigsawSolverTests(unittest.TestCase):
    def test_geometry_finds_only_loose_tiles_and_empty_cells(self) -> None:
        layout = _layout()
        self.assertEqual([item["object_id"] for item in layout.candidates], ["14", "10", "8"])
        self.assertEqual(
            [cell.name for cell in layout.empty_cells],
            ["top-middle", "middle-left", "bottom-middle"],
        )
        self.assertEqual(layout.target_rotation["yaw"], 131.0)

    def test_mapping_parser_enforces_bijection(self) -> None:
        layout = _layout()
        valid = '{"placements":[{"object_id":"8","cell":"top-middle"},'
        valid += '{"object_id":"10","cell":"middle-left"},'
        valid += '{"object_id":"14","cell":"bottom-middle"}]}'
        self.assertEqual(len(parse_mapping(valid, layout)), 3)
        with self.assertRaises(ValueError):
            parse_mapping(valid.replace('"14"', '"8"'), layout)

    def test_offline_global_assignment_and_montage(self) -> None:
        layout = _layout()
        image = np.full((720, 1280, 3), 210, dtype=np.uint8)
        reference = image[320:432, 99:221]
        x_edges = np.rint(np.linspace(0, reference.shape[1], 4)).astype(int)
        y_edges = np.rint(np.linspace(0, reference.shape[0], 4)).astype(int)
        patterns = {
            (0, 1): np.array([[[10, 30, 220], [240, 240, 240]], [[220, 30, 10], [20, 20, 20]]], dtype=np.uint8),
            (1, 0): np.array([[[10, 210, 30], [10, 210, 30]], [[230, 20, 210], [230, 20, 210]]], dtype=np.uint8),
            (2, 1): np.array([[[230, 220, 20], [20, 220, 230]], [[230, 220, 20], [20, 220, 230]]], dtype=np.uint8),
        }
        mapping = {"14": (2, 1), "10": (1, 0), "8": (0, 1)}
        for cell, pattern in patterns.items():
            row, column = cell
            target = reference[y_edges[row] : y_edges[row + 1], x_edges[column] : x_edges[column + 1]]
            target[:] = cv2.resize(pattern, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_NEAREST)
        for candidate in layout.candidates:
            _, world_y, world_z = (
                candidate["place_location"]["X"],
                candidate["place_location"]["Y"],
                candidate["place_location"]["Z"],
            )
            center_x = int(round(1.8426 * world_y - 38.58))
            center_y = int(round(-2.1818 * world_z + 571.0))
            patch = cv2.resize(patterns[mapping[candidate["object_id"]]], (20, 24), interpolation=cv2.INTER_NEAREST)
            image[center_y - 12 : center_y + 12, center_x - 10 : center_x + 10] = patch

        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        result = solve_by_image_cost(image_b64, layout)
        self.assertEqual({item["object_id"]: item["cell"] for item in result}, {
            "14": "bottom-middle",
            "10": "middle-left",
            "8": "top-middle",
        })
        montage_url = build_vlm_montage(image_b64, layout)
        self.assertTrue(montage_url.startswith("data:image/jpeg;base64,"))


if __name__ == "__main__":
    unittest.main()
