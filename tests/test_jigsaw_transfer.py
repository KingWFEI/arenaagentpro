from __future__ import annotations

import base64
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from arenaagent.preliminary_baseline_agent.preliminary_baseline_agent import PreliminaryBaselineAgent
from arenaagent.preliminary_baseline_agent.tasks.jigsaw.runner import run_dedicated_jigsaw
from arenaagent.preliminary_baseline_agent.tasks.jigsaw.solver import infer_layout, solve_by_image_cost


def _tile(mapped_id: int, y: float, z: float) -> dict:
    return {
        "object_id": str(mapped_id),
        "place_location": {"X": 837.0, "Y": y, "Z": z},
        "world_aabb": {
            "min": {"x": 836.5, "y": y - 5, "z": z - 5},
            "max": {"x": 837.5, "y": y + 5, "z": z + 5},
        },
    }


class FakeMapper:
    def get_raw_id(self, mapped_id: int) -> str:
        return f"piece-{mapped_id}"


class FakeTongSim:
    def __init__(self, objects: list[dict], *, fail_second_hand: bool = False) -> None:
        self.locations = {f"piece-{obj['object_id']}": dict(obj["place_location"]) for obj in objects}
        self.calls: list[tuple] = []
        self.hands: dict[int, str] = {}
        self.fail_second_hand = fail_second_hand

    def get_object_basic_info(self, object_id: str) -> dict:
        return {"rotation": {"roll": 0, "pitch": 0, "yaw": 0}, "place_location": self.locations[object_id]}

    def move_to_object(self, character_id: str, object_id: str) -> dict:
        self.calls.append(("approach", object_id))
        return {"result": "success"}

    def move_and_take_object(self, character_id: str, object_id: str, which_hand: int = 0) -> dict:
        self.calls.append(("take", object_id, which_hand))
        if which_hand == 1 and self.fail_second_hand:
            return {"result": "failed"}
        self.hands[which_hand] = object_id
        return {"result": "success"}

    def get_object_in_hand(self, character_id: str):
        for hand in (0, 1):
            if hand in self.hands:
                return (self.hands[hand], hand)
        return None

    def move_to_location(self, character_id: str, target: list[float], stop_distance: float = 0.5) -> dict:
        self.calls.append(("approach_cell", tuple(target), stop_distance))
        return {"result": "success"}

    def put_down_to_location(self, character_id: str, target_location: list[float], **kwargs) -> dict:
        hand = kwargs["which_hand"]
        held = self.hands.pop(hand)
        self.calls.append(("place", held, tuple(target_location), kwargs))
        self.locations[held] = dict(zip(("X", "Y", "Z"), target_location))
        return {"result": "success"}


class FakeLegacyTongSim(FakeTongSim):
    def __init__(self, objects: list[dict]) -> None:
        super().__init__(objects)
        self.objects = objects
        self._stub = type("LegacyStub", (), {"acquire_first_person_perception": object()})()

    def acquire_first_person_perception(self, character_id: str, width: int, height: int) -> dict:
        self.calls.append(("capture", width, height))
        return {"image": "test-image", "objects": self.objects}

    def move_and_take_object(
        self,
        character_id: str,
        object_id: str,
        which_hand: int = 0,
        movable_object_ids=None,
    ) -> dict:
        self.calls.append(("take", object_id, which_hand, tuple(movable_object_ids or [])))
        self.hands[which_hand] = object_id
        return {"result": "success"}

    def move_and_take_puzzle_piece(self, character_id: str, piece_object_id: str, which_hand: int = 0) -> dict:
        self.calls.append(("puzzle_take", piece_object_id, which_hand))
        self.hands[which_hand] = piece_object_id
        return {"result": "success"}

    def has_object_in_hand(self, character_id: str):
        if not self.hands:
            return False, None
        return True, min(self.hands)

    def put_down_sth(self, character_id: str, target_location: list[float], **kwargs) -> dict:
        held = self.hands.pop(min(self.hands))
        self.calls.append(("legacy_place", held, tuple(target_location), kwargs))
        self.locations[held] = dict(zip(("X", "Y", "Z"), target_location))
        return {"result": "success"}

    def turn_in_degree(self, character_id: str, degree: float) -> dict:
        self.calls.append(("turn", degree))
        return {"result": "success"}

    def move_and_put_down(
        self,
        character_id: str,
        move_target_location: list[float],
        put_target_location: list[float],
        which_hand: int = 0,
        put_rotation=None,
    ) -> dict:
        held = self.hands.pop(which_hand)
        self.calls.append(("explicit_place", held, tuple(put_target_location), which_hand, put_rotation))
        self.locations[held] = dict(zip(("X", "Y", "Z"), put_target_location))
        return {"result": "success"}


class JigsawTransferTests(unittest.TestCase):
    def setUp(self) -> None:
        # The missing cells are top-right, middle-left, bottom-middle.
        occupied = [(10, 30), (20, 30), (20, 20), (30, 20), (30, 10), (10, 10)]
        self.objects = [_tile(index, y, z) for index, (y, z) in enumerate(occupied, start=1)]
        self.objects += [_tile(index, 50, z) for index, z in enumerate((30, 20, 10), start=7)]
        self.subject = {"task_type": "jigsaw", "reference_bounding": [5, 35, 35, 5]}

    def test_layout_infers_three_empty_cells_and_loose_candidates(self) -> None:
        layout = infer_layout(self.objects, self.subject["reference_bounding"])
        self.assertEqual({cell.name for cell in layout.empty_cells}, {"top-right", "middle-left", "bottom-middle"})
        self.assertEqual({obj["object_id"] for obj in layout.candidates}, {"7", "8", "9"})

    def test_runner_uses_raw_ids_and_two_hands_when_release_hand_is_explicit(self) -> None:
        class FakeAgent:
            character_id = "character"
            semantic_mapper = FakeMapper()

            def _acquire_camera_perception(self, subject, **kwargs):
                self.capture_calls += 1
                return "test-image", [], self.objects

        agent = FakeAgent()
        agent.objects = self.objects
        agent.capture_calls = 0
        agent.tongsim = FakeTongSim(self.objects)
        mapping = [
            {"object_id": "7", "cell": "top-right"},
            {"object_id": "8", "cell": "middle-left"},
            {"object_id": "9", "cell": "bottom-middle"},
        ]
        with patch("arenaagent.preliminary_baseline_agent.tasks.jigsaw.runner.solve_by_image_cost", return_value=mapping):
            run_dedicated_jigsaw(agent, self.subject)
        self.assertEqual(agent.capture_calls, 1)
        self.assertEqual([(call[1], call[2]) for call in agent.tongsim.calls if call[0] == "take"],
                         [("piece-7", 0), ("piece-8", 1), ("piece-9", 0)])
        self.assertEqual([(call[1], call[3]["which_hand"]) for call in agent.tongsim.calls if call[0] == "place"],
                         [("piece-7", 0), ("piece-8", 1), ("piece-9", 0)])
        self.assertEqual(agent.tongsim.hands, {})

    def test_single_hand_path_is_unaffected_by_second_hand_failure(self) -> None:
        class FakeAgent:
            character_id = "character"
            semantic_mapper = FakeMapper()

            def _acquire_camera_perception(self, subject, **kwargs):
                return "test-image", [], self.objects

        agent = FakeAgent()
        agent.objects = self.objects
        agent.tongsim = FakeTongSim(self.objects, fail_second_hand=True)
        mapping = [
            {"object_id": "7", "cell": "top-right"},
            {"object_id": "8", "cell": "middle-left"},
            {"object_id": "9", "cell": "bottom-middle"},
        ]
        with patch("arenaagent.preliminary_baseline_agent.tasks.jigsaw.runner.solve_by_image_cost", return_value=mapping):
            run_dedicated_jigsaw(agent, self.subject)
        self.assertEqual([(call[1], call[2]) for call in agent.tongsim.calls if call[0] == "take"],
                         [("piece-7", 0), ("piece-8", 1), ("piece-8", 0), ("piece-9", 0)])
        self.assertEqual([(call[1], call[3]["which_hand"]) for call in agent.tongsim.calls if call[0] == "place"],
                         [("piece-7", 0), ("piece-8", 0), ("piece-9", 0)])

    def test_legacy_competition_server_path_uses_1280_capture_and_old_place_rpc(self) -> None:
        class FakeAgent:
            character_id = "character"
            semantic_mapper = object()

        agent = FakeAgent()
        agent.tongsim = FakeLegacyTongSim(self.objects)
        mapping = [
            {"object_id": "7", "cell": "top-right"},
            {"object_id": "8", "cell": "middle-left"},
            {"object_id": "9", "cell": "bottom-middle"},
        ]
        with patch("arenaagent.preliminary_baseline_agent.tasks.jigsaw.runner.solve_by_image_cost", return_value=mapping):
            run_dedicated_jigsaw(agent, self.subject)

        self.assertIn(("capture", 1280, 720), agent.tongsim.calls)
        self.assertEqual(len([call for call in agent.tongsim.calls if call[0] == "legacy_place"]), 3)
        self.assertEqual(
            [(call[1], call[2]) for call in agent.tongsim.calls if call[0] == "take"],
            [("7", 0), ("8", 0), ("9", 0)],
        )

    def test_legacy_path_normalizes_only_board_rotation_outliers(self) -> None:
        class FakeAgent:
            character_id = "character"
            semantic_mapper = object()

        for obj in self.objects[:6]:
            obj["rotation"] = {"roll": 0.0, "pitch": 0.0, "yaw": 90.0}
        self.objects[2]["rotation"] = {"roll": 0.0, "pitch": 90.0, "yaw": 0.0}
        agent = FakeAgent()
        agent.tongsim = FakeLegacyTongSim(self.objects)
        mapping = [
            {"object_id": "7", "cell": "top-right"},
            {"object_id": "8", "cell": "middle-left"},
            {"object_id": "9", "cell": "bottom-middle"},
        ]
        with patch("arenaagent.preliminary_baseline_agent.tasks.jigsaw.runner.solve_by_image_cost", return_value=mapping):
            run_dedicated_jigsaw(agent, self.subject)

        takes = [(call[1], call[2]) for call in agent.tongsim.calls if call[0] == "take"]
        self.assertEqual(takes, [("7", 0), ("8", 0), ("9", 0), ("3", 0)])
        normalized = [call for call in agent.tongsim.calls if call[0] == "legacy_place" and call[1] == "3"]
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0][2], (837.0, 20.0, 20.0))
        self.assertEqual(normalized[0][3]["target_rotation"].yaw, 90.0)

    def test_rotation_uses_an_observed_board_pose_across_yaw_wraparound(self) -> None:
        rotations = [179.0, -179.0, 178.0, -178.0, 180.0, -180.0]
        for obj, yaw in zip(self.objects[:6], rotations):
            obj["rotation"] = {"roll": 0.0, "pitch": 0.0, "yaw": yaw}
        layout = infer_layout(self.objects, self.subject["reference_bounding"])
        self.assertIn(layout.target_rotation["yaw"], rotations)
        self.assertGreater(abs(layout.target_rotation["yaw"]), 170.0)

    def test_image_cost_matches_three_distinct_reference_colors(self) -> None:
        layout = infer_layout(self.objects, self.subject["reference_bounding"])
        clean = np.full((720, 640, 3), 225, dtype=np.uint8)
        colors = {
            "top-right": (15, 25, 215),
            "middle-left": (35, 195, 30),
            "bottom-middle": (215, 30, 25),
        }
        # Reference image occupies x=99..221, y=320..432 in YH's calibration.
        for cell in layout.empty_cells:
            x1 = round(99 + cell.column * 122 / 3)
            x2 = round(99 + (cell.column + 1) * 122 / 3)
            y1 = round(320 + cell.row * 112 / 3)
            y2 = round(320 + (cell.row + 1) * 112 / 3)
            clean[y1:y2, x1:x2] = colors[cell.name]
        expected = {"7": "middle-left", "8": "bottom-middle", "9": "top-right"}
        for candidate in layout.candidates:
            mapped_id = candidate["object_id"]
            world_y = candidate["place_location"]["Y"]
            world_z = candidate["place_location"]["Z"]
            x = round(1.8426 * world_y - 38.58)
            y = round(-2.1818 * world_z + 571)
            clean[y - 14:y + 14, x - 14:x + 14] = colors[expected[mapped_id]]
        combined = np.concatenate((clean, np.zeros_like(clean)), axis=1)
        encoded = cv2.imencode(".jpg", combined, [int(cv2.IMWRITE_JPEG_QUALITY), 98])[1]
        mapping = solve_by_image_cost(base64.b64encode(encoded.tobytes()).decode("ascii"), layout)
        self.assertEqual({item["object_id"]: item["cell"] for item in mapping}, expected)

    def test_only_jigsaw_enters_dedicated_branch(self) -> None:
        agent = PreliminaryBaselineAgent(stub=None, channel=None)
        agent.action_space = {"key": "answer"}
        with patch("arenaagent.preliminary_baseline_agent.preliminary_baseline_agent.run_dedicated_jigsaw") as dedicated:
            result = agent.run_step(self.subject, {})
        dedicated.assert_called_once()
        self.assertTrue(agent.subject_finished)
        self.assertIn("answer", result)


if __name__ == "__main__":
    unittest.main()
