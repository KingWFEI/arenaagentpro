from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from arenaagent.preliminary_baseline_agent.preliminary_baseline_agent import PreliminaryBaselineAgent
from arenaagent.preliminary_baseline_agent.tasks.counting.runtime import run_counting_subject

from arenaagent.preliminary_baseline_agent.tasks.counting.strategy import (
    CountingMemory,
    CountingRecord,
    CountingStrategy,
    _matches_competition_apple,
    _matches_competition_cup,
    _parse_review_response,
    count_for_option,
    option_for_count,
    target_category,
)


def observed(object_id, shape="Unknown", color="Unknown", x=0, y=0, z=0, size=10):
    return {
        "object_id": str(object_id),
        "shape": shape,
        "color": color,
        "place_location": {"X": x, "Y": y, "Z": z},
        "world_aabb": {
            "min": {"x": x, "y": y, "z": z},
            "max": {"x": x + size, "y": y + size, "z": z + size},
        },
    }


class CountingMemoryTests(unittest.TestCase):
    def test_stable_ids_are_deduplicated_across_views(self):
        memory = CountingMemory()
        memory.add(0, None, [observed("1", "Round", "Red"), observed("2", "Bowl", x=20)])
        second = memory.add(90, None, [observed("99", "Round", "Red", x=0.5), observed("3", "Chair", x=40)])
        self.assertEqual(set(memory.records), {"o1", "o2", "o3"})
        self.assertEqual(memory.records["o1"].views, {1, 2})
        self.assertEqual(second.image_labels["o1"], "99")

    def test_candidate_filter_keeps_small_round_apple_and_excludes_known_bowl(self):
        memory = CountingMemory()
        memory.add(0, None, [observed("apple", "Round", "Red"), observed("bowl", "Bowl", x=20)])
        self.assertEqual(memory.candidate_ids("apple"), {"o1"})

    def test_clock_candidates_include_small_square_electronic_clock_casings(self):
        memory = CountingMemory()
        memory.add(0, None, [observed("1", "Cube", "White", size=12)])
        self.assertEqual(memory.candidate_ids("clock"), {"o1"})

    def test_greedy_view_cover_avoids_redundant_images(self):
        memory = CountingMemory()
        memory.add(0, "a", [observed("1"), observed("2", x=10)])
        memory.add(90, "b", [observed("8", x=10)])
        memory.add(180, "c", [observed("3", x=20)])
        selected = memory.covering_views({"o1", "o2", "o3"})
        self.assertEqual([view.index for view in selected], [1, 3])


class CountingProtocolTests(unittest.TestCase):
    def test_nearest_option_breaks_exact_tie_toward_higher_count(self):
        option = CountingStrategy._nearest_option(4.5, {"A": 4, "B": 5, "C": 7})

        self.assertEqual(option, "B")

    def test_competition_cup_prototype_excludes_larger_white_containers(self):
        red_cup = CountingRecord("cup", color="Red", shape="Cylinder", size=(7, 7, 8))
        white_container = CountingRecord(
            "container", color="White", shape="Cylinder", size=(12, 12, 15)
        )
        self.assertTrue(_matches_competition_cup(red_cup))
        self.assertFalse(_matches_competition_cup(white_container))

    def test_competition_apple_prototype_accepts_red_round_asset(self):
        apple = CountingRecord("apple", color="Red", shape="Round", size=(22, 18, 22))
        self.assertTrue(_matches_competition_apple(apple))

    def test_local_cup_path_counts_five_red_cups_not_three_white_containers(self):
        strategy = CountingStrategy()
        strategy.reset({})
        records = [
            CountingRecord(f"red{index}", color="Red", shape="Cylinder", size=(6.5, 6.5, 7.6))
            for index in range(5)
        ]
        records.extend(
            CountingRecord(
                f"white{index}", color="White", shape="Cylinder", size=(11.7, 11.7, 12.2)
            )
            for index in range(3)
        )
        strategy.memory.records = {record.object_id: record for record in records}

        matches, decisive = strategy._local_prototype_ids("cup")

        self.assertTrue(decisive)
        self.assertEqual(len(matches), 5)

    def test_local_clock_path_is_decisive_when_all_candidates_match_asset(self):
        strategy = CountingStrategy()
        strategy.reset({})
        records = [
            CountingRecord(f"clock{index}", color="Black", shape="Rectangle", size=(11, 5, 3.4))
            for index in range(4)
        ]
        strategy.memory.records = {record.object_id: record for record in records}

        matches, decisive = strategy._local_prototype_ids("clock")

        self.assertTrue(decisive)
        self.assertEqual(len(matches), 4)

    def test_single_candidate_review_row_without_array_is_accepted(self):
        row = {
            "object_id": "o1",
            "label": "target",
            "confidence": 95,
            "source_view": 1,
            "evidence": "可见电子时间显示",
        }
        self.assertEqual(_parse_review_response(json.dumps(row, ensure_ascii=False)), [row])

    def test_target_and_option_are_parsed_from_public_subject(self):
        subject = {"question": "一共有多少个苹果在房间中？", "options": {"A": 4, "G": 5}}
        self.assertEqual(target_category(subject), "apple")
        self.assertEqual(option_for_count(5, subject["options"]), "G")
        self.assertEqual(count_for_option("G", subject["options"]), 5)

    def test_review_rows_must_reference_a_shown_view(self):
        memory = CountingMemory()
        view = memory.add(0, None, [observed("1")])
        rows = [
            {
                "object_id": "o1",
                "label": "target",
                "confidence": 95,
                "source_view": 99,
                "evidence": "可见苹果果柄",
            }
        ]
        accepted = CountingStrategy._validate_review_rows(rows, {"o1"}, {1: view})
        self.assertEqual(accepted, [])

    def test_clock_negative_requires_a_concrete_alternative_object(self):
        memory = CountingMemory()
        view = memory.add(0, None, [observed("1", "Rectangle", "Black")])
        rows = [
            {
                "object_id": "o1",
                "label": "not_target",
                "confidence": 95,
                "source_view": 1,
                "evidence": "没有看见圆形表盘或数字",
            }
        ]
        accepted = CountingStrategy._validate_review_rows(rows, {"o1"}, {1: view}, "clock")
        self.assertEqual(accepted, [])

    def test_clock_prompt_defines_white_square_digital_asset(self):
        strategy = CountingStrategy()
        strategy.reset({})
        strategy.memory.add(0, None, [observed("1", "Rectangle", "Black", size=10)])
        messages, _ = strategy._build_review_messages(
            SimpleNamespace(),
            {"question": "房间中有多少个钟？"},
            {},
            "clock",
            {"o1"},
            False,
        )
        self.assertIn("白色方形/矩形电子钟", messages[0]["content"])

    def test_unresolved_clock_geometry_maps_to_exact_option_instead_of_half_vote(self):
        strategy = CountingStrategy()
        strategy.reset({})
        for index in range(4):
            obj = observed(str(index + 1), "Rectangle", "Black", x=index * 30, size=10)
            obj["world_aabb"]["max"] = {
                "x": index * 30 + 11,
                "y": 3.5,
                "z": 5,
            }
            strategy.memory.add(index * 90, None, [obj])
        option = strategy._review_and_choose(
            FakeAgent(),
            {"options": {"A": 4, "B": 1, "C": 6}},
            {},
            "clock",
            set(strategy.memory.records),
        )
        self.assertEqual(option, "A")

    def test_vote_fusion_counts_unique_positive_instances(self):
        strategy = CountingStrategy()
        strategy.reset({})
        strategy.review_votes = {
            "1": [("target", 92)],
            "2": [("target", 86), ("target", 90)],
            "3": [("not_target", 95)],
        }
        count, expectation = strategy._fuse_votes({"1", "2", "3"})
        self.assertEqual(count, 2)
        self.assertGreater(expectation, 1.5)

    def test_missing_review_is_uncertain_not_an_implicit_negative(self):
        strategy = CountingStrategy()
        strategy.reset({})
        count, expectation = strategy._fuse_votes({"unreviewed"})
        self.assertEqual(count, 0)
        self.assertEqual(expectation, 0.5)

    def test_unsynchronized_post_turn_perception_is_rejected(self):
        consistent, reason = CountingStrategy._perception_is_consistent(
            {
                "aligned_object_count": 2,
                "segmentation_region_count": 10,
                "mapped_pixel_coverage": 0.2,
            }
        )
        self.assertFalse(consistent)
        self.assertIn("not_synchronized", reason)


class FakeResponse:
    def __init__(self, rows):
        self.text = json.dumps({"object_review": rows}, ensure_ascii=False)


class FakeClient:
    def __init__(self, events):
        self.calls = 0
        self.events = events

    def invoke(self, messages, max_retries=1):
        del max_retries
        self.calls += 1
        self.events.append("vlm")
        source_views = {}
        for item in messages[1]["content"]:
            if item.get("type") != "text":
                continue
            try:
                payload = json.loads(item["text"])
            except (KeyError, TypeError, ValueError):
                continue
            source_view = payload.get("source_view")
            for object_id in payload.get("visible_candidate_ids", []):
                source_views[object_id] = source_view
        return FakeResponse(
            [
                {
                    "object_id": "o1",
                    "label": "target",
                    "confidence": 95,
                    "source_view": source_views.get("o1", 4),
                    "evidence": "红色圆形果实带清晰果柄",
                },
                {
                    "object_id": "o2",
                    "label": "not_target",
                    "confidence": 96,
                    "source_view": source_views.get("o2", 4),
                    "evidence": "可见碗沿和中空容器",
                },
            ]
        )


class FakeAgent:
    def __init__(self):
        self.cfg = SimpleNamespace(
            counting_post_turn_settle_seconds=0,
            counting_capture_max_attempts=3,
            counting_image_max_width=0,
            counting_move_distance=80,
            counting_max_model_calls=2,
        )
        self.action_space = {"key": "answer"}
        self.events = []
        self.vlm_client = FakeClient(self.events)
        self._last_visible_objects_info = []
        self.actions = []
        self.saved_prompts = []
        self.capture_calls = 0

    def _execute_action_and_record(self, action):
        self.actions.append(action)
        self.events.append(action["action"])
        if action["action"] == "submit_answer":
            return {"answer": action["output"]}
        return {"result": "success"}

    def _acquire_camera_perception(self, subject, is_save=True, **kwargs):
        del subject, is_save, kwargs
        self.capture_calls += 1
        objects = [
            observed("1", "Round", "Red", x=10),
            observed("2", "Bowl", "White", x=30),
        ]
        self._last_visible_objects_info = objects
        return None, [{"object_id": "1"}, {"object_id": "2"}], objects

    def _save_prompt_messages(self, messages):
        self.saved_prompts.append(messages)

    @staticmethod
    def _to_data_url(value):
        return value


class CountingIntegrationTests(unittest.TestCase):
    def test_integer_counting_output_does_not_change_other_tasks(self):
        agent = PreliminaryBaselineAgent(stub=None, channel=None)
        agent.action_space = {"key": "answer", "type": "int"}
        agent._active_task_type = "counting"
        self.assertEqual(agent._handle_submit_answer({}, {"output": 5}), {"answer": 5})
        agent._active_task_type = "raven"
        self.assertEqual(agent._handle_submit_answer({}, {"output": 5}), {"answer": "5"})

    def test_counting_routes_without_calling_other_task_loops(self):
        agent = PreliminaryBaselineAgent(stub=None, channel=None)
        with patch.object(agent, "_get_subject_from_task", return_value={"task_type": "counting"}), \
                patch("arenaagent.preliminary_baseline_agent.preliminary_baseline_agent.run_counting_subject") as runner:
            agent._run_subject()
        runner.assert_called_once_with(agent, {"task_type": "counting"})

    def test_explicit_wrong_count_uses_zrq_ranked_recovery(self):
        strategy = Mock()
        strategy.run_fast.return_value = {"answer": 3}
        strategy.ranked_recovery_counts.return_value = [4, 5]
        agent = Mock()
        agent._ensure_task_strategy.return_value = strategy
        agent._raven_current_subject_index.return_value = 1
        agent._apply_action.side_effect = [
            {"answer_right": False},
            {"answer_right": True},
        ]
        agent.action_space = {"key": "answer"}
        agent.cfg = SimpleNamespace(counting_max_recovery_submissions=7)
        run_counting_subject(agent, {"task_type": "counting", "options": {"A": 3, "B": 4, "C": 5}})
        self.assertEqual([call.args[0] for call in agent._apply_action.call_args_list],
                         [{"answer": 3}, {"answer": 4}])
        self.assertTrue(agent.subject_finished)
        agent._evaluate_subject.assert_called_once()


class CountingFastPathTests(unittest.TestCase):
    def test_corner_route_moves_once_and_scans_only_ninety_degrees(self):
        agent = FakeAgent()
        agent._spawn_xy = (282.0, -353.0)
        agent._spawn_yaw = -90.0
        agent.cfg.counting_corner_move_distance = 80.0
        strategy = CountingStrategy()
        subject = {
            "task_type": "counting",
            "counting_type": "CountingObjects",
            "question": "房间中有多少个碗？",
            "options": {"A": 0, "B": 1, "C": 2},
        }
        strategy.reset(subject)

        result = strategy.run_fast(agent, subject, {})

        self.assertEqual(result, {"answer": 1})
        self.assertEqual(agent.capture_calls, 2)
        self.assertEqual(
            [action["parameters"]["degree"] for action in agent.actions if action["action"] == "turn_in_degree"],
            [90.0, 180.0],
        )
        self.assertEqual(agent.actions[0]["action"], "move_to_location")
        self.assertAlmostEqual(agent.actions[0]["parameters"]["target_location"]["X"], 362.0)
        self.assertAlmostEqual(agent.actions[0]["parameters"]["target_location"]["Y"], -353.0)

    def test_fast_runner_scans_deduplicates_reviews_and_submits_count(self):
        agent = FakeAgent()
        strategy = CountingStrategy()
        subject = {
            "task_type": "counting",
            "counting_type": "CountingObjects",
            "question": "一共有多少个苹果在房间中？",
            "options": {"A": 0, "B": 1, "C": 2},
        }
        strategy.reset(subject)
        result = strategy.run_fast(agent, subject, {})
        self.assertEqual(result, {"answer": 1})
        self.assertEqual(len(strategy.memory.records), 2)
        self.assertEqual(agent.vlm_client.calls, 0)
        self.assertEqual(agent.capture_calls, 8)
        self.assertIn("move_forward", agent.events)
        self.assertEqual(agent.actions[-1]["action"], "submit_answer")

    def test_reliable_bowl_count_stops_after_panorama_and_submits_count(self):
        agent = FakeAgent()
        strategy = CountingStrategy()
        subject = {
            "task_type": "counting",
            "counting_type": "CountingObjects",
            "question": "房间中有多少个碗？",
            "options": {"A": 0, "B": 1, "C": 2},
        }
        strategy.reset(subject)

        result = strategy.run_fast(agent, subject, {})

        self.assertEqual(result, {"answer": 1})
        self.assertEqual(agent.capture_calls, 4)
        self.assertEqual(agent.vlm_client.calls, 0)
        self.assertEqual(agent.actions[-1]["action"], "submit_answer")

    def test_occlusion_plan_prefers_large_unknown_furniture(self):
        strategy = CountingStrategy()
        strategy.reset({})
        small_target = observed("1", "Round", "Red", x=10, size=12)
        furniture = observed("2", "Rectangle", "Brown", x=100, size=10)
        furniture["world_aabb"]["max"] = {"x": 280, "y": 120, "z": 90}
        strategy.memory.add(90, None, [small_target, furniture])

        plan = strategy._select_occlusion_plan(SimpleNamespace(_spawn_xy=(0, 0)))

        self.assertIsNotNone(plan)
        self.assertEqual(plan.object_id, "o2")
        self.assertEqual(plan.heading, 90)

    def test_occlusion_move_reuses_current_view_object_id_without_refresh(self):
        strategy = CountingStrategy()
        strategy.reset({})
        furniture = observed("17", "Rectangle", "Brown", x=100, size=10)
        furniture["world_aabb"]["max"] = {"x": 280, "y": 120, "z": 90}
        strategy.memory.add(90, None, [furniture])
        plan = strategy._select_occlusion_plan(SimpleNamespace(_spawn_xy=(0, 0)))
        agent = FakeAgent()

        def refreshed_perception(subject, is_save=True):
            del subject, is_save
            refreshed = observed("99", "Rectangle", "Brown", x=100, size=10)
            refreshed["world_aabb"]["max"] = {"x": 280, "y": 120, "z": 90}
            return None, [{"object_id": "99"}], [refreshed]

        agent._acquire_camera_perception = refreshed_perception
        moved, heading = strategy._move_to_occlusion_zone(agent, {}, plan)

        self.assertTrue(moved)
        self.assertEqual(heading, 90)
        self.assertEqual(agent.actions[-1]["action"], "move_to_object")
        self.assertEqual(agent.actions[-1]["parameters"]["object_id"], "17")
        self.assertEqual(agent.capture_calls, 0)


if __name__ == "__main__":
    unittest.main()
