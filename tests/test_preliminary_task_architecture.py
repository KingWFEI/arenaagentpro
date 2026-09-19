from __future__ import annotations

import io
import unittest
from unittest.mock import call, patch

import numpy as np
from PIL import Image

from arenaagent.preliminary_baseline_agent.preliminary_baseline_agent import PreliminaryBaselineAgent
from arenaagent.preliminary_baseline_agent.task_registry import create_task_strategy, supported_task_types
from arenaagent.preliminary_baseline_agent.task_runtime import TaskContext, normalize_task_type
from arenaagent.preliminary_baseline_agent.tasks.base import GenericTaskStrategy
from arenaagent.preliminary_baseline_agent.tasks.tidyroom.geometry import placement_check
from arenaagent.preliminary_baseline_agent.tasks.tidyroom.destinations import (
    infer_target_category,
    normalize_destination_type,
    recommended_destination,
)
from arenaagent.preliminary_baseline_agent.tasks.tidyroom.recovery import RecoveryPolicy
from arenaagent.preliminary_baseline_agent.tasks.tidyroom.planner import TidyRoomPlanner
from arenaagent.preliminary_baseline_agent.tasks.tidyroom.scheduler import TargetScheduler
from arenaagent.preliminary_baseline_agent.tasks.tidyroom.strategy import TidyRoomStrategy
from arenaagent.preliminary_baseline_agent.tasks.tidyroom.world_model import TidyRoomWorldModel
from arenaagent.semantic_mapper import SemanticMapper
from arenaagent.vlm_agent.vlm_agent import VLMAgent


class TaskRegistryTests(unittest.TestCase):
    def test_registry_contains_exactly_five_preliminary_tasks(self) -> None:
        self.assertEqual(
            supported_task_types(),
            ("tidyroom", "jigsaw", "counting", "npc", "raven"),
        )

    def test_each_task_has_an_external_prompt(self) -> None:
        for task_type in supported_task_types():
            with self.subTest(task_type=task_type):
                prompt = create_task_strategy(task_type).load_prompt()
                self.assertTrue(prompt.strip())

    def test_unknown_task_uses_safe_fallback(self) -> None:
        strategy = create_task_strategy("future_task")
        self.assertIsInstance(strategy, GenericTaskStrategy)
        self.assertEqual(strategy.task_type, "future_task")

    def test_drink_container_defaults_to_dining_table(self) -> None:
        category = infer_target_category("BP_DrinkContainer_Can_07_TEST")
        self.assertEqual(category, "drink_container")
        self.assertEqual(recommended_destination(category), "dining_table")

    def test_open_food_labels_share_the_closed_dining_table_destination(self) -> None:
        self.assertEqual(recommended_destination("food_banana"), "dining_table")
        self.assertEqual(recommended_destination("fruit_orange"), "dining_table")
        self.assertIsNone(recommended_destination("collectible_toy"))

    def test_destination_aliases_are_normalized_to_four_execution_types(self) -> None:
        self.assertEqual(normalize_destination_type("table"), "dining_table")
        self.assertEqual(normalize_destination_type("shoe rack"), "shoe_storage")
        self.assertIsNone(normalize_destination_type("bookshelf"))


class TongSimLifecycleTests(unittest.TestCase):
    class FakeTongSim:
        def __init__(self) -> None:
            self.destroyed = []
            self.close_calls = 0

        def destory_character(self, character_id):
            self.destroyed.append(character_id)

        def close(self):
            self.close_calls += 1

    def test_shared_tongsim_is_retained_between_subjects(self) -> None:
        client = self.FakeTongSim()
        agent = VLMAgent(stub=None, channel=None)
        agent.configure_shared_tongsim(client)
        agent.character_id = "subject-character"
        agent._initialized = True

        agent.deinit()

        self.assertEqual(client.destroyed, ["subject-character"])
        self.assertEqual(client.close_calls, 0)
        self.assertIsNone(agent.tongsim)

    def test_single_subject_still_closes_its_tongsim_client(self) -> None:
        client = self.FakeTongSim()
        agent = VLMAgent(stub=None, channel=None)
        agent.tongsim = client
        agent.character_id = "single-character"
        agent._initialized = True

        agent.deinit()

        # close() performs client-scoped cleanup, including the character and
        # camera.  Calling destory_character() as well double-releases UE state.
        self.assertEqual(client.destroyed, [])
        self.assertEqual(client.close_calls, 1)


class TaskRuntimeTests(unittest.TestCase):
    def test_lightweight_perception_reuses_object_details_without_images(self) -> None:
        class FakeTongSim:
            def __init__(self) -> None:
                self.info_calls = 0
                self.aabb_calls = 0

            @staticmethod
            def fetch_first_person_visible_objects(character_id):
                del character_id
                return [{"object_id": "raw-cup", "segmentation_id": 7}]

            def get_object_basic_info(self, raw_id):
                self.info_calls += 1
                return {"color": "red", "shape": "cup", "place_location": {"X": 1, "Y": 2, "Z": 3}}

            def get_object_world_aabb(self, raw_id):
                self.aabb_calls += 1
                return {
                    "min": {"x": 0, "y": 1, "z": 2},
                    "max": {"x": 2, "y": 3, "z": 4},
                }

        tongsim = FakeTongSim()
        mapper = SemanticMapper(tongsim=tongsim, character_id="agent")
        for _ in range(2):
            image, objects, details = mapper.get_perception_from_camera(
                include_images=False,
                include_object_details=True,
                use_cached_details=True,
            )
            self.assertIsNone(image)
            self.assertEqual(objects[0]["object_id"], 1)
            self.assertEqual(details[0]["color"], "red")
        self.assertEqual(tongsim.info_calls, 1)
        self.assertEqual(tongsim.aabb_calls, 1)

    def test_semantic_mapper_reports_mapped_pixel_coverage(self) -> None:
        rgb_buffer = io.BytesIO()
        Image.new("RGB", (2, 2), color=(255, 255, 255)).save(rgb_buffer, format="PNG")
        segmentation = np.array(
            [
                [[1, 0, 0], [1, 0, 0]],
                [[1, 0, 0], [2, 0, 0]],
            ],
            dtype=np.uint8,
        )
        segmentation_buffer = io.BytesIO()
        Image.fromarray(segmentation).save(segmentation_buffer, format="PNG")

        class FakeTongSim:
            @staticmethod
            def acquire_first_person_image(character_id, encode_base64=False):
                del character_id, encode_base64
                return rgb_buffer.getvalue()

            @staticmethod
            def acquire_first_person_segmantic_image(character_id, encode_base64=False):
                del character_id, encode_base64
                return segmentation_buffer.getvalue()

            @staticmethod
            def fetch_first_person_visible_objects(character_id):
                del character_id
                return [{"object_id": "raw-object", "segmentation_id": 1}]

        mapper = SemanticMapper(tongsim=FakeTongSim(), character_id="agent")
        mapper.get_perception_from_camera(include_object_details=False)

        self.assertEqual(mapper.last_perception_diagnostics["total_pixel_count"], 4)
        self.assertEqual(mapper.last_perception_diagnostics["mapped_pixel_count"], 3)
        self.assertEqual(mapper.last_perception_diagnostics["mapped_pixel_coverage"], 0.75)

    def test_tidyroom_post_turn_perception_retries_until_counts_are_consistent(self) -> None:
        class DiagnosticMapper:
            def __init__(self) -> None:
                self.calls: list[dict] = []
                self.last_perception_diagnostics: dict[str, int] = {}

            def get_perception_from_camera(self, **kwargs):
                self.calls.append(dict(kwargs))
                if len(self.calls) == 1:
                    self.last_perception_diagnostics = {
                        "visible_object_count": 2,
                        "aligned_object_count": 2,
                        "segmentation_region_count": 36,
                        "mapped_pixel_coverage": 0.99,
                    }
                    return "bad-image", [{"object_id": 1}, {"object_id": 2}], []
                self.last_perception_diagnostics = {
                    "visible_object_count": 36,
                    "aligned_object_count": 36,
                    "segmentation_region_count": 36,
                    "mapped_pixel_coverage": 0.99,
                }
                return "good-image", [{"object_id": 1}], [{"object_id": "1"}]

        agent = PreliminaryBaselineAgent(stub=None, channel=None)
        agent.semantic_mapper = DiagnosticMapper()
        agent._last_executed_action_name = "turn_in_degree"

        with patch("arenaagent.preliminary_baseline_agent.preliminary_baseline_agent.time.sleep") as sleep_mock:
            result = agent._acquire_camera_perception(
                {"task_type": "tidyroom"},
                is_save=True,
                include_images=True,
                include_object_details=True,
            )

        self.assertEqual(result[0], "good-image")
        self.assertEqual(sleep_mock.call_args_list, [call(0.75)] * 2)
        self.assertEqual(len(agent.semantic_mapper.calls), 2)
        self.assertEqual(
            agent.semantic_mapper.calls[0]["save_label"],
            "postturn_001_attempt_01",
        )
        self.assertEqual(
            agent.semantic_mapper.calls[1]["save_label"],
            "postturn_001_attempt_02",
        )
        self.assertFalse(agent._tidyroom_post_turn_perception_blocked)

    def test_tidyroom_unified_perception_waits_for_stable_ids_not_fixed_count(self) -> None:
        class StabilizingUnifiedMapper:
            def __init__(self) -> None:
                self.calls = 0
                self.last_perception_diagnostics: dict = {}

            def get_perception_from_camera(self, **kwargs):
                del kwargs
                self.calls += 1
                ids = ("2", "7") if self.calls == 1 else tuple(str(i) for i in range(10))
                self.last_perception_diagnostics = {
                    "source": "unified",
                    "visible_object_count": len(ids),
                    "visible_object_ids": ids,
                    "image_present": True,
                    "right_nonblack_ratio": 0.8,
                    "left_right_difference_ratio": 0.8,
                }
                return f"image-{self.calls}", [{"object_id": value} for value in ids], []

        agent = PreliminaryBaselineAgent(stub=None, channel=None)
        agent.semantic_mapper = StabilizingUnifiedMapper()
        agent._last_executed_action_name = "turn_in_degree"
        agent._last_executed_action = {
            "action": "turn_in_degree",
            "parameters": {"degree": 90},
        }

        with patch("arenaagent.preliminary_baseline_agent.preliminary_baseline_agent.time.sleep"):
            result = agent._acquire_camera_perception(
                {"task_type": "tidyroom"},
                include_images=True,
            )

        self.assertEqual(result[0], "image-3")
        self.assertEqual(agent.semantic_mapper.calls, 3)
        self.assertFalse(agent._tidyroom_post_turn_perception_blocked)

    def test_tidyroom_post_turn_perception_blocks_bad_data_after_retry_limit(self) -> None:
        class AlwaysBadMapper:
            def __init__(self) -> None:
                self.calls = 0
                self.last_perception_diagnostics = {
                    "visible_object_count": 2,
                    "aligned_object_count": 2,
                    "segmentation_region_count": 36,
                    "mapped_pixel_coverage": 0.99,
                }

            def get_perception_from_camera(self, **kwargs):
                del kwargs
                self.calls += 1
                return "bad-image", [{"object_id": 1}, {"object_id": 2}], []

        agent = PreliminaryBaselineAgent(stub=None, channel=None)
        agent.semantic_mapper = AlwaysBadMapper()
        agent._last_executed_action_name = "turn_in_degree"

        with patch("arenaagent.preliminary_baseline_agent.preliminary_baseline_agent.time.sleep"):
            result = agent._acquire_camera_perception(
                {"task_type": "tidyroom"},
                is_save=True,
                include_images=True,
            )

        self.assertEqual(result, (None, [], []))
        self.assertEqual(agent.semantic_mapper.calls, 10)
        self.assertTrue(agent._tidyroom_post_turn_perception_blocked)

    def test_tidyroom_post_turn_perception_retries_low_pixel_coverage(self) -> None:
        class CoverageMapper:
            def __init__(self) -> None:
                self.calls = 0
                self.last_perception_diagnostics: dict[str, float | int] = {}

            def get_perception_from_camera(self, **kwargs):
                del kwargs
                self.calls += 1
                coverage = 0.03 if self.calls == 1 else 0.97
                self.last_perception_diagnostics = {
                    "visible_object_count": 3 if self.calls == 1 else 11,
                    "aligned_object_count": 3 if self.calls == 1 else 11,
                    "segmentation_region_count": 4 if self.calls == 1 else 11,
                    "mapped_pixel_count": int(1280 * 1280 * coverage),
                    "total_pixel_count": 1280 * 1280,
                    "mapped_pixel_coverage": coverage,
                }
                return f"image-{self.calls}", [], []

        agent = PreliminaryBaselineAgent(stub=None, channel=None)
        agent.semantic_mapper = CoverageMapper()
        agent._last_executed_action_name = "turn_in_degree"

        with patch("arenaagent.preliminary_baseline_agent.preliminary_baseline_agent.time.sleep") as sleep_mock:
            result = agent._acquire_camera_perception(
                {"task_type": "tidyroom"},
                include_images=True,
            )

        self.assertEqual(result[0], "image-2")
        self.assertEqual(agent.semantic_mapper.calls, 2)
        self.assertEqual(sleep_mock.call_args_list, [call(0.75)] * 2)

    def test_shared_runtime_skips_vlm_when_local_action_is_available(self) -> None:
        class FakeMapper:
            @staticmethod
            def get_perception_from_camera(is_save: bool = False):
                del is_save
                return None, [], []

        class NeverInvokeClient:
            @staticmethod
            def invoke(messages):
                del messages
                raise AssertionError("本地动作存在时不应调用 VLM")

        class LocalOnlyAgent(VLMAgent):
            def _should_handle_piece_transfer(self):
                return False

            def _get_local_action(self, subject, task_response, prompt_variables):
                del subject, task_response, prompt_variables
                return {"action": "turn_in_degree", "parameters": {"degree": 90}, "output": 0}

            def _do_action(self, action):
                return {"result": "success", "action": action["action"]}

        agent = LocalOnlyAgent(stub=None, channel=None)
        agent._initialized = True
        agent.semantic_mapper = FakeMapper()
        agent.vlm_client = NeverInvokeClient()
        result = agent.run_step({"subject": "本地控制测试", "goal": "验证本地动作路径"}, {})
        self.assertEqual(result, {"result": "success", "action": "turn_in_degree"})

    def test_stage_aliases_are_normalized(self) -> None:
        self.assertEqual(normalize_task_type({"stage": "tidy_room"}), "tidyroom")
        self.assertEqual(normalize_task_type({"task_type": "RAVEN"}), "raven")

    def test_agent_switches_strategy_without_network_access(self) -> None:
        agent = PreliminaryBaselineAgent(stub=None, channel=None)
        first = agent._ensure_task_strategy({"task_type": "tidyroom", "subject": "整理"})
        agent.history_messages = [{"role": "user", "content": "old task"}]
        agent._action_histories = [{"action": {"action": "old"}, "result": {}}]
        second = agent._ensure_task_strategy({"task_type": "counting", "subject": "计数"})
        self.assertIsInstance(first, TidyRoomStrategy)
        self.assertEqual(second.task_type, "counting")
        self.assertEqual(agent.active_task_type, "counting")
        self.assertIsNot(first, second)
        self.assertEqual(agent.history_messages, [])
        self.assertEqual(agent._action_histories, [])

    def test_tidyroom_uses_lightweight_perception_and_defers_intermediate_updates(self) -> None:
        agent = PreliminaryBaselineAgent(stub=None, channel=None)
        agent._ensure_task_strategy({"task_type": "tidyroom", "subject": "整理"})
        self.assertFalse(agent._should_use_lightweight_perception({"task_type": "tidyroom"}))
        agent._last_executed_action_name = "move_and_take_object"
        result = agent._apply_action({"result": "success"})
        self.assertEqual(result, {"deferred": True})
        self.assertEqual(agent._last_apply_resp, {"deferred": True})

    def test_tidyroom_seeds_scheduler_with_spawn_position(self) -> None:
        agent = PreliminaryBaselineAgent(stub=None, channel=None)
        agent._spawn_xy = (634.0, -40.0)

        strategy = agent._ensure_task_strategy({"task_type": "tidyroom", "subject": "整理"})

        self.assertEqual(strategy.scheduler.estimated_agent_xy, (634.0, -40.0))

    def test_tidyroom_uses_short_action_specific_settle_delays(self) -> None:
        agent = PreliminaryBaselineAgent(stub=None, channel=None)
        agent.sleep_between_steps = 0.25
        agent._last_executed_action_name = "move_and_take_object"
        agent._last_executed_action = {"parameters": {}}
        self.assertEqual(agent._tidyroom_settle_delay(), 0.05)

        agent._last_executed_action_name = "put_down_to_location"
        agent._last_executed_action = {"parameters": {"disable_physics": True}}
        self.assertEqual(agent._tidyroom_settle_delay(), 0.02)

        agent._last_executed_action = {"parameters": {"disable_physics": False}}
        self.assertEqual(agent._tidyroom_settle_delay(), 0.25)

        agent._last_executed_action_name = "turn_in_degree"
        agent._last_executed_action = {"parameters": {"degree": 45}}
        self.assertEqual(agent._tidyroom_settle_delay(), 0.0)

    def test_tidyroom_first_perception_has_no_blind_rotation_or_delay(self) -> None:
        class FakeTongSim:
            def __init__(self) -> None:
                self.turns = []

            def turn_in_degree(self, character_id, degree):
                self.turns.append((character_id, degree))
                return {"result": "success"}

        subject = {"task_type": "tidyroom", "subject": "整理房间"}
        agent = PreliminaryBaselineAgent(stub=None, channel=None)
        strategy = agent._ensure_task_strategy(subject)
        agent.tongsim = FakeTongSim()
        agent.character_id = "agent-1"

        strategy.before_step(subject, {})
        agent._before_lightweight_initial_perception(subject)

        self.assertEqual(agent.tongsim.turns, [])
        self.assertEqual(agent._action_histories, [])
        self.assertEqual(strategy.scanner.turns_completed, 0)
        self.assertEqual(agent._lightweight_initial_perception_delay(subject), 0.0)
        self.assertEqual(agent._lightweight_empty_scene_retry_delay(subject), 0.0)
        self.assertFalse(agent._should_force_vlm_after_empty_lightweight_perception(subject))

    def test_tidyroom_initial_full_capture_does_not_enter_lightweight_retry(self) -> None:
        class RetryMapper:
            def __init__(self) -> None:
                self.calls = 0
                self.object_id_map = {}

            def get_perception_from_camera(self, **kwargs):
                del kwargs
                self.calls += 1
                if self.calls == 1:
                    return None, [], []
                return None, [{"object_id": 1}], [{"object_id": "1"}]

        class RetryAgent(PreliminaryBaselineAgent):
            def _lightweight_initial_perception_delay(self, subject):
                del subject
                return 0.0

            def _should_force_vlm_after_empty_lightweight_perception(self, subject):
                del subject
                return False

            def _lightweight_empty_scene_retry_delay(self, subject):
                del subject
                return 0.001

            def _get_local_action(self, subject, task_response, prompt_variables):
                del subject, task_response, prompt_variables
                return {"action": "turn_in_degree", "parameters": {"degree": 0}, "output": 0}

            def _do_action(self, action):
                return {"result": "success", "action": action["action"]}

        agent = RetryAgent(stub=None, channel=None)
        agent._initialized = True
        agent.semantic_mapper = RetryMapper()
        result = agent.run_step(
            {
                "task_type": "tidyroom",
                "subject": "整理房间",
                "movable_object_id": ["BP_Pillow_TEST"],
            },
            {},
        )

        self.assertEqual(result["result"], "success")
        self.assertEqual(agent.semantic_mapper.calls, 1)

    def test_tidyroom_first_empty_scene_waits_once_then_turns_without_vlm(self) -> None:
        class FirstFrameMapper:
            def __init__(self) -> None:
                self.calls: list[dict] = []
                self.object_id_map = {}

            def get_perception_from_camera(self, **kwargs):
                self.calls.append(dict(kwargs))
                return None, [], []

        class NoInvokeClient:
            def __init__(self) -> None:
                self.calls = 0

            def invoke(self, messages):
                del messages
                self.calls += 1
                raise AssertionError("first frame must not call VLM")

        class FirstFrameAgent(PreliminaryBaselineAgent):
            def _do_action(self, action):
                return {"result": "success", "action": action["action"]}

        agent = FirstFrameAgent(stub=None, channel=None)
        agent._initialized = True
        agent.semantic_mapper = FirstFrameMapper()
        agent.vlm_client = NoInvokeClient()

        with patch("arenaagent.vlm_agent.vlm_agent.time.sleep") as sleep_mock:
            result = agent.run_step({"task_type": "tidyroom", "subject": "整理房间"}, {})

        self.assertEqual(result["action"], "turn_in_degree")
        self.assertEqual(result["result"], "success")
        self.assertEqual(sleep_mock.call_args_list, [call(0.75)])
        self.assertEqual(len(agent.semantic_mapper.calls), 1)
        self.assertTrue(agent.semantic_mapper.calls[0].get("include_images", True))
        self.assertEqual(agent.vlm_client.calls, 0)
        self.assertFalse(agent._task_strategy._second_frame_vlm_pending)
        self.assertEqual(agent._task_strategy.scanner.turns_completed, 0)

    def test_tidyroom_unified_readiness_accepts_small_valid_view_but_rejects_black(self) -> None:
        consistent, reason, _, threshold = PreliminaryBaselineAgent._tidyroom_perception_is_consistent(
            {
                "source": "unified",
                "visible_object_count": 10,
                "image_present": True,
                "right_nonblack_ratio": 0.75,
                "left_right_difference_ratio": 0.75,
            }
        )
        self.assertTrue(consistent)
        self.assertEqual(reason, "unified_perception_ready")
        self.assertEqual(threshold, 1)

        consistent, reason, _, _ = PreliminaryBaselineAgent._tidyroom_perception_is_consistent(
            {
                "source": "unified",
                "visible_object_count": 36,
                "image_present": True,
                "right_nonblack_ratio": 0.0,
                "left_right_difference_ratio": 0.75,
            }
        )
        self.assertFalse(consistent)
        self.assertEqual(reason, "unified_segmentation_black")

        consistent, reason, _, _ = PreliminaryBaselineAgent._tidyroom_perception_is_consistent(
            {
                "source": "unified",
                "visible_object_count": 3,
                "image_present": True,
                "right_nonblack_ratio": 0.99,
                "left_right_difference_ratio": 0.02,
            }
        )
        self.assertFalse(consistent)
        self.assertEqual(reason, "unified_segmentation_still_rgb")

    def test_tidyroom_first_nonempty_scene_waits_without_changing_heading(self) -> None:
        lightweight_details = [{"object_id": "1", "shape": "unrelated-chair"}]

        class NoPairMapper:
            def __init__(self) -> None:
                self.calls: list[dict] = []
                self.object_id_map = {"raw-chair": 1}

            def get_perception_from_camera(self, **kwargs):
                self.calls.append(dict(kwargs))
                return None, [{"object_id": 1}], lightweight_details

        class NoInvokeClient:
            def __init__(self) -> None:
                self.calls = 0

            def invoke(self, messages):
                del messages
                self.calls += 1
                raise AssertionError("first frame must not call VLM")

        class NoPairAgent(PreliminaryBaselineAgent):
            def _do_action(self, action):
                return {"result": "success", "action": action["action"]}

        raw_target = "BP_Pillow_10_TEST"
        agent = NoPairAgent(stub=None, channel=None)
        agent._initialized = True
        agent.semantic_mapper = NoPairMapper()
        agent.vlm_client = NoInvokeClient()

        with patch("arenaagent.vlm_agent.vlm_agent.time.sleep") as sleep_mock:
            result = agent.run_step(
                {"task_type": "tidyroom", "subject": "整理房间", "movable_object_id": [raw_target]},
                {},
            )

        self.assertEqual(result["action"], "turn_in_degree")
        self.assertEqual(sleep_mock.call_args_list, [call(0.75)])
        self.assertEqual(len(agent.semantic_mapper.calls), 1)
        self.assertTrue(agent.semantic_mapper.calls[0]["include_images"])
        self.assertEqual(agent.vlm_client.calls, 0)
        self.assertFalse(agent._task_strategy._second_frame_vlm_pending)
        self.assertEqual(agent._task_strategy.scanner.turns_completed, 0)

    def test_tidyroom_second_frame_diagnostic_and_startup_scan_are_disabled(self) -> None:
        strategy = TidyRoomStrategy()
        strategy.reset({"task_type": "tidyroom", "subject": "整理房间"})
        strategy._second_frame_vlm_pending = True

        self.assertFalse(strategy.should_force_second_frame_vlm())
        self.assertTrue(strategy.scanner.complete)
        self.assertIsNone(strategy.scanner.next_action())
        self.assertEqual(strategy.scanner.turns_required, 0)

    def test_tidyroom_without_a_target_list_skips_the_second_frame_vlm(self) -> None:
        """912 不下发清单时，第二帧问模型只会得到"继续转"，白白花掉一分多钟。"""

        class QuietMapper:
            def __init__(self) -> None:
                self.calls: list[dict] = []
                self.object_id_map = {}

            def get_perception_from_camera(self, **kwargs):
                self.calls.append(dict(kwargs))
                return None, [], []

        class CountingClient:
            def __init__(self) -> None:
                self.calls = 0

            def invoke(self, messages):
                del messages
                self.calls += 1
                return type("FakeResponse", (), {"text": "[]", "token_usage": None})()

        class QuietAgent(PreliminaryBaselineAgent):
            def _do_action(self, action):
                return {"result": "success", "action": action["action"]}

        agent = QuietAgent(stub=None, channel=None)
        agent._initialized = True
        agent.semantic_mapper = QuietMapper()
        agent.vlm_client = CountingClient()

        first_result = agent.run_step({"task_type": "tidyroom", "subject": "整理房间"}, {})
        second_result = agent.run_step({"task_type": "tidyroom", "subject": "整理房间"}, {})

        self.assertEqual(first_result["action"], "turn_in_degree")
        self.assertEqual(second_result["action"], "turn_in_degree")
        self.assertEqual(agent.vlm_client.calls, 0)
        self.assertEqual(agent._task_strategy.vlm_call_count, 0)
        # 清单缺失时不再消耗这次诊断，但标记保持不变，等扫描完再一次性补充语义。
        self.assertFalse(agent._task_strategy._second_frame_vlm_pending)
        self.assertEqual(agent._task_strategy.scanner.turns_completed, 0)

    def test_tidyroom_fast_loop_caches_task_service_data(self) -> None:
        agent = PreliminaryBaselineAgent(stub=None, channel=None)
        agent.sleep_between_steps = 0
        calls = {"subject": 0, "response": 0, "steps": 0, "apply": 0, "evaluate": 0}

        def get_subject():
            calls["subject"] += 1
            return {"task_type": "tidyroom", "subject": "整理房间"}

        def get_response():
            calls["response"] += 1
            return {}

        def run_step(subject, response):
            del subject, response
            calls["steps"] += 1
            agent._last_executed_action_name = (
                "move_and_take_object" if calls["steps"] == 1 else "submit_answer"
            )
            return {"result": "success"} if calls["steps"] == 1 else {"done": "0"}

        def apply_action(result):
            del result
            calls["apply"] += 1
            return {}

        def evaluate_subject():
            calls["evaluate"] += 1
            return {}

        agent._get_subject_from_task = get_subject
        agent._get_response_from_task = get_response
        agent.run_step = run_step
        agent._apply_action = apply_action
        agent._evaluate_subject = evaluate_subject
        agent._run_subject()

        self.assertEqual(calls, {"subject": 1, "response": 1, "steps": 2, "apply": 2, "evaluate": 1})

    def test_same_task_type_still_resets_for_a_new_subject(self) -> None:
        agent = PreliminaryBaselineAgent(stub=None, channel=None)
        first = agent._ensure_task_strategy({"task_type": "counting", "subject": "数杯子"})
        second = agent._ensure_task_strategy({"task_type": "counting", "subject": "数鞋子"})
        self.assertIsNot(first, second)

    def test_prompt_variables_are_enriched_by_selected_task(self) -> None:
        agent = PreliminaryBaselineAgent(stub=None, channel=None)
        agent._movable_objects = ["raw-a", "raw-b"]
        variables = agent._build_prompt_variables(
            subject={"task_type": "tidyroom", "subject": "整理房间"},
            task_response={},
            api_info={"actions": []},
            visible_objects_info=[{"object_id": "1"}],
            object_in_hand=None,
        )
        self.assertIn("整理房间任务", variables["task_prompt"])
        self.assertEqual(variables["task_strategy_state"]["target_count"], 2)
        self.assertEqual(variables["movable_object_count"], 2)

    def test_vlm_escalation_reason_is_refreshed_before_request(self) -> None:
        raw_id = "BP_UnknownMovable_TEST"
        agent = PreliminaryBaselineAgent(stub=None, channel=None)
        strategy = agent._ensure_task_strategy(
            {"task_type": "tidyroom", "subject": "整理", "movable_object_id": [raw_id]}
        )
        strategy.scanner.turns_completed = strategy.scanner.turns_required
        context = TaskContext(
            task_type="tidyroom",
            subject={},
            task_response={},
            visible_objects=[],
            object_in_hand=None,
            movable_objects=[raw_id],
            action_histories=[],
        )
        strategy.observe(context)
        agent._task_context = context
        prompt_variables = {"task_prompt": ""}
        action = agent._get_local_action({}, {}, prompt_variables)
        self.assertIsNone(action)
        self.assertEqual(prompt_variables["task_strategy_state"]["vlm_reason"], "unknown_target_category")
        self.assertEqual(prompt_variables["task_strategy_state"]["vlm_call_count"], 1)

    def test_second_scene_vlm_turn_does_not_reset_search_budget(self) -> None:
        strategy = TidyRoomStrategy()
        strategy.reset({"movable_object_id": []})
        strategy.local_search_turns = 2
        strategy.local_search_reason = "target_not_observed"
        strategy.note_forced_vlm("second_scene_diagnostic")

        action = {"action": "turn_in_degree", "parameters": {"degree": 90}, "output": 0}
        validated = strategy.validate_action(action, None)

        self.assertEqual(validated, action)
        self.assertEqual(strategy.local_search_turns, 2)
        self.assertEqual(strategy.vlm_call_count, 1)
        self.assertIsNone(strategy.vlm_reason)

    def test_legacy_second_scene_reason_no_longer_rewrites_turn_angle(self) -> None:
        strategy = TidyRoomStrategy()
        strategy.reset({"movable_object_id": []})
        strategy.note_forced_vlm("second_scene_diagnostic")

        action = {
            "action": "turn_in_degree",
            "parameters": {"degree": 15},
            "output": 0,
            "think": "换一个方向观察。",
        }
        validated = strategy.validate_action(action, None)

        self.assertEqual(validated["parameters"]["degree"], 15)
        self.assertEqual(validated["think"], "换一个方向观察。")
        self.assertEqual(action["parameters"]["degree"], 15)
        self.assertEqual(strategy.scanner.turns_completed, 0)

    def test_tidyroom_missing_target_does_not_trigger_a_search_turn(self) -> None:
        raw_id = "BP_DrinkContainer_Can_07_TEST"
        strategy = TidyRoomStrategy()
        strategy.reset({"movable_object_id": [raw_id]})
        strategy.scanner.turns_completed = strategy.scanner.turns_required
        context = TaskContext(
            task_type="tidyroom",
            subject={},
            task_response={},
            visible_objects=[],
            object_in_hand=None,
            movable_objects=[raw_id],
            action_histories=[],
        )
        strategy.observe(context)

        action = strategy.next_local_action(context)

        self.assertIsNone(action)
        self.assertEqual(strategy.state_for_prompt()["vlm_call_count"], 1)
        self.assertEqual(strategy.state_for_prompt()["local_search"]["turns_completed"], 0)

    def test_tidyroom_local_search_is_disabled_in_single_frame_mode(self) -> None:
        strategy = TidyRoomStrategy()
        strategy.reset({"movable_object_id": ["BP_Fruit_Apple_01_TEST"]})
        strategy.scanner.turns_completed = strategy.scanner.turns_required

        first = strategy._next_local_search_action("target_not_observed")
        second = strategy._next_local_search_action("missing_destination:dining_table")

        self.assertIsNone(first)
        self.assertIsNone(second)
        self.assertEqual(strategy.local_search_turns, 0)

    def test_tidyroom_held_item_is_not_interrupted_by_an_unobserved_target(self) -> None:
        held_raw_id = "BP_Garbage_Bag_TEST"
        missing_raw_id = "BP_Fruit_Apple_01_TEST"
        trash_raw_id = "BP_TrashBin_01_TEST"
        target_info = {
            "object_id": "3",
            "place_location": {"X": 517.0, "Y": 234.0, "Z": 2.5},
            "world_aabb": {
                "min": {"x": 513.3, "y": 230.2, "z": 2.5},
                "max": {"x": 520.7, "y": 237.8, "z": 18.2},
            },
        }
        trash_info = {
            "object_id": "8",
            "place_location": {"X": 805.0, "Y": 276.0, "Z": 3.0},
            "world_aabb": {
                "min": {"x": 800.4, "y": 253.3, "z": 3.9},
                "max": {"x": 827.5, "y": 280.5, "z": 38.8},
            },
        }
        strategy = TidyRoomStrategy()
        strategy.reset({"movable_object_id": [held_raw_id, missing_raw_id]})
        strategy.scanner.turns_completed = strategy.scanner.turns_required
        context = TaskContext(
            task_type="tidyroom",
            subject={},
            task_response={},
            visible_objects=[target_info, trash_info],
            object_in_hand=(held_raw_id, 0),
            movable_objects=[held_raw_id, missing_raw_id],
            action_histories=[],
            raw_to_mapped_id={held_raw_id: "3", trash_raw_id: "8"},
        )
        strategy.observe(context)

        action = strategy.next_local_action(context)

        self.assertEqual(action["action"], "put_down_sth_to_location")
        self.assertTrue(action["parameters"]["force_locate"])
        self.assertEqual(strategy.state_for_prompt()["vlm_call_count"], 0)

    def test_tidyroom_state_is_updated_after_successful_actions(self) -> None:
        target_raw_id = "BP_Garbage_Bag_TEST"
        trash_raw_id = "BP_TrashBin_01_TEST"
        target_info = {
            "object_id": "3",
            "place_location": {"X": 629.0, "Z": 2.498, "Y": 368.0},
            "world_aabb": {
                "min": {"x": 625.3, "y": 364.2, "z": 2.5},
                "max": {"x": 632.7, "y": 371.8, "z": 18.18},
            },
        }
        trash_info = {
            "object_id": "8",
            "place_location": {"X": 805.0, "Z": 3.0, "Y": 276.0},
            "world_aabb": {
                "min": {"x": 800.39, "y": 253.29, "z": 3.88},
                "max": {"x": 827.52, "y": 280.55, "z": 38.82},
            },
        }
        floor_info = {
            "object_id": "30",
            "world_aabb": {
                "min": {"x": 79.5126, "y": -174.0, "z": -2.5},
                "max": {"x": 852.4874, "y": 654.9581, "z": 2.5},
            },
        }
        tv_stand_info = {
            "object_id": "17",
            "world_aabb": {
                "min": {"x": 792.71, "y": 292.50, "z": -0.22},
                "max": {"x": 835.19, "y": 533.10, "z": 37.22},
            },
        }
        strategy = TidyRoomStrategy()
        strategy.reset({"movable_object_id": [target_raw_id]})
        context = TaskContext(
            task_type="tidyroom",
            subject={},
            task_response={},
            visible_objects=[target_info, trash_info, floor_info, tv_stand_info],
            object_in_hand=None,
            movable_objects=[target_raw_id],
            action_histories=[],
            raw_to_mapped_id={target_raw_id: "3", trash_raw_id: "8"},
        )
        strategy.observe(context)
        strategy.scanner.turns_completed = strategy.scanner.turns_required
        take_action = strategy.next_local_action(context)
        self.assertIsNotNone(take_action)
        self.assertEqual(take_action["action"], "move_and_take_object")
        self.assertEqual(strategy.state_for_prompt()["vlm_call_count"], 0)
        strategy.after_action(take_action, {"result": "success"}, context)
        self.assertEqual(strategy.state_for_prompt()["successful_takes"], 0)

        held_context = TaskContext(
            task_type="tidyroom",
            subject={},
            task_response={},
            visible_objects=[target_info, trash_info, floor_info, tv_stand_info],
            object_in_hand=(target_raw_id, 0),
            movable_objects=[target_raw_id],
            action_histories=[],
            raw_to_mapped_id={target_raw_id: "3", trash_raw_id: "8"},
        )
        strategy.observe(held_context)
        put_action = strategy.validate_action(
            {"action": "move_and_put_down", "parameters": {}, "output": 0}, held_context
        )
        self.assertEqual(put_action["action"], "put_down_sth_to_location")
        self.assertTrue(put_action["parameters"]["force_locate"])
        strategy.after_action(put_action, {"result": "success"}, held_context)

        # put_down_sth_to_location 按坐标强制放置：动作返回 success 就完成，
        # 不再回读 AABB 做几何校验。
        state = strategy.state_for_prompt()
        self.assertEqual(state["successful_takes"], 1)
        self.assertEqual(state["successful_puts"], 1)
        self.assertEqual(state["targets"][0]["status"], "done")
        self.assertEqual(strategy.scheduler.estimated_agent_xy, (629.0, 368.0))
        self.assertEqual(strategy.next_local_action(held_context)["action"], "submit_answer")

    def test_tidyroom_trash_uses_direct_coordinate_placement(self) -> None:
        target_raw_id = "BP_Garbage_Bag_TEST"
        strategy = TidyRoomStrategy()
        strategy.reset({"movable_object_id": [target_raw_id]})
        strategy.targets[target_raw_id].update({"object_id": "3", "status": "held"})
        strategy.held_raw_id = target_raw_id
        strategy.active_plan = {
            "target_raw_id": target_raw_id,
            "target_object_id": "3",
            "target_category": "garbage",
            "target_info": {
                "place_location": {"X": 517.0, "Y": 234.0, "Z": 2.5},
                "world_aabb": {
                    "min": {"x": 513.3, "y": 230.2, "z": 2.5},
                    "max": {"x": 520.7, "y": 237.8, "z": 18.2},
                },
            },
            "destination_raw_id": "BP_TrashBin_01_TEST",
            "destination_object_id": "8",
            "destination_type": "trash_bin",
            "destination_info": {
                "world_aabb": {
                    "min": {"x": 800.4, "y": 253.3, "z": 3.9},
                    "max": {"x": 827.5, "y": 280.5, "z": 38.8},
                }
            },
            "attempt": 0,
            "slot_index": 0,
            "last_failure": None,
        }
        strategy.planner.refresh_plan(strategy.active_plan, strategy.scene_anchors)

        first_put = strategy._put_action(strategy.active_plan, "测试容器动作")
        # 四类目的地统一使用 force_locate 坐标放置。
        self.assertEqual(first_put["action"], "put_down_sth_to_location")
        self.assertIn("target_location", first_put["parameters"])
        self.assertTrue(first_put["parameters"]["force_locate"])

    def test_tidyroom_computes_dining_table_coordinates_by_field_name(self) -> None:
        apple_raw_id = "BP_Fruit_Apple_01_TEST"
        table_raw_id = "BP_DiningTable_01_TEST"
        apple_info = {
            "object_id": "35",
            "place_location": {"X": 625.829, "Z": 8.666, "Y": 326.657},
            "world_aabb": {
                "min": {"x": 623.58, "y": 316.03, "z": -5.17},
                "max": {"x": 642.07, "y": 333.89, "z": 14.57},
            },
        }
        table_info = {
            "object_id": "16",
            "place_location": {"X": -217.0, "Z": 3.0, "Y": 521.0},
            "world_aabb": {
                "min": {"x": -297.0, "y": 481.0, "z": 3.0},
                "max": {"x": -137.0, "y": 561.0, "z": 78.757},
            },
        }
        floor_info = {
            "object_id": "31",
            "world_aabb": {
                "min": {"x": -370.0, "y": 348.0, "z": -2.5},
                "max": {"x": 80.0, "y": 651.298, "z": 2.5},
            },
        }
        strategy = TidyRoomStrategy()
        strategy.reset({"movable_object_id": [apple_raw_id]})
        context = TaskContext(
            task_type="tidyroom",
            subject={},
            task_response={},
            visible_objects=[apple_info, table_info, floor_info],
            object_in_hand=None,
            movable_objects=[apple_raw_id],
            action_histories=[],
            raw_to_mapped_id={apple_raw_id: "35", table_raw_id: "16"},
        )
        strategy.observe(context)
        take_action = strategy.validate_action(
            {"action": "move_and_take_object", "parameters": {"object_id": "35"}}, context
        )
        strategy.after_action(take_action, {"result": "success"}, context)
        context.object_in_hand = (apple_raw_id, 0)
        strategy.observe(context)
        direct_put_action = strategy.validate_action(
            {"action": "move_and_put_down", "parameters": {}, "output": 0}, context
        )
        self.assertEqual(direct_put_action["action"], "put_down_sth_to_location")
        put_location = direct_put_action["parameters"]["target_location"]
        self.assertEqual(put_location["X"], -217.0)
        self.assertEqual(put_location["Y"], 521.0)
        self.assertAlmostEqual(put_location["Z"], 80.757, places=3)

        # 失败后更换规划槽位，再次调用同一个受支持接口。
        strategy.after_action(direct_put_action, {"result": "failed"}, context)
        fallback_action = strategy._put_action(strategy.active_plan, "直接放置失败后的回退")
        self.assertEqual(fallback_action["action"], "put_down_sth_to_location")
        self.assertTrue(fallback_action["parameters"]["force_locate"])
        self.assertIn("target_location", fallback_action["parameters"])

    def test_dining_table_height_does_not_depend_on_moved_target_aabb(self) -> None:
        planner = TidyRoomPlanner()
        destination_aabb = (
            {"x": -297.0, "y": 481.0, "z": 3.0},
            {"x": -137.0, "y": 561.0, "z": 78.757},
        )
        plan = {
            "destination_type": "dining_table",
            "target_category": "drink_container",
            "target_info": {"place_location": {"X": 500.0, "Y": 200.0, "Z": 12.0}},
        }

        original_height = planner._put_origin_z(
            plan,
            ({"x": 490.0, "y": 190.0, "z": -4.0}, {"x": 510.0, "y": 210.0, "z": 20.0}),
            78.757,
            destination_aabb,
        )
        moved_height = planner._put_origin_z(
            plan,
            ({"x": -260.0, "y": 510.0, "z": 96.0}, {"x": -240.0, "y": 530.0, "z": 120.0}),
            78.757,
            destination_aabb,
        )

        self.assertAlmostEqual(original_height, 80.757, places=3)
        self.assertAlmostEqual(moved_height, 80.757, places=3)

    def test_tidyroom_retries_when_placement_is_outside_destination(self) -> None:
        result = placement_check(
            {"min": {"x": 700.0, "y": 200.0, "z": 3.0}, "max": {"x": 708.0, "y": 208.0, "z": 20.0}},
            {
                "min": {"x": 800.0, "y": 253.0, "z": 3.0},
                "max": {"x": 828.0, "y": 281.0, "z": 39.0},
            },
            {"destination_type": "trash_bin"},
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "outside_xy")

    def test_tidyroom_rejects_trash_suspended_near_bin_opening(self) -> None:
        result = placement_check(
            {
                "min": {"x": 810.0, "y": 260.0, "z": 21.0},
                "max": {"x": 817.0, "y": 267.0, "z": 36.8},
            },
            {
                "min": {"x": 800.0, "y": 253.0, "z": 3.8},
                "max": {"x": 828.0, "y": 281.0, "z": 38.8},
            },
            {"destination_type": "trash_bin"},
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "not_resting_in_container")

    def test_tidyroom_geometric_table_prefers_the_one_with_a_chair(self) -> None:
        """标注不可信时按几何挑餐桌：旁边有椅子的那件才是餐桌。

        实测：模型被要求"不要把椅子标成餐桌"，就把扶手椅标成了餐桌，物品穿过
        "桌面"落到座面上。只按面积挑并不可靠（扶手椅 150x120 比餐桌 160x80 还
        大），椅子是稳定得多的信号。
        """
        world = TidyRoomWorldModel({})
        world.scene_objects = {
            "12": {
                "object_id": "12",
                "shape": "rectangle",
                "world_aabb": {
                    "min": {"x": 486.0, "y": 530.0, "z": -0.5},
                    "max": {"x": 636.0, "y": 650.0, "z": 83.4},
                },
            },
            "16": {
                "object_id": "16",
                "shape": "rectangle",
                "world_aabb": {
                    "min": {"x": -297.0, "y": 481.0, "z": 3.0},
                    "max": {"x": -137.0, "y": 561.0, "z": 78.757},
                },
            },
        }
        # 没有椅子信息时只能按面积挑：扶手椅更大就被选中。
        self.assertEqual(world.geometric_candidates("dining_table")[0], "12")

        world.scene_objects["21"] = {
            "object_id": "21",
            "shape": "chair",
            "world_aabb": {
                "min": {"x": -281.0, "y": 405.0, "z": 0.0},
                "max": {"x": -195.0, "y": 485.0, "z": 80.1},
            },
        }
        # 看到旁边的椅子之后，真正的餐桌排到前面。
        self.assertEqual(world.geometric_candidates("dining_table")[0], "16")
        self.assertEqual(world.promote_geometric_candidate("dining_table"), "16")
        self.assertEqual(world.scene_anchors["16"]["type"], "dining_table")
        self.assertEqual(world.scene_anchors["16"]["source"], "geometric_candidate")

    def test_tidyroom_promotes_geometric_table_before_scheduler_gives_up(self) -> None:
        """无餐桌 anchor 时先补建候选，不能让调度器返回空后直接 blocked。"""
        strategy = TidyRoomStrategy()
        strategy.reset({"task_type": "tidyroom", "subject": "整理房间"})
        context = TaskContext(
            task_type="tidyroom",
            subject={},
            task_response={},
            visible_objects=[
                {
                    "object_id": "32",
                    "world_aabb": {
                        "min": {"x": 500.0, "y": 220.0, "z": 2.0},
                        "max": {"x": 508.0, "y": 228.0, "z": 18.0},
                    },
                },
                {
                    "object_id": "16",
                    "shape": "rectangle",
                    "world_aabb": {
                        "min": {"x": -297.0, "y": 481.0, "z": 3.0},
                        "max": {"x": -137.0, "y": 561.0, "z": 78.757},
                    },
                },
                {
                    "object_id": "21",
                    "shape": "chair",
                    "world_aabb": {
                        "min": {"x": -281.0, "y": 405.0, "z": 0.0},
                        "max": {"x": -195.0, "y": 485.0, "z": 80.1},
                    },
                },
            ],
            object_in_hand=None,
            movable_objects=[],
            action_histories=[],
            raw_to_mapped_id={"32": "32", "unknown16": "16", "chair": "21"},
        )
        strategy.observe(context)
        strategy.world.apply_semantic_hints(
            {"object_id": "32", "semantic_label": "cup", "destination_type": "dining_table"},
            context,
        )

        action = strategy.next_local_action(context)

        self.assertEqual(action["action"], "move_and_take_object")
        self.assertEqual(action["parameters"]["object_id"], "32")
        self.assertEqual(strategy.active_plan["destination_object_id"], "16")
        self.assertEqual(strategy.scene_anchors["16"]["source"], "geometric_candidate")

    def test_tidyroom_places_bolster_on_main_sofa_seat_instead_of_ottoman(self) -> None:
        pillow_raw_id = "BP_Pillow_10_TEST"
        raw_to_mapped_id = {
            pillow_raw_id: "33",
            "BP_Sofa_Armchair_TEST": "12",
            "BP_Sofa_Ottoman_TEST": "13",
            "BP_Sofa_Main_TEST": "14",
            "BP_CoffeeTable_TEST": "15",
        }
        visible_objects = [
            {
                "object_id": "33",
                "place_location": {"X": 446.0, "Y": 212.0, "Z": 9.827},
                "world_aabb": {
                    "min": {"x": 423.36, "y": 203.88, "z": 2.5},
                    "max": {"x": 468.66, "y": 220.12, "z": 17.15},
                },
            },
            {
                "object_id": "12",
                "world_aabb": {
                    "min": {"x": 486.14, "y": 539.61, "z": -0.52},
                    "max": {"x": 597.74, "y": 630.06, "z": 83.42},
                },
            },
            {
                "object_id": "13",
                "world_aabb": {
                    "min": {"x": 350.51, "y": 175.45, "z": 1.67},
                    "max": {"x": 409.49, "y": 250.55, "z": 28.60},
                },
            },
            {
                "object_id": "14",
                "world_aabb": {
                    "min": {"x": 226.524, "y": 169.114, "z": -0.482},
                    "max": {"x": 338.406, "y": 579.456, "z": 98.927},
                },
            },
            {
                "object_id": "15",
                "world_aabb": {
                    "min": {"x": 541.224, "y": 296.539, "z": 15.961},
                    "max": {"x": 613.150, "y": 433.499, "z": 45.034},
                },
            },
            {
                "object_id": "30",
                "world_aabb": {
                    "min": {"x": 79.513, "y": -174.0, "z": -2.5},
                    "max": {"x": 852.487, "y": 654.958, "z": 2.5},
                },
            },
        ]
        strategy = TidyRoomStrategy()
        strategy.reset({"movable_object_id": [pillow_raw_id]})
        context = TaskContext(
            task_type="tidyroom",
            subject={},
            task_response={},
            visible_objects=visible_objects,
            object_in_hand=None,
            movable_objects=[pillow_raw_id],
            action_histories=[],
            raw_to_mapped_id=raw_to_mapped_id,
        )
        strategy.observe(context)
        strategy.validate_action(
            {"action": "move_and_take_object", "parameters": {"object_id": "33"}}, context
        )
        plan = strategy.state_for_prompt()["active_plan"]
        self.assertEqual(plan["destination_object_id"], "14")
        self.assertAlmostEqual(plan["support_z"], 44.252, places=2)
        self.assertGreaterEqual(plan["put_target_location"]["X"], (226.524 + 338.406) / 2.0)

    def test_tidyroom_pickup_requires_structured_hand_confirmation(self) -> None:
        target_raw_id = "BP_DrinkContainer_Can_07_TEST"
        strategy = TidyRoomStrategy()
        strategy.reset({"movable_object_id": [target_raw_id]})
        context = TaskContext(
            task_type="tidyroom",
            subject={},
            task_response={},
            visible_objects=[{"object_id": "3"}],
            object_in_hand=None,
            movable_objects=[target_raw_id],
            action_histories=[],
            raw_to_mapped_id={target_raw_id: "3"},
        )
        strategy.observe(context)
        action = {"action": "move_and_take_object", "parameters": {"object_id": "3"}}
        strategy.after_action(action, {"result": "success"}, context)
        strategy.observe(context)
        state = strategy.state_for_prompt()
        self.assertEqual(state["successful_takes"], 0)
        self.assertEqual(state["pickup_verification_failures"], 1)
        self.assertEqual(state["targets"][0]["status"], "pending")

    def test_tidyroom_server_not_pickup_blocks_target_immediately(self) -> None:
        target_raw_id = "fixed-display-shoe"
        strategy = TidyRoomStrategy()
        strategy.reset({"movable_object_id": [target_raw_id]})
        context = TaskContext(
            task_type="tidyroom",
            subject={},
            task_response={},
            visible_objects=[{"object_id": "10"}],
            object_in_hand=None,
            movable_objects=[target_raw_id],
            action_histories=[],
            raw_to_mapped_id={target_raw_id: "10"},
        )
        strategy.observe(context)

        strategy.after_action(
            {"action": "move_and_take_object", "parameters": {"object_id": "10"}},
            {"result": "failed", "error": "can not take this object for not pickup"},
            context,
        )

        self.assertEqual(strategy.targets[target_raw_id]["status"], "blocked")
        self.assertEqual(strategy.targets[target_raw_id]["blocked_reason"], "server_not_pickup")
        self.assertEqual(strategy.recovery.total_failures, {})

    def test_action_parser_preserves_top_level_scene_annotations(self) -> None:
        agent = PreliminaryBaselineAgent(stub=None, channel=None)
        agent._task_strategy = TidyRoomStrategy()
        agent._task_strategy.reset({"task_type": "tidyroom", "subject": "整理房间"})
        raw_id = "33"
        context = TaskContext(
            task_type="tidyroom",
            subject={},
            task_response={},
            visible_objects=[{"object_id": "33"}],
            object_in_hand=None,
            movable_objects=[],
            action_histories=[],
            raw_to_mapped_id={raw_id: "33"},
        )
        agent._task_context = context
        agent._task_strategy.observe(context)

        parsed = agent._parse_action_from_response(
            [
                {
                    "action": "move_and_take_object",
                    "parameters": {"object_id": "33", "which_hand": 0},
                    "scene_annotations": [
                        {
                            "object_id": "33",
                            "semantic_label": "cup",
                            "destination_type": "dining_table",
                        }
                    ],
                    "output": 0,
                }
            ]
        )

        self.assertIn("scene_annotations", parsed)
        self.assertIn(raw_id, agent._task_strategy.targets)
        self.assertEqual(agent._task_strategy.targets[raw_id]["destination_type"], "dining_table")

    def test_tidyroom_scans_locally_before_considering_vlm(self) -> None:
        strategy = TidyRoomStrategy()
        strategy.reset({"movable_object_id": ["BP_Pillow_TEST"]})
        context = TaskContext(
            task_type="tidyroom",
            subject={},
            task_response={},
            visible_objects=[],
            object_in_hand=None,
            movable_objects=[],
            action_histories=[],
        )
        strategy.observe(context)
        for _ in range(strategy.scanner.turns_required):
            action = strategy.next_local_action(context)
            self.assertEqual(action["action"], "turn_in_degree")
            strategy.after_action(action, {"result": "success"}, context)
        self.assertTrue(strategy.scanner.complete)
        self.assertEqual(strategy.state_for_prompt()["vlm_call_count"], 0)

    def test_tidyroom_scan_stops_when_targets_and_destinations_are_mapped(self) -> None:
        pillow_raw_id = "BP_Pillow_10_TEST"
        apple_raw_id = "BP_Fruit_Apple_01_TEST"
        sofa_raw_id = "BP_Sofa_Main_TEST"
        table_raw_id = "BP_DiningTable_01_TEST"
        strategy = TidyRoomStrategy()
        strategy.reset({"movable_object_id": [pillow_raw_id, apple_raw_id]})
        visible_objects = [
            {"object_id": "1", "world_aabb": {"min": {"x": 0, "y": 0, "z": 0}, "max": {"x": 10, "y": 10, "z": 10}}},
            {"object_id": "2", "world_aabb": {"min": {"x": 20, "y": 0, "z": 0}, "max": {"x": 30, "y": 10, "z": 10}}},
            {"object_id": "3", "world_aabb": {"min": {"x": 40, "y": 0, "z": 0}, "max": {"x": 80, "y": 40, "z": 40}}},
            {"object_id": "4", "world_aabb": {"min": {"x": 90, "y": 0, "z": 0}, "max": {"x": 150, "y": 50, "z": 70}}},
        ]
        context = TaskContext(
            task_type="tidyroom",
            subject={},
            task_response={},
            visible_objects=visible_objects,
            object_in_hand=None,
            movable_objects=[pillow_raw_id, apple_raw_id],
            action_histories=[],
            raw_to_mapped_id={
                pillow_raw_id: "1",
                apple_raw_id: "2",
                sofa_raw_id: "3",
                table_raw_id: "4",
            },
        )

        strategy.observe(context)

        scanner_state = strategy.state_for_prompt()["scanner"]
        self.assertTrue(scanner_state["complete"])
        self.assertFalse(scanner_state["finished_early"])
        self.assertEqual(scanner_state["turns_completed"], 0)
        self.assertEqual(scanner_state["turn_degrees"], 90)
        self.assertIsNone(scanner_state["early_stop_reason"])
        self.assertFalse(strategy.needs_scene_perception())

    def test_tidyroom_processes_actionable_current_view_before_full_scan(self) -> None:
        """另一个目标尚未发现时，也应先处理当前视野中可完整规划的物体。"""
        apple_raw_id = "BP_Fruit_Apple_01_TEST"
        pillow_raw_id = "BP_Pillow_10_NOT_YET_VISIBLE"
        table_raw_id = "BP_DiningTable_01_TEST"
        strategy = TidyRoomStrategy()
        strategy.reset({"movable_object_id": [apple_raw_id, pillow_raw_id]})
        strategy.before_step({}, {})
        context = TaskContext(
            task_type="tidyroom",
            subject={},
            task_response={},
            visible_objects=[
                {
                    "object_id": "2",
                    "world_aabb": {
                        "min": {"x": 20, "y": 0, "z": 0},
                        "max": {"x": 30, "y": 10, "z": 10},
                    },
                },
                {
                    "object_id": "4",
                    "world_aabb": {
                        "min": {"x": 90, "y": 0, "z": 0},
                        "max": {"x": 150, "y": 50, "z": 70},
                    },
                },
            ],
            object_in_hand=None,
            movable_objects=[apple_raw_id],
            action_histories=[],
            raw_to_mapped_id={apple_raw_id: "2", table_raw_id: "4"},
        )

        strategy.observe(context)
        self.assertTrue(strategy.scanner.complete)

        action = strategy.next_local_action(context)

        self.assertEqual(action["action"], "move_and_take_object")
        self.assertEqual(action["parameters"]["object_id"], "2")
        self.assertEqual(strategy.scanner.turns_completed, 0)
        self.assertEqual(strategy.vlm_call_count, 0)

    def test_tidyroom_scheduler_uses_nearest_neighbor_for_direct_table_items(self) -> None:
        near_raw_id = "BP_DrinkContainer_Can_03_NEAR"
        far_raw_id = "BP_DrinkContainer_Can_07_FAR"
        world = TidyRoomWorldModel({"movable_object_id": [far_raw_id, near_raw_id]})
        world.targets[near_raw_id].update(
            {
                "object_id": "1",
                "object_info": {
                    "world_aabb": {
                        "min": {"x": 10, "y": 0, "z": 0},
                        "max": {"x": 12, "y": 2, "z": 10},
                    }
                },
            }
        )
        world.targets[far_raw_id].update(
            {
                "object_id": "2",
                "object_info": {
                    "world_aabb": {
                        "min": {"x": 100, "y": 0, "z": 0},
                        "max": {"x": 102, "y": 2, "z": 10},
                    }
                },
            }
        )
        world.scene_anchors["3"] = {
            "raw_id": "BP_DiningTable_TEST",
            "object_id": "3",
            "type": "dining_table",
            "object_info": {
                "world_aabb": {
                    "min": {"x": 200, "y": 0, "z": 0},
                    "max": {"x": 260, "y": 60, "z": 70},
                }
            },
        }
        scheduler = TargetScheduler()
        scheduler.estimated_agent_xy = (0.0, 0.0)

        chosen = scheduler.choose(
            world,
            TidyRoomPlanner(),
            {},
            frozenset({"dining_table"}),
        )

        self.assertEqual(chosen, near_raw_id)

    def test_tidyroom_scheduler_looks_ahead_over_complete_remaining_route(self) -> None:
        raw_ids = [
            "BP_DrinkContainer_Can_A",
            "BP_DrinkContainer_Can_B",
            "BP_DrinkContainer_Can_C",
        ]
        centers = [(4.0, 5.0), (10.0, 2.0), (-4.0, -7.0)]
        world = TidyRoomWorldModel({"movable_object_id": raw_ids})
        for index, (raw_id, (x, y)) in enumerate(zip(raw_ids, centers), start=1):
            world.targets[raw_id].update(
                {
                    "object_id": str(index),
                    "object_info": {
                        "world_aabb": {
                            "min": {"x": x - 1, "y": y - 1, "z": 0},
                            "max": {"x": x + 1, "y": y + 1, "z": 10},
                        }
                    },
                }
            )
        world.scene_anchors["4"] = {
            "raw_id": "BP_DiningTable_TEST",
            "object_id": "4",
            "type": "dining_table",
            "object_info": {
                "world_aabb": {
                    "min": {"x": 20, "y": 20, "z": 0},
                    "max": {"x": 30, "y": 30, "z": 70},
                }
            },
        }
        scheduler = TargetScheduler()
        scheduler.estimated_agent_xy = (0.0, 0.0)

        chosen = scheduler.choose(
            world,
            TidyRoomPlanner(),
            {},
            frozenset({"dining_table"}),
        )

        # A 是离出生点最近的物体，但 C -> A -> B 的全局总路程更短。
        self.assertEqual(chosen, raw_ids[2])

    def test_tidyroom_escalates_only_after_local_recovery_threshold(self) -> None:
        recovery = RecoveryPolicy(vlm_threshold=3)
        raw_id = "BP_Pillow_TEST"
        recovery.record_failure(raw_id, "outside_xy")
        recovery.record_failure(raw_id, "outside_xy")
        self.assertFalse(recovery.needs_vlm(raw_id))
        recovery.record_failure(raw_id, "outside_xy")
        self.assertTrue(recovery.needs_vlm(raw_id))
        recovery.mark_vlm_consulted(raw_id)
        self.assertFalse(recovery.needs_vlm(raw_id))
        recovery.record_failure(raw_id, "outside_xy")
        recovery.record_failure(raw_id, "outside_xy")
        self.assertFalse(recovery.needs_vlm(raw_id))
        recovery.record_failure(raw_id, "outside_xy")
        self.assertTrue(recovery.needs_vlm(raw_id))

    def test_tidyroom_tracks_declared_targets_before_they_are_visible(self) -> None:
        raw_id = "BP_DrinkContainer_Can_07_TEST"
        strategy = TidyRoomStrategy()
        strategy.reset({"movable_object_id": [raw_id]})
        self.assertIn(raw_id, strategy.tracked_raw_ids())

    def test_tidyroom_absorbs_vlm_semantics_but_plans_coordinates_locally(self) -> None:
        target_raw_id = "BP_UnknownMovable_TEST"
        destination_raw_id = "BP_UnknownFurniture_TEST"
        strategy = TidyRoomStrategy()
        strategy.reset({"movable_object_id": [target_raw_id]})
        context = TaskContext(
            task_type="tidyroom",
            subject={},
            task_response={},
            visible_objects=[
                {
                    "object_id": "33",
                    "place_location": {"X": 446.0, "Y": 212.0, "Z": 9.0},
                    "world_aabb": {
                        "min": {"x": 423.0, "y": 203.0, "z": 2.5},
                        "max": {"x": 468.0, "y": 220.0, "z": 17.0},
                    },
                },
                {
                    "object_id": "14",
                    "world_aabb": {
                        "min": {"x": 226.0, "y": 169.0, "z": -0.5},
                        "max": {"x": 338.0, "y": 579.0, "z": 99.0},
                    },
                },
            ],
            object_in_hand=None,
            movable_objects=[target_raw_id],
            action_histories=[],
            raw_to_mapped_id={target_raw_id: "33", destination_raw_id: "14"},
        )
        strategy.observe(context)
        action = strategy.validate_action(
            {
                "action": "move_and_take_object",
                "parameters": {
                    "object_id": "33",
                    "target_category": "pillow",
                    "destination_object_id": "14",
                    "destination_type": "sofa",
                    "front_side": "+x",
                    "move_target_location": {"X": 9999, "Y": 9999, "Z": 0},
                    "scene_annotations": [
                        {"anchor_object_id": "14", "destination_type": "sofa", "front_side": "+x"}
                    ],
                },
            },
            context,
        )
        self.assertEqual(action["action"], "move_and_take_object")
        self.assertNotIn("move_target_location", action["parameters"])
        self.assertEqual(strategy.targets[target_raw_id]["category"], "pillow")
        self.assertEqual(strategy.active_plan["destination_object_id"], "14")
        self.assertEqual(strategy.active_plan["front_side"], "+x")

    def test_tidyroom_accepts_open_vlm_label_with_closed_destination(self) -> None:
        target_raw_id = "BP_PreviouslyUnseenPlate_TEST"
        table_raw_id = "BP_PreviouslyUnseenFurniture_TEST"
        world = TidyRoomWorldModel({"movable_object_id": [target_raw_id]})
        context = TaskContext(
            task_type="tidyroom",
            subject={},
            task_response={},
            visible_objects=[
                {"object_id": "7", "world_aabb": {"min": {"x": 0, "y": 0, "z": 0}, "max": {"x": 8, "y": 8, "z": 4}}},
                {"object_id": "9", "world_aabb": {"min": {"x": 100, "y": 100, "z": 0}, "max": {"x": 260, "y": 180, "z": 70}}},
            ],
            object_in_hand=None,
            movable_objects=[target_raw_id],
            action_histories=[],
            raw_to_mapped_id={target_raw_id: "7", table_raw_id: "9"},
        )
        world.observe(context)

        world.apply_scene_annotations(
            [
                {"anchor_object_id": "9", "destination_type": "dining_table"},
                {
                    "object_id": "7",
                    "semantic_label": "Ceramic Plate",
                    "destination_object_id": "9",
                    "destination_type": "dining_table",
                },
            ],
            context,
        )

        record = world.targets[target_raw_id]
        self.assertEqual(record["semantic_label"], "ceramic_plate")
        self.assertEqual(world.destination_type_for(record), "dining_table")
        self.assertEqual(world.scene_anchors["9"]["type"], "dining_table")

    def test_tidyroom_rejects_vlm_destination_outside_closed_set(self) -> None:
        target_raw_id = "BP_PreviouslyUnseenToy_TEST"
        destination_raw_id = "BP_BookShelf_TEST"
        world = TidyRoomWorldModel({"movable_object_id": [target_raw_id]})
        context = TaskContext(
            task_type="tidyroom",
            subject={},
            task_response={},
            visible_objects=[{"object_id": "7"}, {"object_id": "9"}],
            object_in_hand=None,
            movable_objects=[target_raw_id],
            action_histories=[],
            raw_to_mapped_id={target_raw_id: "7", destination_raw_id: "9"},
        )
        world.observe(context)

        world.apply_semantic_hints(
            {
                "object_id": "7",
                "semantic_label": "Collectible Toy",
                "destination_object_id": "9",
                "destination_type": "bookshelf",
            },
            context,
        )

        record = world.targets[target_raw_id]
        self.assertEqual(record["semantic_label"], "collectible_toy")
        self.assertIsNone(world.destination_type_for(record))
        self.assertNotIn("9", world.scene_anchors)

    def test_tidyroom_maps_targets_and_rewrites_furniture_navigation(self) -> None:
        pillow_raw_id = "BP_Pillow_10_TEST"
        sofa_raw_id = "BP_Sofa_01_TEST"
        strategy = TidyRoomStrategy()
        strategy.reset({"movable_object_id": [pillow_raw_id]})
        context = TaskContext(
            task_type="tidyroom",
            subject={},
            task_response={},
            visible_objects=[
                {"object_id": "13", "world_aabb": {"max": {"z": 28.0}}},
                {"object_id": "14", "world_aabb": {"max": {"z": 99.0}}},
            ],
            object_in_hand=None,
            movable_objects=[pillow_raw_id],
            action_histories=[],
            raw_to_mapped_id={pillow_raw_id: "13", sofa_raw_id: "14"},
        )
        strategy.observe(context)
        rewritten = strategy.validate_action(
            {"action": "move_to_object", "parameters": {"object_id": "14"}, "output": 0},
            context,
        )
        self.assertEqual(rewritten["action"], "move_and_take_object")
        self.assertEqual(rewritten["parameters"]["object_id"], "13")
        state = strategy.state_for_prompt()
        self.assertEqual(state["targets"][0]["category"], "pillow")
        self.assertEqual(state["scene_anchors"][0]["type"], "sofa")

    def test_tidyroom_does_not_retain_full_image_message_history(self) -> None:
        agent = PreliminaryBaselineAgent(stub=None, channel=None)
        agent._ensure_task_strategy({"task_type": "tidyroom", "subject": "整理"})
        agent._append_history_messages([{"role": "user", "content": "large image prompt"}])
        self.assertEqual(agent.history_messages, [])


if __name__ == "__main__":
    unittest.main()
