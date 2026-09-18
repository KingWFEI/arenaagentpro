"""912 统一感知路径，以及它对旧服务端的回退。"""

from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from arenaagent.preliminary_baseline_agent.preliminary_baseline_agent import PreliminaryBaselineAgent
from arenaagent.preliminary_baseline_agent.task_runtime import TaskContext
from arenaagent.preliminary_baseline_agent.tasks.tidyroom.strategy import TidyRoomStrategy
from arenaagent.preliminary_baseline_agent.tasks.tidyroom.survey import run_scene_survey
from arenaagent.preliminary_baseline_agent.tasks.tidyroom.world_model import TidyRoomWorldModel
from arenaagent.semantic_mapper import SemanticMapper
from arenaagent.tongsim_grpc_client import TongSimGrpcClient
from arenaagent.vlm_agent.json_parsor import extract_last_json_from_text
from arenaagent.vlm_agent.vlm_agent import VLMAgent


def _image_b64(size=(8, 4), color=(10, 20, 30)) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# 取自真实 912 响应的字段形状。
UNIFIED_OBJECTS = [
    {
        "object_id": "4",
        "color": "brown",
        "shape": "rectangle",
        "place_location": {"X": 180.0, "Y": -566.0, "Z": 2.0},
        "rotation": {"roll": 0.0, "yaw": 90.0, "pitch": 0.0},
        "world_aabb": {
            "min": {"x": 84.57, "y": -672.58, "z": 1.55},
            "max": {"x": 275.91, "y": -450.96, "z": 127.0},
        },
    },
    {
        "object_id": "7",
        "color": "red",
        "shape": "sphere",
        "place_location": {"X": 200.0, "Y": -500.0, "Z": 20.0},
        "world_aabb": {"min": {"x": 1.0, "y": 2.0, "z": 3.0}, "max": {"x": 4.0, "y": 5.0, "z": 6.0}},
    },
]


class UnifiedTongSim:
    """只实现统一感知的服务端。"""

    def __init__(self, image_b64: str) -> None:
        self.image_b64 = image_b64
        self.calls: list[tuple] = []

    def acquire_first_person_perception(self, character_id, width=None, height=None):
        self.calls.append((character_id, width, height))
        return {"image": self.image_b64, "objects": [dict(item) for item in UNIFIED_OBJECTS]}


class NoUnifiedTongSim:
    """只有旧接口的服务端，统一感知按协议抛 NotImplementedError。"""

    def __init__(self) -> None:
        self.image_b64 = _image_b64()
        self.unified_attempts = 0
        self.split_calls: list[str] = []

    def acquire_first_person_perception(self, character_id, width=None, height=None):
        self.unified_attempts += 1
        raise NotImplementedError()

    def acquire_first_person_image(self, character_id, encode_base64=False):
        self.split_calls.append("rgb")
        return self.image_b64

    def acquire_first_person_segmantic_image(self, character_id, encode_base64=False):
        self.split_calls.append("segmentation")
        return self.image_b64

    def fetch_first_person_visible_objects(self, character_id):
        self.split_calls.append("visible")
        return [{"object_id": "4"}]


class AbsentUnifiedTongSim(NoUnifiedTongSim):
    """根本不提供统一感知方法的服务端。"""

    acquire_first_person_perception = None


class UnifiedPerceptionTest(unittest.TestCase):
    def test_unified_response_is_mapped_and_surfaced(self) -> None:
        image = _image_b64()
        tongsim = UnifiedTongSim(image)
        mapper = SemanticMapper(tongsim=tongsim, character_id="agent")

        returned_image, visible, info = mapper.get_perception_from_camera()

        self.assertEqual(tongsim.calls, [("agent", None, None)])
        self.assertEqual(returned_image, image)
        # 服务端在分割图上画的就是原始 ID，提示词又要求模型只用图上的数字，
        # 因此映射必须退化为恒等，否则模型给的数字会被翻译成另一个物体。
        self.assertEqual([item["object_id"] for item in visible], [4, 7])
        self.assertEqual(mapper.get_raw_id(4), "4")
        self.assertEqual(mapper.get_raw_id(7), "7")
        self.assertEqual(info[0]["object_id"], "4")
        self.assertEqual(info[0]["color"], "brown")
        self.assertEqual(info[0]["shape"], "rectangle")
        self.assertEqual(info[0]["place_location"], {"X": 180.0, "Y": -566.0, "Z": 2.0})
        self.assertEqual(info[0]["world_aabb"]["max"]["z"], 127.0)
        self.assertEqual(info[1]["object_id"], "7")

    def test_unified_path_leaves_diagnostics_unavailable(self) -> None:
        """服务端已保证同帧，因此不再计算像素覆盖率来二次校验。"""
        mapper = SemanticMapper(tongsim=UnifiedTongSim(_image_b64()), character_id="agent")

        mapper.get_perception_from_camera()

        self.assertEqual(mapper.last_perception_diagnostics, {})

    def test_details_can_be_skipped(self) -> None:
        mapper = SemanticMapper(tongsim=UnifiedTongSim(_image_b64()), character_id="agent")

        _, visible, info = mapper.get_perception_from_camera(include_object_details=False)

        self.assertEqual(len(visible), 2)
        self.assertEqual(info, [])

    def test_image_can_be_skipped(self) -> None:
        mapper = SemanticMapper(tongsim=UnifiedTongSim(_image_b64()), character_id="agent")

        image, visible, info = mapper.get_perception_from_camera(include_images=False)

        self.assertIsNone(image)
        self.assertEqual(len(visible), 2)
        self.assertEqual(len(info), 2)

    def test_missing_image_is_reported_but_objects_survive(self) -> None:
        tongsim = UnifiedTongSim("")
        mapper = SemanticMapper(tongsim=tongsim, character_id="agent")

        image, visible, info = mapper.get_perception_from_camera()

        self.assertIsNone(image)
        self.assertEqual(len(visible), 2)
        self.assertEqual(len(info), 2)

    def test_save_writes_the_server_composite(self) -> None:
        mapper = SemanticMapper(tongsim=UnifiedTongSim(_image_b64()), character_id="agent")

        with tempfile.TemporaryDirectory() as log_dir:
            mapper.get_perception_from_camera(is_save=True, log_dir=log_dir, save_label="unified")
            saved = list((Path(log_dir) / "prompts").glob("*.jpg"))

        self.assertEqual(len(saved), 1)
        self.assertIn("unified", saved[0].name)


class SplitPerceptionFallbackTest(unittest.TestCase):
    def test_not_implemented_falls_back_to_split_calls(self) -> None:
        tongsim = NoUnifiedTongSim()
        mapper = SemanticMapper(tongsim=tongsim, character_id="agent")

        image, visible, _ = mapper.get_perception_from_camera()

        self.assertEqual(tongsim.unified_attempts, 1)
        self.assertEqual(tongsim.split_calls, ["rgb", "segmentation", "visible"])
        self.assertIsNotNone(image)
        self.assertEqual(len(visible), 1)

    def test_absent_method_falls_back_to_split_calls(self) -> None:
        tongsim = AbsentUnifiedTongSim()
        mapper = SemanticMapper(tongsim=tongsim, character_id="agent")

        _, visible, _ = mapper.get_perception_from_camera()

        self.assertEqual(tongsim.unified_attempts, 0)
        self.assertEqual(tongsim.split_calls, ["rgb", "segmentation", "visible"])
        self.assertEqual(len(visible), 1)


class ObjectInHandQueryTest(unittest.TestCase):
    """912 只说手里有没有东西，物 ID 由最近一次抓取目标补上。"""

    def _agent(self, tongsim, last_pick=None):
        return SimpleNamespace(tongsim=tongsim, character_id="agent", _last_pick_raw_id=last_pick)

    def test_hand_id_comes_from_the_last_pick(self) -> None:
        client = SimpleNamespace(has_object_in_hand=lambda character_id: (True, 1))
        agent = self._agent(client, last_pick="7")

        self.assertEqual(VLMAgent._query_object_in_hand(agent), ("7", 1))

    def test_empty_hand_reports_nothing(self) -> None:
        client = SimpleNamespace(has_object_in_hand=lambda character_id: (False, None))
        agent = self._agent(client, last_pick="7")

        self.assertIsNone(VLMAgent._query_object_in_hand(agent))

    def test_holding_an_unidentified_object_still_reports_the_hand(self) -> None:
        client = SimpleNamespace(has_object_in_hand=lambda character_id: (True, 0))
        agent = self._agent(client, last_pick=None)

        self.assertEqual(VLMAgent._query_object_in_hand(agent), (None, 0))

    def test_older_server_falls_back_to_get_object_in_hand(self) -> None:
        client = SimpleNamespace(
            has_object_in_hand=lambda character_id: None,
            get_object_in_hand=lambda character_id: ("9", 1),
        )
        agent = self._agent(client)

        self.assertEqual(VLMAgent._query_object_in_hand(agent), ("9", 1))

    def test_server_without_either_rpc_reports_nothing(self) -> None:
        client = SimpleNamespace(get_object_in_hand=lambda character_id: None)
        agent = self._agent(client)

        self.assertIsNone(VLMAgent._query_object_in_hand(agent))


class SceneSurveyTest(unittest.TestCase):
    """扫完整圈后一次性标注：一次请求换掉逐帧往返。"""

    @staticmethod
    def _setup(response_text: str, frames: list[str], raw_to_mapped: dict[str, str]):
        strategy = TidyRoomStrategy()
        strategy.reset({"task_type": "tidyroom", "subject": "整理房间"})
        context = TaskContext(
            task_type="tidyroom",
            subject={},
            task_response={},
            visible_objects=[],
            object_in_hand=None,
            movable_objects=[],
            action_histories=[],
            raw_to_mapped_id=dict(raw_to_mapped),
        )

        class FakeClient:
            def __init__(self) -> None:
                self.calls: list[list[dict]] = []

            def invoke(self, messages):
                self.calls.append(messages)
                return SimpleNamespace(text=response_text)

        client = FakeClient()
        agent = SimpleNamespace(
            _tidyroom_survey_frames=list(frames),
            _task_context=context,
            _vlm_client_for_current_task=lambda: client,
        )
        return strategy, agent, client

    def test_items_and_furniture_are_applied(self) -> None:
        response = json.dumps(
            {
                "items": [{"object_id": "37", "semantic_label": "shoe", "destination_type": "shoe_storage"}],
                "furniture": [{"object_id": "19", "destination_type": "shoe_storage"}],
            }
        )
        strategy, agent, _ = self._setup(response, [_image_b64()], {"37": "37", "19": "19"})

        applied = run_scene_survey(strategy, agent)

        self.assertEqual(applied, 2)
        self.assertEqual(strategy.world.declared_targets, ["37"])
        self.assertEqual(strategy.world.scene_anchors["19"]["type"], "shoe_storage")

    def test_every_retained_frame_is_sent_in_one_request(self) -> None:
        strategy, agent, client = self._setup('{"items": [], "furniture": []}', [_image_b64()] * 3, {})

        run_scene_survey(strategy, agent)

        self.assertEqual(len(client.calls), 1)
        parts = client.calls[0][0]["content"]
        self.assertEqual(sum(1 for part in parts if part["type"] == "image_url"), 3)

    def test_survey_accepts_a_bare_list(self) -> None:
        """模型有时直接回一个数组，每条按自己的字段分类。"""
        strategy, agent, _ = self._setup(
            json.dumps([{"object_id": "14", "destination_type": "trash_bin"}]),
            [_image_b64()],
            {"14": "14"},
        )
        strategy.world.scene_objects["14"] = {
            "object_id": "14",
            "world_aabb": {
                "min": {"x": 800.4, "y": 253.3, "z": 3.9},
                "max": {"x": 827.5, "y": 280.5, "z": 38.8},
            },
        }

        applied = run_scene_survey(strategy, agent)

        self.assertEqual(applied, 1)
        self.assertEqual(strategy.world.scene_anchors["14"]["type"], "trash_bin")

    def test_unparsable_response_changes_nothing(self) -> None:
        strategy, agent, _ = self._setup("模型今天不想说话", [_image_b64()], {"37": "37"})

        applied = run_scene_survey(strategy, agent)

        self.assertEqual(applied, 0)
        self.assertEqual(strategy.world.targets, {})

    def test_survey_is_due_only_after_a_full_scan(self) -> None:
        strategy = TidyRoomStrategy()
        strategy.reset({"task_type": "tidyroom", "subject": "整理房间"})

        self.assertFalse(strategy.survey_is_due())

        strategy.scanner.turns_completed = strategy.scanner.turns_required
        self.assertTrue(strategy.survey_is_due())

        strategy.note_survey_attempted()
        self.assertFalse(strategy.survey_is_due())

    def test_missing_destination_is_abandoned_instead_of_looping(self) -> None:
        """模型把物品认成"鞋子"、屋里却没有鞋架时，不能一直转圈把时间耗光。"""
        strategy = TidyRoomStrategy()
        strategy.reset({"task_type": "tidyroom", "subject": "整理房间"})
        strategy.scanner.turns_completed = strategy.scanner.turns_required
        context = TaskContext(
            task_type="tidyroom",
            subject={},
            task_response={},
            visible_objects=[],
            object_in_hand=None,
            movable_objects=[],
            action_histories=[],
            raw_to_mapped_id={"51": "51"},
        )
        strategy.world.apply_scene_annotations(
            [{"object_id": "51", "semantic_label": "shoe", "destination_type": "shoe_storage"}],
            context,
        )
        self.assertEqual(strategy.world.destination_type_for(strategy.targets["51"]), "shoe_storage")

        # 每一轮都把 4 次补扫转完，等价于"转满整轮仍然找不到鞋架"
        for _ in range(strategy._MAX_MISSING_DESTINATION_ROUNDS):
            strategy.local_search_turns = strategy._MAX_LOCAL_SEARCH_TURNS
            strategy._next_local_search_action("missing_destination:shoe_storage")

        self.assertEqual(strategy.targets["51"]["status"], "blocked")
        self.assertTrue(strategy._all_targets_settled())

    def test_missing_destination_keeps_searching_before_giving_up(self) -> None:
        strategy = TidyRoomStrategy()
        strategy.reset({"task_type": "tidyroom", "subject": "整理房间"})
        strategy.scanner.turns_completed = strategy.scanner.turns_required
        context = TaskContext(
            task_type="tidyroom",
            subject={},
            task_response={},
            visible_objects=[],
            object_in_hand=None,
            movable_objects=[],
            action_histories=[],
            raw_to_mapped_id={"51": "51"},
        )
        strategy.world.apply_scene_annotations(
            [{"object_id": "51", "semantic_label": "shoe", "destination_type": "shoe_storage"}],
            context,
        )

        strategy.local_search_turns = strategy._MAX_LOCAL_SEARCH_TURNS
        strategy._next_local_search_action("missing_destination:shoe_storage")

        self.assertEqual(strategy.targets["51"]["status"], "pending")
        self.assertFalse(strategy._all_targets_settled())

    def test_survey_is_not_due_when_the_task_declared_targets(self) -> None:
        strategy = TidyRoomStrategy()
        strategy.reset(
            {"task_type": "tidyroom", "subject": "整理房间", "movable_object_id": ["BP_Pillow_10_TEST"]}
        )
        strategy.scanner.turns_completed = strategy.scanner.turns_required

        self.assertFalse(strategy.survey_is_due())


class JsonExtractionTest(unittest.TestCase):
    """对象里再套数组时，懒惰正则会在内层 ] 处提前收尾，必须按括号配对扫描。"""

    def test_nested_array_inside_an_object_still_parses(self) -> None:
        """2026-09-17 18:17 的真实失败载荷：整段 JSON 被截断，动作变成空的。"""
        text = """```json
[
  {
    "think": "当前视野中可以看到客厅内有多件杂乱物品，包括地板上的鞋子（ID 20）。",
    "action": "move_and_take_object",
    "parameters": {
      "object_id": "20",
      "which_hand": 0
    },
    "scene_annotations": [
      {"object_id": "20", "semantic_label": "shoe", "destination_type": "shoe_storage", "destination_object_id": "20"},
      {"object_id": "13", "semantic_label": "pillow", "destination_type": "sofa", "destination_object_id": "14"},
      {"anchor_object_id": "14", "destination_type": "sofa"}
    ]
  }
]
```"""
        parsed = extract_last_json_from_text(text)

        self.assertIsInstance(parsed, list)
        self.assertEqual(parsed[0]["action"], "move_and_take_object")
        self.assertEqual(parsed[0]["parameters"]["object_id"], "20")
        annotations = parsed[0]["scene_annotations"]
        self.assertEqual(len(annotations), 3)
        self.assertEqual(annotations[-1]["anchor_object_id"], "14")

    def test_plain_array_still_parses(self) -> None:
        text = '说明文字\n```json\n[{"action":"turn_in_degree","parameters":{"degree":45},"output":0}]\n```'

        parsed = extract_last_json_from_text(text)

        self.assertEqual(parsed[0]["action"], "turn_in_degree")

    def test_bare_object_still_parses(self) -> None:
        parsed = extract_last_json_from_text('{"action":"finish_task","parameters":{}}')

        self.assertEqual(parsed["action"], "finish_task")

    def test_brackets_inside_strings_do_not_confuse_the_scan(self) -> None:
        text = (
            '```json\n[{"think":"看到 { 和 ] 这类字符","action":"turn_in_degree",'
            '"parameters":{"degree":45},"output":0}]\n```'
        )

        parsed = extract_last_json_from_text(text)

        self.assertEqual(parsed[0]["action"], "turn_in_degree")


class TidyRoomCoverageGuardTest(unittest.TestCase):
    """912 不下发清单时，模型逐帧发现的清单不足以判定"目标已找齐"。"""

    @staticmethod
    def _strategy(subject) -> TidyRoomStrategy:
        strategy = TidyRoomStrategy()
        strategy.reset(subject)
        return strategy

    @staticmethod
    def _annotate_one_target_with_destination(strategy) -> None:
        context = TaskContext(
            task_type="tidyroom",
            subject={},
            task_response={},
            visible_objects=[],
            object_in_hand=None,
            movable_objects=[],
            action_histories=[],
            raw_to_mapped_id={"51": "51", "14": "14"},
        )
        strategy.world.apply_scene_annotations(
            [
                {
                    "object_id": "51",
                    "semantic_label": "pillow",
                    "destination_type": "sofa",
                    "destination_object_id": "14",
                }
            ],
            context,
        )

    def test_discovered_targets_do_not_finish_the_scan_early(self) -> None:
        """扫完整圈是唯一的覆盖保证，模型的一帧标注不能替它。"""
        strategy = self._strategy({"task_type": "tidyroom", "subject": "整理房间"})

        self._annotate_one_target_with_destination(strategy)

        self.assertEqual(strategy.world.declared_targets, ["51"])
        self.assertFalse(strategy._scan_requirements_met())

    def test_declared_targets_still_finish_the_scan_early(self) -> None:
        """任务系统给了完整清单时保持旧行为，不为凑满一圈空转。"""
        strategy = self._strategy(
            {"task_type": "tidyroom", "subject": "整理房间", "movable_object_id": ["51"]}
        )
        strategy.targets["51"]["object_id"] = "51"

        self._annotate_one_target_with_destination(strategy)

        self.assertTrue(strategy.world.targets_declared_by_task)
        self.assertTrue(strategy._scan_requirements_met())


class TaskSpecificVlmClientTest(unittest.TestCase):
    """整理房间挂专属视觉模型，其余任务仍用主模型。"""

    @staticmethod
    def _agent(dedicated, task_type):
        agent = PreliminaryBaselineAgent(stub=None, channel=None)
        agent.tidyroom_vlm_client = dedicated
        agent.vlm_client = "MAIN"
        agent._active_task_type = task_type
        return agent

    def test_tidyroom_prefers_its_dedicated_model(self) -> None:
        dedicated = object()

        agent = self._agent(dedicated, "tidyroom")

        self.assertIs(agent._vlm_client_for_current_task(), dedicated)

    def test_tidyroom_falls_back_when_unconfigured(self) -> None:
        agent = self._agent(None, "tidyroom")

        self.assertEqual(agent._vlm_client_for_current_task(), "MAIN")

    def test_other_tasks_keep_the_main_model(self) -> None:
        agent = self._agent(object(), "counting")

        self.assertEqual(agent._vlm_client_for_current_task(), "MAIN")


class TidyRoomTargetDiscoveryTest(unittest.TestCase):
    """912 不下发目标清单，整理房间的目标只能由模型在感知中指名。"""

    @staticmethod
    def _context(raw_to_mapped: dict[str, str]) -> TaskContext:
        return TaskContext(
            task_type="tidyroom",
            subject={},
            task_response={},
            visible_objects=[],
            object_in_hand=None,
            movable_objects=[],
            action_histories=[],
            raw_to_mapped_id=dict(raw_to_mapped),
        )

    def test_labelled_object_becomes_a_target(self) -> None:
        world = TidyRoomWorldModel({})

        world.apply_semantic_hints(
            {
                "object_id": "4",
                "semantic_label": "decorative_pillow",
                "destination_object_id": "14",
                "destination_type": "sofa",
            },
            self._context({"4": "4", "14": "14"}),
        )

        self.assertEqual(world.declared_targets, ["4"])
        record = world.targets["4"]
        # 立刻带上模型看到的 ID，本帧的 _raw_id_for_mapped_id 才能认出它。
        self.assertEqual(record["object_id"], "4")
        self.assertEqual(record["semantic_label"], "decorative_pillow")
        self.assertEqual(record["destination_type"], "sofa")
        self.assertEqual(world.pending_raw_ids(), ["4"])

    def test_navigation_without_a_label_does_not_create_a_target(self) -> None:
        """模型单纯导航到一件家具时不能把家具登记成待整理物品。"""
        world = TidyRoomWorldModel({})

        world.apply_semantic_hints({"object_id": "14"}, self._context({"14": "14"}))

        self.assertEqual(world.declared_targets, [])
        self.assertEqual(world.targets, {})

    def test_destination_anchor_is_created_from_the_hint(self) -> None:
        world = TidyRoomWorldModel({})

        world.apply_semantic_hints(
            {
                "object_id": "4",
                "semantic_label": "pillow",
                "destination_object_id": "14",
                "destination_type": "sofa",
            },
            self._context({"4": "4", "14": "14"}),
        )

        self.assertEqual(world.scene_anchors["14"]["type"], "sofa")
        self.assertTrue(world.has_destination("sofa"))

    def test_scene_annotations_register_many_targets_at_once(self) -> None:
        """一次回答就建立多个目标与家具，不必为每件物品再往返一次模型。"""
        world = TidyRoomWorldModel({})
        context = self._context({"37": "37", "41": "41", "20": "20", "38": "38", "42": "42"})

        applied = world.apply_scene_annotations(
            [
                {
                    "object_id": "37",
                    "semantic_label": "shoe",
                    "destination_type": "shoe_storage",
                    "destination_object_id": "20",
                },
                {
                    "object_id": "41",
                    "semantic_label": "discarded_can",
                    "destination_type": "trash_bin",
                    "destination_object_id": "38",
                },
                {"anchor_object_id": "42", "destination_type": "dining_table"},
            ],
            context,
        )

        self.assertEqual(applied, 3)
        self.assertEqual(sorted(world.declared_targets), ["37", "41"])
        self.assertEqual(world.targets["37"]["destination_type"], "shoe_storage")
        self.assertEqual(world.targets["41"]["destination_type"], "trash_bin")
        self.assertEqual(world.scene_anchors["20"]["type"], "shoe_storage")
        self.assertEqual(world.scene_anchors["38"]["type"], "trash_bin")
        self.assertEqual(world.scene_anchors["42"]["type"], "dining_table")

    def test_furniture_written_in_item_shape_becomes_an_anchor(self) -> None:
        """模型有时把家具名直接填进 semantic_label，不能当成待整理物品。"""
        world = TidyRoomWorldModel({})
        context = self._context({"14": "14", "35": "35", "38": "38"})

        applied = world.apply_scene_annotations(
            [
                {"object_id": "14", "semantic_label": "sofa", "destination_type": "sofa"},
                {
                    "object_id": "35",
                    "semantic_label": "trash_item",
                    "destination_type": "trash_bin",
                    "destination_object_id": "38",
                },
            ],
            context,
        )

        self.assertEqual(applied, 2)
        self.assertEqual(world.declared_targets, ["35"])
        self.assertEqual(world.scene_anchors["14"]["type"], "sofa")

    def test_objects_seen_earlier_survive_later_frames(self) -> None:
        """912 没有按物体查 AABB 的接口，只能靠累积缓存。

        走到目的地跟前时那件家具往往已经不在视野里，若每帧整体替换，放置校验
        就会永远 missing_aabb。
        """
        frames = [
            {"image": "a", "objects": [{"object_id": "19", "world_aabb": {"max": {"x": 2}}}]},
            {"image": "b", "objects": [{"object_id": "37", "world_aabb": {"max": {"x": 4}}}]},
        ]
        client = TongSimGrpcClient.__new__(TongSimGrpcClient)
        client._last_unified_objects = {}
        client._call_dynamic = lambda name, payload: frames.pop(0)

        client.acquire_first_person_perception("character")
        client.acquire_first_person_perception("character")

        self.assertEqual(client._last_unified_objects["19"]["world_aabb"]["max"]["x"], 2)
        self.assertEqual(client._last_unified_objects["37"]["world_aabb"]["max"]["x"], 4)

    def test_structural_regions_cannot_become_destinations(self) -> None:
        """地板被标成鞋架时，东西会被放到地上、校验还通过——那正是"瞎放东西"。"""
        world = TidyRoomWorldModel({})
        # 实测的地板包围盒
        world.scene_objects["20"] = {
            "object_id": "20",
            "world_aabb": {"min": {"x": 90.5, "y": -163.0, "z": 0.0}, "max": {"x": 841.5, "y": 644.0, "z": 5.0}},
        }
        # 实测的垃圾桶包围盒
        world.scene_objects["14"] = {
            "object_id": "14",
            "world_aabb": {"min": {"x": 800.4, "y": 253.3, "z": 3.9}, "max": {"x": 827.5, "y": 280.5, "z": 38.8}},
        }
        context = self._context({"14": "14", "20": "20"})

        world.apply_scene_annotations(
            [
                {"object_id": "20", "destination_type": "shoe_storage"},
                {"object_id": "14", "destination_type": "trash_bin"},
            ],
            context,
        )

        self.assertNotIn("20", world.scene_anchors)
        self.assertEqual(world.scene_anchors["14"]["type"], "trash_bin")

    def test_a_destination_is_never_also_a_target(self) -> None:
        """模型偶尔把垃圾桶同时写成物品和目的地，角色会去搬垃圾桶。"""
        world = TidyRoomWorldModel({})
        context = self._context({"18": "18", "35": "35"})

        applied = world.apply_scene_annotations(
            [
                {
                    "object_id": "18",
                    "semantic_label": "trash",
                    "destination_type": "trash_bin",
                    "destination_object_id": "18",
                },
                {
                    "object_id": "35",
                    "semantic_label": "trash_item",
                    "destination_type": "trash_bin",
                    "destination_object_id": "18",
                },
            ],
            context,
        )

        self.assertEqual(applied, 2)
        self.assertEqual(world.declared_targets, ["35"])
        self.assertEqual(world.scene_anchors["18"]["type"], "trash_bin")

    def test_destination_type_is_locked_to_the_first_identification(self) -> None:
        """模型逐帧标注时会摇摆，每帧覆盖会让规划器拿到互相矛盾的落点。"""
        world = TidyRoomWorldModel({})
        context = self._context({"19": "19"})

        world.apply_scene_annotations([{"anchor_object_id": "19", "destination_type": "shoe_storage"}], context)
        world.apply_scene_annotations([{"anchor_object_id": "19", "destination_type": "dining_table"}], context)

        self.assertEqual(world.scene_anchors["19"]["type"], "shoe_storage")

    def test_scene_annotations_are_read_from_the_action_top_level(self) -> None:
        """模型有时把 scene_annotations 放在动作顶层而不是 parameters 里。"""
        strategy = TidyRoomStrategy()
        strategy.reset({"task_type": "tidyroom", "subject": "整理房间"})
        context = TaskContext(
            task_type="tidyroom",
            subject={},
            task_response={},
            visible_objects=[],
            object_in_hand=None,
            movable_objects=[],
            action_histories=[],
            raw_to_mapped_id={"51": "51", "38": "38"},
        )

        strategy.validate_action(
            {
                "action": "move_and_take_object",
                "parameters": {"object_id": "51", "which_hand": 0},
                "output": 0,
                "scene_annotations": [
                    {
                        "object_id": "51",
                        "semantic_label": "trash_item",
                        "destination_type": "trash_bin",
                        "destination_object_id": "38",
                    }
                ],
            },
            context,
        )

        self.assertEqual(strategy.world.declared_targets, ["51"])
        self.assertEqual(strategy.world.targets["51"]["destination_type"], "trash_bin")

    def test_scene_annotations_drop_entries_that_resolve_to_nothing(self) -> None:
        """模型偶尔会写上这帧没看到的 ID，或写出非法的目的地类型。"""
        world = TidyRoomWorldModel({})
        context = self._context({"14": "14"})

        applied = world.apply_scene_annotations(
            [
                {"semantic_label": "sofa"},
                "not-a-dict",
                {"object_id": "99", "semantic_label": "cup", "destination_type": "dining_table"},
                {"anchor_object_id": "14", "destination_type": "bookshelf"},
            ],
            context,
        )

        self.assertEqual(applied, 0)
        self.assertEqual(world.targets, {})
        self.assertEqual(world.scene_anchors, {})

    def test_scene_annotations_tolerate_missing_or_malformed_input(self) -> None:
        world = TidyRoomWorldModel({})
        context = self._context({"14": "14"})

        self.assertEqual(world.apply_scene_annotations(None, context), 0)
        self.assertEqual(world.apply_scene_annotations("nonsense", context), 0)
        self.assertEqual(world.apply_scene_annotations([], context), 0)

    def test_unlabelled_navigation_still_annotates_a_known_target(self) -> None:
        """已在清单里的目标仍然接受无标签的语义补充。"""
        world = TidyRoomWorldModel({"movable_object_id": ["4"]})

        world.apply_semantic_hints({"object_id": "4"}, self._context({"4": "4"}))

        self.assertEqual(world.declared_targets, ["4"])
        self.assertIn("4", world.targets)


class RawObjectIdTranslationTest(unittest.TestCase):
    """统一感知给出的已经是服务端原始 ID，不能再按映射 ID 反查一次。

    反查失败会让 move_to_object 报「object_id 无效」，计数任务的遮挡区导航
    就是这样整个失败的。
    """

    @staticmethod
    def _agent(mapper):
        return SimpleNamespace(semantic_mapper=mapper)

    def test_empty_mapper_passes_the_id_through(self) -> None:
        mapper = SemanticMapper(tongsim=UnifiedTongSim(_image_b64()), character_id="agent")

        agent = self._agent(mapper)

        self.assertEqual(VLMAgent._to_raw_object_id(agent, "4"), "4")
        self.assertEqual(VLMAgent._to_raw_object_id(agent, 4), 4)
        # 记忆库的合成 ID 不是数字，本来就走透传分支。
        self.assertEqual(VLMAgent._to_raw_object_id(agent, "o4"), "o4")

    def test_populated_mapper_still_translates_mapped_ids(self) -> None:
        """旧服务端的顺序映射仍然要翻译——只有查不到时才透传。"""
        mapper = SemanticMapper(tongsim=UnifiedTongSim(_image_b64()), character_id="agent")
        mapper.get_id_mapping([{"object_id": "raw-a"}, {"object_id": "raw-b"}])

        agent = self._agent(mapper)

        self.assertEqual(VLMAgent._to_raw_object_id(agent, "1"), "raw-a")
        self.assertEqual(VLMAgent._to_raw_object_id(agent, 2), "raw-b")

    def test_absent_mapper_returns_the_input(self) -> None:
        agent = self._agent(None)

        self.assertEqual(VLMAgent._to_raw_object_id(agent, "4"), "4")


if __name__ == "__main__":
    unittest.main()
