import unittest

from google.protobuf import json_format, struct_pb2

from arenaagent.preliminary_baseline_agent.preliminary_baseline_agent import (
    PreliminaryBaselineAgent,
    PreliminaryBaselineAgentCfg,
)
from arenaagent.tongsim_grpc_client import TongSimGrpcClient
from arenaagent.vlm_agent.client import ClientResponse


class CameraMustNotRun:
    def get_perception_from_camera(self, *args, **kwargs):
        raise AssertionError("NPC runtime must not acquire camera perception")


class LegacyNpcTongSim:
    def __init__(self) -> None:
        self.moves: list[tuple[str, str]] = []

    def get_object_in_hand(self, *args, **kwargs):
        raise AssertionError("NPC runtime must not query the held object")

    def get_object_id_by_name(self, *args, **kwargs):
        raise AssertionError("Legacy NPC movement must not query object IDs")

    def move_to_npc(self, character_id: str, asset_name: str) -> dict:
        self.moves.append((character_id, asset_name))
        return {"result": "success"}


class StaticDecisionClient:
    def __init__(self) -> None:
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return ClientResponse(
            text='[{"action":"submit_answer","output":"张奶奶"}]',
            token_usage=None,
        )


class LegacyRpcChannel:
    def __init__(self) -> None:
        self.path = ""
        self.payload = None
        self.metadata = None

    def unary_unary(self, path, *, request_serializer, response_deserializer):
        del request_serializer, response_deserializer
        self.path = path

        def invoke(request, *, metadata):
            self.payload = json_format.MessageToDict(request)
            self.metadata = metadata
            response = struct_pb2.Struct()
            response.update({"result": "success"})
            return response

        return invoke


class NpcFastRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tongsim = LegacyNpcTongSim()
        self.agent = PreliminaryBaselineAgent(None, None, cfg=PreliminaryBaselineAgentCfg())
        self.agent._initialized = True
        self.agent.character_id = "character-1"
        self.agent.tongsim = self.tongsim
        self.agent.semantic_mapper = CameraMustNotRun()
        self.agent.action_space = {"key": "answer"}
        self.agent.vlm_client = StaticDecisionClient()
        self.agent._save_prompt_messages = lambda messages: None
        self.asked_names: list[str] = []

        def call_struct(method_name, payload, response_type):
            del response_type
            if method_name != "speak_to":
                raise AssertionError(f"Unexpected task RPC: {method_name}")
            name = payload["npc_name"]
            self.asked_names.append(name)
            response = struct_pb2.Struct()
            response.update({"npc_reply": f"{name}的有效证词"})
            return response

        self.agent._call_struct = call_struct
        self.subject = {
            "task_type": "npc",
            "subject": "找出谁拿了钥匙",
            "options": ["江淑艳", "刘伟东", "赵爷爷", "张奶奶"],
            "npc_asset_name": {
                "江淑艳": "asset-jiang",
                "刘伟东": "asset-liu",
                "赵爷爷": "asset-zhao",
                "张奶奶": "asset-zhang",
            },
        }

    def test_first_npc_step_skips_camera_and_uses_legacy_npc_movement(self) -> None:
        result = self.agent.run_step(self.subject, {})

        self.assertEqual("江淑艳", result["npc_name"])
        self.assertEqual(["江淑艳"], self.asked_names)
        self.assertEqual([("character-1", "asset-jiang")], self.tongsim.moves)
        self.assertIsNone(self.agent.vlm_client.messages)

    def test_final_npc_decision_uses_ordered_text_only_evidence(self) -> None:
        for _ in range(4):
            self.agent.run_step(self.subject, {})

        result = self.agent.run_step(self.subject, {})

        self.assertEqual(["江淑艳", "刘伟东", "赵爷爷", "张奶奶"], self.asked_names)
        self.assertEqual({"answer": "张奶奶"}, result)
        messages = self.agent.vlm_client.messages
        self.assertEqual(["system", "user"], [message["role"] for message in messages])
        self.assertTrue(all(isinstance(message["content"], str) for message in messages))
        self.assertNotIn("image_url", str(messages))
        self.assertLess(str(messages).index("江淑艳"), str(messages).index("张奶奶"))

    def test_tongsim_client_can_call_legacy_move_to_npc_rpc(self) -> None:
        channel = LegacyRpcChannel()
        client = TongSimGrpcClient.__new__(TongSimGrpcClient)
        client._channel = channel
        client._metadata = (("x-tongsim-client-id", "test-client"),)

        self.assertTrue(hasattr(client, "move_to_npc"), "缺少旧版 move_to_npc 兼容入口")
        result = client.move_to_npc("character-1", "asset-jiang")

        self.assertEqual("/tongsim.service.TongSimService/move_to_npc", channel.path)
        self.assertEqual(
            {"character_id": "character-1", "name": "asset-jiang"},
            channel.payload,
        )
        self.assertEqual((("x-tongsim-client-id", "test-client"),), channel.metadata)
        self.assertEqual({"result": "success"}, result)


if __name__ == "__main__":
    unittest.main()
