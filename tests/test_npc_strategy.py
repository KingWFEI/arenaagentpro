import unittest

from arenaagent.preliminary_baseline_agent.task_runtime import TaskContext
from arenaagent.preliminary_baseline_agent.tasks.npc.strategy import NpcStrategy


class NpcStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.subject = {
            "task_type": "npc",
            "subject": "找出谁拿了钥匙",
            "options": ["江淑艳", "刘伟东", "赵爷爷", "张奶奶"],
        }
        self.strategy = NpcStrategy()
        self.strategy.reset(self.subject)

    def context(self) -> TaskContext:
        return TaskContext(
            task_type="npc",
            subject=self.subject,
            task_response={},
            visible_objects=[],
            object_in_hand=None,
            movable_objects=[],
            action_histories=[],
        )

    @staticmethod
    def dialogue_action(name: str) -> dict:
        return {
            "action": "speak_to_npc",
            "parameters": {"npc_name": name, "message": "调查问题"},
            "output": 0,
        }

    def record_reply(self, name: str, reply: str, *, error: str = "") -> None:
        result = {"npc_name": name, "npc_reply": reply}
        if error:
            result["error"] = error
        self.strategy.after_action(self.dialogue_action(name), result, self.context())

    def test_interviews_first_unanswered_candidate_in_subject_order(self) -> None:
        self.record_reply("江淑艳", "我在客厅。")

        action = self.strategy.next_local_action(self.context())

        self.assertIsNotNone(action)
        self.assertEqual("speak_to_npc", action["action"])
        self.assertEqual("刘伟东", action["parameters"]["npc_name"])

    def test_empty_or_failed_reply_is_retried(self) -> None:
        self.record_reply("江淑艳", "")
        self.record_reply("江淑艳", "暂时无法对话", error="rpc failed")

        action = self.strategy.next_local_action(self.context())

        self.assertIsNotNone(action)
        self.assertEqual("江淑艳", action["parameters"]["npc_name"])
        state = self.strategy.state_for_prompt()
        self.assertIn("testimonies", state)
        self.assertEqual([], state["testimonies"])

    def test_explicit_failure_markers_are_retried_even_with_reply_text(self) -> None:
        failure_markers = (
            {"success": False},
            {"result": "failed"},
            {"status": "failed"},
            {"result": False},
            {"result": 0},
            {"status": False},
            {"status": 0},
        )
        for marker in failure_markers:
            with self.subTest(marker=marker):
                strategy = NpcStrategy()
                strategy.reset(self.subject)
                result = {"npc_name": "江淑艳", "npc_reply": "服务暂时不可用", **marker}

                strategy.after_action(self.dialogue_action("江淑艳"), result, self.context())

                action = strategy.next_local_action(self.context())
                self.assertIsNotNone(action)
                self.assertEqual("江淑艳", action["parameters"]["npc_name"])

    def test_premature_submission_is_replaced_with_next_interview(self) -> None:
        self.record_reply("江淑艳", "我在客厅。")

        action = self.strategy.validate_action(
            {"action": "submit_answer", "parameters": {}, "output": "张奶奶"},
            self.context(),
        )

        self.assertEqual("speak_to_npc", action["action"])
        self.assertEqual("刘伟东", action["parameters"]["npc_name"])

    def test_premature_finish_is_replaced_with_next_interview(self) -> None:
        action = self.strategy.validate_action(
            {"action": "finish_task", "parameters": {}, "output": ""},
            self.context(),
        )

        self.assertEqual("speak_to_npc", action["action"])
        self.assertEqual("江淑艳", action["parameters"]["npc_name"])

    def test_complete_evidence_is_exposed_without_old_reasoning(self) -> None:
        replies = {
            "江淑艳": "我在客厅。",
            "刘伟东": "我看到张奶奶靠近钥匙柜。",
            "赵爷爷": "我一直在院子里。",
            "张奶奶": "我没拿钥匙。",
        }
        for name, reply in replies.items():
            self.record_reply(name, reply)

        self.assertIsNone(self.strategy.next_local_action(self.context()))
        state = self.strategy.state_for_prompt()

        self.assertIn("question", state)
        self.assertEqual("找出谁拿了钥匙", state["question"])
        self.assertEqual(
            [{"npc_name": name, "reply": reply} for name, reply in replies.items()],
            state["testimonies"],
        )
        self.assertNotIn("think", state)

    def test_invalid_final_candidate_is_not_submitted(self) -> None:
        for name in self.subject["options"]:
            self.record_reply(name, f"{name}的证言")

        action = self.strategy.validate_action(
            {"action": "submit_answer", "parameters": {}, "output": "其他人"},
            self.context(),
        )

        self.assertNotEqual("submit_answer", action["action"])

    def test_exact_final_candidate_is_submitted(self) -> None:
        for name in self.subject["options"]:
            self.record_reply(name, f"{name}的证言")
        expected = {"action": "submit_answer", "parameters": {}, "output": "张奶奶"}

        action = self.strategy.validate_action(expected, self.context())

        self.assertEqual(expected, action)

    def test_final_candidate_whitespace_is_removed_before_submission(self) -> None:
        for name in self.subject["options"]:
            self.record_reply(name, f"{name}的证言")

        action = self.strategy.validate_action(
            {"action": "submit_answer", "parameters": {}, "output": " 张奶奶 "},
            self.context(),
        )

        self.assertEqual("submit_answer", action["action"])
        self.assertEqual("张奶奶", action["output"])

    def test_candidate_mentioned_in_prose_is_not_submitted(self) -> None:
        for name in self.subject["options"]:
            self.record_reply(name, f"{name}的证言")

        action = self.strategy.validate_action(
            {"action": "submit_answer", "parameters": {}, "output": "不是张奶奶"},
            self.context(),
        )

        self.assertNotEqual("submit_answer", action["action"])

    def test_malformed_options_still_require_all_four_known_candidates(self) -> None:
        strategy = NpcStrategy()
        subject = dict(self.subject, options=["江淑艳", "江淑艳", "陌生人"])
        strategy.reset(subject)
        strategy.after_action(
            self.dialogue_action("江淑艳"),
            {"npc_name": "江淑艳", "npc_reply": "我在客厅。"},
            self.context(),
        )

        state = strategy.state_for_prompt()

        self.assertEqual(self.subject["options"], state["candidates"])
        self.assertEqual(["刘伟东", "赵爷爷", "张奶奶"], state["pending_npcs"])
        self.assertFalse(state["ready_to_decide"])


if __name__ == "__main__":
    unittest.main()
