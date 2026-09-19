import unittest

from arenaagent.preliminary_baseline_agent.preliminary_baseline_agent import (
    PreliminaryBaselineAgent,
)
from arenaagent.preliminary_baseline_agent.tasks.npc.strategy import NpcStrategy


class NpcCompletionTests(unittest.TestCase):
    def test_npc_fast_loop_only_requests_subject_evaluation_after_submit(self):
        agent = object.__new__(PreliminaryBaselineAgent)
        events = []
        agent._NPC_MAX_LOCAL_STEPS = 5
        agent.subject_finished = False
        agent.sleep_between_steps = 0.0
        agent._get_response_from_task = lambda: {}
        agent._evaluate_subject = lambda: events.append("subject")
        agent._evaluate_task = lambda: self.fail("task evaluation races the final subject result")
        agent._apply_action = lambda action: events.append("apply") or {"answer_right": True}

        def run_step(subject, task_response):
            agent._last_executed_action_name = "submit_answer"
            return {"answer": "刘伟东"}

        agent.run_step = run_step
        agent._run_npc_subject_fast({})

        self.assertEqual(events, ["apply", "subject"])

    def test_npc_fast_loop_retries_after_server_rejects_answer(self):
        agent = object.__new__(PreliminaryBaselineAgent)
        events = []
        strategy = NpcStrategy()
        strategy.reset({})
        agent._task_strategy = strategy
        agent._NPC_MAX_LOCAL_STEPS = 8
        agent.subject_finished = False
        agent.sleep_between_steps = 0.0
        agent._get_response_from_task = lambda: {}
        agent._evaluate_subject = lambda: events.append("subject")
        agent._evaluate_task = lambda: self.fail("task evaluation must not run here")

        answers = iter(("张奶奶", "江淑艳"))

        def run_step(subject, task_response):
            del subject, task_response
            answer = next(answers)
            agent._last_executed_action_name = "submit_answer"
            return {"answer": answer}

        def apply_action(action):
            events.append(("apply", action["answer"]))
            return {"answer_right": action["answer"] == "江淑艳"}

        agent.run_step = run_step
        agent._apply_action = apply_action
        agent._run_npc_subject_fast({})

        self.assertEqual(events, [("apply", "张奶奶"), ("apply", "江淑艳"), "subject"])
        self.assertEqual(strategy.rejected_answers, ["张奶奶"])


if __name__ == "__main__":
    unittest.main()
