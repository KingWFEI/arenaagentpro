from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, ImageDraw

from arenaagent.agent_base import summarize_subject_for_log
from arenaagent.preliminary_baseline_agent.preliminary_baseline_agent import PreliminaryBaselineAgent
from arenaagent.preliminary_baseline_agent.task_runtime import TaskContext
from arenaagent.preliminary_baseline_agent.tasks.raven.ensemble import combine_question
from arenaagent.preliminary_baseline_agent.tasks.raven.experience import (
    record_successful_subject,
    relevant_experience_hints,
)
from arenaagent.preliminary_baseline_agent.tasks.raven.reasoners import (
    ask_text_verifier,
    ask_visual_reasoner,
)
from arenaagent.preliminary_baseline_agent.tasks.raven.rules import RuleInductionResult, induce_rules
from arenaagent.preliminary_baseline_agent.tasks.raven.solver import (
    HybridRavenSolver,
    restore_prior_model_votes,
    select_suspect_questions,
)
from arenaagent.preliminary_baseline_agent.tasks.raven.strategy import RavenStrategy
from arenaagent.preliminary_baseline_agent.tasks.raven.text_client import (
    build_raven_text_client_from_env,
)
from arenaagent.preliminary_baseline_agent.tasks.raven.verifier import (
    ModelVote,
    parse_direct_model_votes,
    parse_model_votes,
    parse_single_model_vote,
    sanitize_ranked_triples,
)
from arenaagent.preliminary_baseline_agent.tasks.raven.vision import extract_question_observations
from arenaagent.preliminary_baseline_agent.tasks.raven.vision_client import (
    build_raven_vision_client_from_env,
)
from arenaagent.vlm_agent.raven_skill import group_image_to_raven_list, handle


def _dots(*centers: tuple[int, int]) -> Image.Image:
    image = Image.new("L", (128, 128), "white")
    draw = ImageDraw.Draw(image)
    for x, y in centers:
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill="black")
    return image


def _circle(radius: int) -> Image.Image:
    image = Image.new("L", (128, 128), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((64 - radius, 64 - radius, 64 + radius, 64 + radius), fill="black")
    return image


class RavenRuleEngineTests(unittest.TestCase):
    def test_verified_success_becomes_reusable_technique_without_answer_key(self) -> None:
        context = [
            _dots((28, 36)),
            _dots((94, 36)),
            _dots((28, 36), (94, 36)),
            _dots((28, 64)),
            _dots((94, 64)),
            _dots((28, 64), (94, 64)),
            _dots((28, 92)),
            _dots((94, 92)),
        ]
        candidates = [
            _dots((28, 92), (94, 92)),
            _dots((28, 92)),
            _dots((94, 92)),
            _dots((64, 92)),
            _dots((28, 36), (94, 36)),
            _dots((28, 64), (94, 64)),
            _dots(),
            _dots((28, 92), (64, 92), (94, 92)),
        ]
        group = context + candidates
        diagnostics = {
            "visual_votes": [
                {
                    "question": question,
                    "answer": 1,
                    "rule": "Candidate 1 is the pixel union of the first two panels.",
                    "evidence": ["Candidate 1 preserves both object locations."],
                }
                for question in range(1, 4)
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experience.json"
            learned = record_successful_subject(
                [group, group, group],
                [1, 1, 1],
                diagnostics=diagnostics,
                subject_fingerprint="verified-subject",
                path=path,
            )
            hints = relevant_experience_hints(
                induce_rules(extract_question_observations(group)),
                path=path,
            )
            stored = path.read_text(encoding="utf-8")
        self.assertTrue(learned)
        self.assertIsNotNone(hints)
        self.assertEqual((hints or {}).get("verified_unique_subjects"), 1)
        self.assertTrue((hints or {}).get("matched_techniques"))
        self.assertNotIn("Candidate 1", stored)

    def test_monotonic_size_rule_prefers_smaller_third_item(self) -> None:
        context = [
            _circle(30),
            _circle(22),
            _circle(14),
            _circle(28),
            _circle(20),
            _circle(12),
            _circle(32),
            _circle(24),
        ]
        candidates = [
            _circle(16),
            _circle(24),
            _circle(30),
            _circle(8),
            _circle(20),
            _circle(12),
            _circle(26),
            _circle(10),
        ]
        result = induce_rules(extract_question_observations(context + candidates))
        monotonic = [match for match in result.matches if "monotonic_decrease" in match.name]
        self.assertTrue(monotonic)
        self.assertGreater(result.scores[0], result.scores[1])

    def test_circle_contours_do_not_induce_polygon_vertex_arithmetic(self) -> None:
        observations = extract_question_observations([_circle(radius) for radius in range(12, 28)])
        result = induce_rules(observations)
        unsafe = [
            match
            for match in result.matches
            if match.name.startswith("polygon_vertex")
            and any(
                token in match.name
                for token in (
                    ":sum",
                    ":mean",
                    ":absolute_difference",
                    "arithmetic_progression",
                    "distribute_three",
                    "cyclic",
                )
            )
        ]
        self.assertEqual(unsafe, [])

    def test_aligned_objects_are_not_mistaken_for_layout_divider(self) -> None:
        image = Image.new("L", (128, 128), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((44, 4, 84, 56), fill="black")
        draw.rectangle((44, 72, 84, 124), fill="black")
        observation = extract_question_observations([image] * 16)[0]
        self.assertEqual(observation.features.component_count, 2.0)

    def test_pixel_union_rule_ranks_exact_candidate_first(self) -> None:
        context = [
            _dots((28, 36)),
            _dots((94, 36)),
            _dots((28, 36), (94, 36)),
            _dots((28, 64)),
            _dots((94, 64)),
            _dots((28, 64), (94, 64)),
            _dots((28, 92)),
            _dots((94, 92)),
        ]
        candidates = [
            _dots((28, 92), (94, 92)),
            _dots((28, 92)),
            _dots((94, 92)),
            _dots((64, 92)),
            _dots((28, 36), (94, 36)),
            _dots((28, 64), (94, 64)),
            _dots(),
            _dots((28, 92), (64, 92), (94, 92)),
        ]
        observations = extract_question_observations(context + candidates)
        result = induce_rules(observations)
        self.assertEqual(max(range(8), key=result.scores.__getitem__), 0)
        self.assertTrue(any(match.name == "pixel_union" for match in result.matches))

    def test_distribute_three_rule_recovers_missing_category(self) -> None:
        # Triangle/pentagon/hexagon values are distributed once per row.
        shapes = [3, 5, 6, 5, 6, 3, 6, 3]
        context = []
        for sides in shapes:
            image = Image.new("L", (128, 128), "white")
            draw = ImageDraw.Draw(image)
            import math

            points = [
                (
                    64 + 30 * math.cos(-math.pi / 2 + 2 * math.pi * index / sides),
                    64 + 30 * math.sin(-math.pi / 2 + 2 * math.pi * index / sides),
                )
                for index in range(sides)
            ]
            draw.polygon(points, outline="black", width=4)
            context.append(image)
        candidates = [context[0], context[1], context[2], context[0], context[1], context[1], context[2], context[0]]
        result = induce_rules(extract_question_observations(context + candidates))
        distribute = [match for match in result.matches if "distribute_three" in match.name]
        self.assertTrue(distribute)
        self.assertTrue(any(max(range(8), key=match.candidate_scores.__getitem__) in {1, 4, 5} for match in distribute))


class RavenVerifierTests(unittest.TestCase):
    def test_whole_image_direct_answers_accept_label_digits_and_json_list(self) -> None:
        for response in ("答案：5,8,2", "582", "[5, 8, 2]"):
            votes = parse_direct_model_votes(response)
            self.assertEqual([vote.answer for vote in votes], [5, 8, 2])
            self.assertTrue(all(vote.confidence == 0.75 for vote in votes))

    def test_whole_image_direct_answers_parse_per_question_confidences(self) -> None:
        votes = parse_direct_model_votes("答案：2,7,6；置信度：0.85,0.90,0.80")
        self.assertEqual([vote.answer for vote in votes], [2, 7, 6])
        self.assertEqual([vote.confidence for vote in votes], [0.85, 0.9, 0.8])

    def test_whole_image_direct_answers_do_not_guess_from_explanatory_numbers(self) -> None:
        self.assertEqual(parse_direct_model_votes("第1题可能是5，第2题可能是8。"), [])

    def test_eight_way_distribution_overrides_inconsistent_declared_answer(self) -> None:
        candidates = [
            {"id": index, "confidence": 0.65 if index == 6 else 0.05, "mismatch": []}
            for index in range(1, 9)
        ]
        vote = parse_single_model_vote(
            json.dumps({"question": 2, "answer": 4, "candidates": candidates}),
            question=2,
        )
        self.assertIsNotNone(vote)
        self.assertEqual(vote.answer if vote else None, 6)
        self.assertEqual(len(vote.candidate_confidences if vote else {}), 8)
        self.assertAlmostEqual(sum((vote.candidate_confidences if vote else {}).values()), 1.0)

    def test_incomplete_single_question_json_is_recovered_and_normalized(self) -> None:
        response = (
            '{"question":3,"answer":6,"confidence":0.8,"candidates":['
            '{"id":1,"confidence":0.1},{"id":2,"confidence":0.1},'
            '{"id":3,"confidence":0.1},{"id":4,"confidence":0.1},'
            '{"id":5,"confidence":0.1},{"id":6,"confidence":0.8},'
            '{"id":7,"confidence":0.1},{"id":8,"confidence":0.1}],"rule":"cut off'
        )
        vote = parse_single_model_vote(response, question=3)
        self.assertIsNotNone(vote)
        self.assertEqual(vote.answer if vote else None, 6)
        self.assertEqual(len(vote.candidate_confidences if vote else {}), 8)
        self.assertAlmostEqual(sum((vote.candidate_confidences if vote else {}).values()), 1.0)

    def test_model_votes_require_complete_one_based_answers(self) -> None:
        response = (
            '{"questions":['
            '{"question":1,"answer":2,"confidence":0.8},'
            '{"question":2,"answer":0,"confidence":0.9},'
            '{"question":3,"answer":8,"confidence":75}'
            "]}"
        )
        self.assertEqual(parse_model_votes(response), [])

    def test_ranked_triples_remove_invalid_and_duplicate_values(self) -> None:
        self.assertEqual(
            sanitize_ranked_triples([[1, 2, 3], [1, 2, 3], [0, 2, 3], [8, 7, 6]]),
            [[1, 2, 3], [8, 7, 6]],
        )

    def test_confident_visual_vote_beats_conflicting_local_prior(self) -> None:
        question = combine_question(
            rule_scores=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            rule_confidence=0.9,
            visual_vote=ModelVote(question=1, answer=2, confidence=0.9),
            text_vote=ModelVote(question=1, answer=3, confidence=0.9),
        )
        self.assertEqual(question.answer, 2)


class RavenRoutingTests(unittest.TestCase):
    def test_strategy_dispatches_solver_locally(self) -> None:
        strategy = RavenStrategy()
        context = TaskContext(
            task_type="raven",
            subject={"task_type": "raven"},
            task_response={},
            visible_objects=[],
            object_in_hand=None,
            movable_objects=[],
            action_histories=[],
        )
        strategy.reset(context.subject)
        self.assertEqual(strategy.next_local_action(context)["action"], "solve_raven")

    def test_raven_skips_3d_camera_without_changing_tidyroom_branch(self) -> None:
        agent = PreliminaryBaselineAgent(stub=None, channel=None)
        self.assertTrue(agent._should_use_lightweight_perception({"task_type": "raven"}))
        self.assertFalse(agent._should_refresh_lightweight_scene({"task_type": "raven"}))
        self.assertTrue(agent._should_use_lightweight_perception({"task_type": "tidyroom"}))


class _FakeVisionClient:
    """只回文本的假视觉客户端，用来驱动纯视觉求解路径。

    `answers` 给定时按**请求里写的题号**回 `{"answers": [n]}`：三题并行发出，
    调用次序没有保证，只看调用次数会让测试随机翻车。
    """

    def __init__(self, text: str, answers: list[int] | None = None) -> None:
        self.text = text
        self.answers = list(answers) if answers else None
        self.calls: list[list[dict]] = []
        self._lock = threading.Lock()

    def invoke(self, messages, max_retries: int = 1):
        del max_retries
        with self._lock:
            self.calls.append(messages)
        question = self._question_of(messages)
        if self.answers and question is not None and 1 <= question <= len(self.answers):
            return SimpleNamespace(text=json.dumps({"answers": [self.answers[question - 1]]}))
        return SimpleNamespace(text=self.text)

    @staticmethod
    def _question_of(messages) -> int | None:
        for part in messages[0].get("content", []) if messages else []:
            if part.get("type") != "text":
                continue
            match = re.search(r"第\s*(\d)\s*题", part.get("text", ""))
            if match:
                return int(match.group(1))
        return None


class _PureVisionAgent:
    """handle() 需要的最小 agent 形状。"""

    subject_finished = False
    action_space = {"key": "answer"}

    def __init__(self, image_path: str) -> None:
        self._raven_image_temp_path = image_path


def _synthetic_raven_panel() -> Image.Image:
    """画一块提取器量得到的题图：8 个矩阵格 + 8 个候选格，每格一个深色五边形。"""
    panel = Image.new("RGB", (792, 1200), "white")
    draw = ImageDraw.Draw(panel)

    def cell(x: int, y: int, size: int, radius_ratio: float) -> None:
        draw.rectangle([x, y, x + size, y + size], outline="black", width=3)
        cx, cy, radius = x + size / 2, y + size / 2, size * radius_ratio
        draw.polygon(
            [
                (
                    cx + radius * math.cos(math.radians(-90 + 72 * step)),
                    cy + radius * math.sin(math.radians(-90 + 72 * step)),
                )
                for step in range(5)
            ],
            fill="black",
        )

    for row, y in enumerate((107, 319, 531)):
        for col, x in enumerate((52, 296, 540)):
            if (row, col) == (2, 2):
                continue  # 问号格
            cell(x, y, 196, 0.3 + 0.03 * row)
    for y in (811, 1010):
        for x in (53, 228, 403, 578):
            cell(x, y, 156, 0.28)
    return panel


class RavenRuntimeSafetyTests(unittest.TestCase):
    def test_raven_third_attempt_uses_per_position_consensus(self) -> None:
        agent = _PureVisionAgent("same-subject.png")
        agent.action_space = {"key": "answer"}
        agent.vlm_client = object()
        with patch(
            "arenaagent.vlm_agent.raven_skill.resolve_raven_image_path",
            return_value="same-subject.png",
        ), patch(
            "arenaagent.vlm_agent.raven_skill.solve_pure_vision",
            side_effect=[
                ([5, 8, 6], {}),
                ([5, 8, 2], {}),
                ([8, 8, 2], {}),
            ],
        ):
            first = handle(agent, {"attempt": 1}, {})
            second = handle(agent, {"attempt": 2}, {})
            third = handle(agent, {"attempt": 3}, {})

        self.assertEqual(first, {"answer": [5, 8, 6]})
        self.assertEqual(second, {"answer": [5, 8, 2]})
        self.assertEqual(third, {"answer": [5, 8, 2]})
        self.assertEqual(agent._raven_last_diagnostics["raw_answers"], [8, 8, 2])

    def test_raven_consensus_tie_prefers_latest_review(self) -> None:
        from arenaagent.vlm_agent.raven_skill import _consensus_raven_answers

        self.assertEqual(
            _consensus_raven_answers([[1, 2, 3], [4, 5, 6]]),
            [4, 5, 6],
        )

    def test_raven_vision_client_uses_non_thinking_k3_with_short_timeout(self) -> None:
        sentinel = object()
        environment = {
            "RAVEN_ENABLE_VLM": "1",
            "RAVEN_VLM_MODEL": "",
            "RAVEN_VLM_API_BASE": "",
            "RAVEN_VLM_API_KEY": "",
            "RAVEN_VLM_TIMEOUT_SECONDS": "",
            "VLM_CLIENT_CFG_API_KEY": "test-only-key",
        }
        with patch.dict("os.environ", environment, clear=False), patch(
            "arenaagent.preliminary_baseline_agent.aux_client.ClientFactory.build",
            return_value=sentinel,
        ) as build:
            client = build_raven_vision_client_from_env()

        self.assertIs(client, sentinel)
        client_type, cfg = build.call_args.args
        self.assertEqual(client_type, "openai")
        self.assertEqual(cfg.name, "kimi-k3")
        self.assertEqual(cfg.request_timeout_seconds, 30.0)
        self.assertEqual(
            cfg.chat_completion_kwargs,
            {
                "extra_body": {"thinking": {"type": "disabled"}},
                # 单题作答要先给一句规律说明再给编号，256 会截断。
                "max_tokens": 512,
            },
        )

    def test_solve_raven_local_action_does_not_require_tongsim(self) -> None:
        agent = PreliminaryBaselineAgent(stub=None, channel=None)
        agent.tongsim = None
        agent.character_id = None
        expected = {"answer": [5, 8, 2]}

        with patch.object(agent, "_handle_solve_raven", return_value=expected) as solve:
            result = agent._do_action({"action": "solve_raven", "parameters": {}})

        self.assertEqual(result, expected)
        solve.assert_called_once()

    def test_raven_submits_once_when_the_subject_settles(self) -> None:
        agent = PreliminaryBaselineAgent(stub=None, channel=None)
        agent.action_space = {"key": "answer"}
        subject = {"task_type": "raven", "task_data": "image"}
        with patch.object(agent, "_raven_current_subject_index", side_effect=[4, 4]), patch.object(
            agent, "_call_struct", return_value=object()
        ), patch(
            "arenaagent.preliminary_baseline_agent.preliminary_baseline_agent.parse_struct_to_data",
            return_value={"key": "answer"},
        ), patch.object(agent, "_get_response_from_task", return_value={}), patch.object(
            agent, "run_step", return_value={"answer": [6, 8, 7]}
        ) as run_step, patch.object(agent, "_apply_action", return_value={}) as apply_action, patch.object(
            agent, "_evaluate_subject", return_value={}
        ) as evaluate, patch.object(agent, "_current_subject_finished", return_value=True):
            agent._run_raven_subject_safely(subject)

        run_step.assert_called_once()
        apply_action.assert_called_once_with({"answer": [6, 8, 7]})
        evaluate.assert_called_once_with()

    def test_raven_test_submission_is_one_shot(self) -> None:
        """test 会屏蔽对错并结算；首个有效答案提交后不得再发第二个答案。"""
        agent = PreliminaryBaselineAgent(stub=None, channel=None)
        agent.action_space = {"key": "answer"}
        subject = {"task_type": "raven", "task_data": "image"}
        with patch.object(agent, "_raven_current_subject_index", side_effect=[4, 4, 4, 4]), patch.object(
            agent, "_call_struct", return_value=object()
        ), patch(
            "arenaagent.preliminary_baseline_agent.preliminary_baseline_agent.parse_struct_to_data",
            return_value={"key": "answer"},
        ), patch.object(agent, "_get_response_from_task", return_value={}), patch.object(
            agent, "run_step", return_value={"answer": [7, 8, 7]}
        ) as run_step, patch.object(
            agent, "_apply_action", return_value={"answer_right": False}
        ) as apply_action, patch.object(agent, "_evaluate_subject", return_value={}), patch.object(
            agent, "_RAVEN_SUBJECT_SETTLE_WINDOW_SECONDS", 0.0
        ), patch("arenaagent.preliminary_baseline_agent.preliminary_baseline_agent.time.sleep"):
            agent._run_raven_subject_safely(subject)

        self.assertEqual(run_step.call_count, 1)
        apply_action.assert_called_once_with({"answer": [7, 8, 7]})

    def test_raven_init_prefetch_skips_tongsim_character_spawn(self) -> None:
        agent = PreliminaryBaselineAgent(stub=None, channel=None)
        spawn_info = {
            "name": "raven-agent",
            "spawn_loc": "[0, 0, 0]",
            "spawn_rot": "[0, 0, 0]",
        }
        raven_subject = {"task_type": "raven", "task_data": "encoded-image"}

        with patch.object(agent, "_get_subject_from_task", return_value=raven_subject), patch(
            "arenaagent.vlm_agent.vlm_agent.ClientFactory.build", return_value=object()
        ), patch(
            "arenaagent.vlm_agent.vlm_agent.TongSimGrpcClient"
        ) as tongsim_client:
            agent.init(spawn_info)

        tongsim_client.assert_not_called()
        self.assertIsNone(agent.character_id)
        self.assertIsNone(agent.semantic_mapper)
        self.assertEqual(agent._prefetched_subject, raven_subject)

    def test_raven_late_subject_builds_bounded_dedicated_client(self) -> None:
        agent = PreliminaryBaselineAgent(stub=None, channel=None)
        primary_client = object()
        dedicated_client = object()
        agent.vlm_client = primary_client

        with patch(
            "arenaagent.preliminary_baseline_agent.preliminary_baseline_agent."
            "build_raven_vision_client_from_env",
            return_value=dedicated_client,
        ) as build:
            agent._ensure_task_strategy({"task_type": "raven", "subject": "matrix"})
            agent._ensure_task_strategy({"task_type": "raven", "subject": "matrix"})

        self.assertIs(agent.vlm_client, dedicated_client)
        self.assertTrue(agent._raven_vlm_client_initialized)
        build.assert_called_once_with()

    def test_non_raven_init_still_spawns_tongsim_character(self) -> None:
        agent = PreliminaryBaselineAgent(stub=None, channel=None)
        spawn_info = {
            "name": "counting-agent",
            "spawn_loc": "[1, 2, 3]",
            "spawn_rot": "[0, 0, 90]",
        }
        fake_tongsim = SimpleNamespace(spawn_character=lambda *args: "character-1")

        with patch.object(
            agent,
            "_get_subject_from_task",
            return_value={"task_type": "counting"},
        ), patch(
            "arenaagent.vlm_agent.vlm_agent.ClientFactory.build", return_value=object()
        ), patch(
            "arenaagent.vlm_agent.vlm_agent.TongSimGrpcClient",
            return_value=fake_tongsim,
        ), patch(
            "arenaagent.vlm_agent.vlm_agent.SemanticMapper", return_value=object()
        ):
            agent.init(spawn_info)

        self.assertEqual(agent.character_id, "character-1")
        self.assertIsNotNone(agent.semantic_mapper)

    def test_deepseek_v4_pro_is_built_as_independent_text_client(self) -> None:
        sentinel = object()
        environment = {
            "RAVEN_ENABLE_TEXT_VERIFIER": "1",
            "RAVEN_TEXT_API_KEY": "test-only-key",
            "DEEPSEEK_API_KEY": "",
            "RAVEN_TEXT_MODEL": "",
            "RAVEN_TEXT_API_BASE": "",
        }
        with patch.dict("os.environ", environment, clear=False), patch(
            "arenaagent.preliminary_baseline_agent.aux_client.ClientFactory.build",
            return_value=sentinel,
        ) as build:
            client = build_raven_text_client_from_env()
        self.assertIs(client, sentinel)
        client_type, cfg = build.call_args.args
        self.assertEqual(client_type, "openai")
        self.assertEqual(cfg.name, "deepseek-v4-pro")
        self.assertEqual(cfg.api_base, "https://api.deepseek.com")
        self.assertEqual(cfg.request_timeout_seconds, 35.0)
        self.assertEqual(cfg.native_max_retries, 0)

    def test_deepseek_verifier_stays_disabled_without_its_own_key(self) -> None:
        environment = {
            "RAVEN_ENABLE_TEXT_VERIFIER": "1",
            "RAVEN_TEXT_API_KEY": "",
            "DEEPSEEK_API_KEY": "",
        }
        with patch.dict("os.environ", environment, clear=False), patch(
            "arenaagent.preliminary_baseline_agent.aux_client.ClientFactory.build"
        ) as build:
            client = build_raven_text_client_from_env()
        self.assertIsNone(client)
        build.assert_not_called()

    def test_independent_text_verifier_only_receives_target_question_without_images(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.messages = []

            def invoke(self, message, max_retries=2):
                del max_retries
                self.messages.append(message)
                candidates = [
                    {"id": index, "confidence": 0.65 if index == 6 else 0.05}
                    for index in range(1, 9)
                ]
                return SimpleNamespace(
                    text=json.dumps({"question": 3, "answer": 6, "candidates": candidates})
                )

        client = FakeClient()
        summaries = [{"question": index, "marker": f"q{index}"} for index in range(1, 4)]
        votes, _ = ask_text_verifier(
            client,
            summaries,
            [ModelVote(question=3, answer=7, confidence=0.7)],
            target_questions=[3],
        )
        self.assertEqual([(vote.question, vote.answer) for vote in votes], [(3, 6)])
        self.assertEqual(len(client.messages), 1)
        content = client.messages[0][0]["content"]
        self.assertIsInstance(content, str)
        self.assertIn('"marker": "q3"', content)
        self.assertNotIn('"marker": "q1"', content)
        self.assertNotIn("image_url", content)

    def test_three_questions_use_one_original_image_and_one_direct_k3_answer(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.messages: list[list[dict]] = []

            def invoke(self, message, max_retries=2):
                del max_retries
                self.messages.append(message[0]["content"])
                return SimpleNamespace(text="答案：1,2,3")

        client = FakeClient()
        groups = [[Image.new("L", (32, 32), "white") for _ in range(16)] for _ in range(3)]
        summaries = [{"question": index, "cv_marker": f"summary-{index}"} for index in range(1, 4)]
        whole_image = Image.new("RGB", (320, 160), "white")
        votes, _ = ask_visual_reasoner(client, groups, summaries, whole_image=whole_image)
        self.assertEqual([vote.answer for vote in votes], [1, 2, 3])
        self.assertEqual(len(client.messages), 1)
        self.assertEqual(sum(item["type"] == "image_url" for item in client.messages[0]), 1)
        self.assertIn("结构化属性分析", client.messages[0][0]["text"])
        self.assertNotIn("STRUCTURED_EVIDENCE=", client.messages[0][0]["text"])
        self.assertIn("答案：X,X,X", client.messages[0][0]["text"])

    def test_missing_original_canvas_never_falls_back_to_split_question_images(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            def invoke(self, message, max_retries=1):
                del message, max_retries
                self.calls += 1
                return SimpleNamespace(text="答案：1,2,3")

        client = FakeClient()
        groups = [[Image.new("L", (32, 32), "white") for _ in range(16)] for _ in range(3)]
        summaries = [{"question": index} for index in range(1, 4)]
        votes, raw = ask_visual_reasoner(client, groups, summaries)
        self.assertEqual(votes, [])
        self.assertEqual(raw, "")
        self.assertEqual(client.calls, 0)

    def test_conflicting_second_question_is_selected_for_targeted_revision(self) -> None:
        attempt = {
            "visual_votes": [
                {"question": 1, "answer": 3, "confidence": 0.85, "candidate_confidences": {3: 0.85, 2: 0.15}},
                {"question": 2, "answer": 4, "confidence": 0.8, "candidate_confidences": {4: 0.8, 6: 0.2}},
                {"question": 3, "answer": 6, "confidence": 0.82, "candidate_confidences": {6: 0.82, 3: 0.18}},
            ],
            "rule_top_candidates": [[3, 6, 4], [4, 6, 5], [5, 1, 6]],
            "legacy_top_candidates": [[3, 1, 2], [5, 6, 3], [5, 3, 1]],
        }
        self.assertEqual(select_suspect_questions([attempt]), [2])

    def test_partial_results_are_merged_across_rounds_with_text_fallback(self) -> None:
        attempts = [
            {
                "visual_votes": [
                    {"question": 1, "answer": 3, "confidence": 0.8},
                    {"question": 2, "answer": 5, "confidence": 0.6},
                ],
                "text_votes": [{"question": 3, "answer": 7, "confidence": 0.55}],
            },
            {
                "visual_votes": [{"question": 2, "answer": 6, "confidence": 0.9}],
                "text_votes": [],
            },
        ]
        restored = restore_prior_model_votes(attempts)
        self.assertEqual([(vote.question, vote.answer) for vote in restored], [(1, 3), (2, 6), (3, 7)])

    def test_missing_question_is_the_only_target_for_next_revision(self) -> None:
        attempt = {
            "visual_votes": [
                {"question": 1, "answer": 3, "confidence": 0.9},
                {"question": 2, "answer": 6, "confidence": 0.9},
            ],
            "text_votes": [],
            "rule_top_candidates": [[3], [6], [5, 1, 6]],
            "legacy_top_candidates": [[3], [6], [5, 3, 1]],
        }
        self.assertEqual(select_suspect_questions([attempt]), [3])

    def test_visual_client_is_not_reused_as_text_verifier(self) -> None:
        visual_client = object()
        solver = HybridRavenSolver(vlm_client=visual_client)
        self.assertIsNone(solver.text_client)

    def test_missing_visual_response_preserves_legacy_first_choice(self) -> None:
        groups = [[Image.new("L", (32, 32), "white") for _ in range(16)] for _ in range(3)]
        with patch(
            "arenaagent.preliminary_baseline_agent.tasks.raven.solver.ask_visual_reasoner",
            return_value=([], "quota exceeded"),
        ):
            result = HybridRavenSolver(vlm_client=object()).solve(
                groups,
                legacy_ranked=[[5, 8, 2], [8, 8, 2]],
            )
        self.assertEqual(result.selected_answers, [5, 8, 2])
        self.assertTrue(result.diagnostics["visual_fallback_to_legacy"])

    def test_initial_visual_override_requires_high_self_reported_confidence(self) -> None:
        groups = [[Image.new("L", (32, 32), "white") for _ in range(16)] for _ in range(3)]
        votes = [
            ModelVote(question=1, answer=8, confidence=0.82),
            ModelVote(question=2, answer=6, confidence=0.88),
            ModelVote(question=3, answer=6, confidence=0.85),
        ]
        with patch(
            "arenaagent.preliminary_baseline_agent.tasks.raven.solver.ask_visual_reasoner",
            return_value=(votes, "{}"),
        ), patch(
            "arenaagent.preliminary_baseline_agent.tasks.raven.solver.induce_rules"
        ) as induce:
            induce.side_effect = [
                RuleInductionResult([0, 0, 0, 0, 0, 0, 0, 1], 0.8, []),
                RuleInductionResult([0, 0, 0, 0, 0, 1, 0, 0], 0.8, []),
                RuleInductionResult([0, 0, 0, 0, 0, 1, 0, 0], 0.8, []),
            ]
            result = HybridRavenSolver(vlm_client=object()).solve(
                groups,
                legacy_ranked=[[5, 5, 5], [5, 7, 7], [5, 6, 6]],
            )
        self.assertEqual(result.selected_answers, [5, 6, 6])
        self.assertEqual(result.diagnostics["conservative_legacy_restores"][0]["question"], 1)

    def test_subject_seven_confidence_gate_repairs_276_to_272(self) -> None:
        groups = [[Image.new("L", (32, 32), "white") for _ in range(16)] for _ in range(3)]
        votes = [
            ModelVote(question=1, answer=2, confidence=0.85),
            ModelVote(question=2, answer=7, confidence=0.90),
            ModelVote(question=3, answer=6, confidence=0.80),
        ]
        with patch(
            "arenaagent.preliminary_baseline_agent.tasks.raven.solver.ask_visual_reasoner",
            return_value=(votes, "答案：2,7,6；置信度：0.85,0.90,0.80"),
        ), patch(
            "arenaagent.preliminary_baseline_agent.tasks.raven.solver.induce_rules"
        ) as induce:
            induce.side_effect = [
                RuleInductionResult([0, 0, 0, 0, 0, 1, 0, 0], 0.8, []),
                RuleInductionResult([0, 1, 0, 0, 0, 0, 0, 0], 0.8, []),
                RuleInductionResult([0, 0, 0, 0, 0, 1, 0, 0], 0.8, []),
            ]
            result = HybridRavenSolver(vlm_client=object()).solve(
                groups,
                legacy_ranked=[[7, 7, 2], [5, 7, 2], [2, 7, 2]],
            )
        self.assertEqual(result.selected_answers, [2, 7, 2])
        self.assertEqual(
            [entry["question"] for entry in result.diagnostics["conservative_legacy_restores"]],
            [3],
        )

    def test_high_visual_confidence_cannot_override_strong_unsupported_legacy(self) -> None:
        groups = [[Image.new("L", (32, 32), "white") for _ in range(16)] for _ in range(3)]
        votes = [
            ModelVote(question=1, answer=4, confidence=0.95),
            ModelVote(question=2, answer=8, confidence=0.90),
            ModelVote(question=3, answer=1, confidence=0.95),
        ]
        with patch(
            "arenaagent.preliminary_baseline_agent.tasks.raven.solver.ask_visual_reasoner",
            return_value=(votes, "答案：4,8,1；置信度：0.95,0.90,0.95"),
        ), patch(
            "arenaagent.preliminary_baseline_agent.tasks.raven.solver.induce_rules"
        ) as induce:
            induce.side_effect = [
                RuleInductionResult([0, 0, 0, 0, 0, 0, 0, 1], 0.8, []),
                RuleInductionResult([0, 0, 0, 0, 0, 0, 1, 0], 0.8, []),
                RuleInductionResult([0, 1, 0, 0, 0, 0, 0, 0], 0.8, []),
            ]
            result = HybridRavenSolver(vlm_client=object()).solve(
                groups,
                legacy_ranked=[[5, 8, 2], [7, 8, 2], [3, 8, 2]],
            )
        self.assertEqual(result.selected_answers, [5, 8, 2])

    def test_subject_four_strong_legacy_blocks_correlated_visual_rule_error(self) -> None:
        groups = [[Image.new("L", (32, 32), "white") for _ in range(16)] for _ in range(3)]
        votes = [
            ModelVote(question=1, answer=7, confidence=0.95),
            ModelVote(question=2, answer=4, confidence=0.80),
            ModelVote(question=3, answer=6, confidence=0.75),
        ]
        with patch(
            "arenaagent.preliminary_baseline_agent.tasks.raven.solver.ask_visual_reasoner",
            return_value=(votes, "答案：7,4,6；置信度：0.95,0.80,0.75"),
        ), patch(
            "arenaagent.preliminary_baseline_agent.tasks.raven.solver.induce_rules"
        ) as induce:
            induce.side_effect = [
                RuleInductionResult([0, 0, 0, 0, 0, 0, 1, 0], 0.9, []),
                RuleInductionResult([0, 0, 0, 0, 0, 0, 1, 0], 0.8, []),
                RuleInductionResult([0, 0, 0, 0, 0, 0, 1, 0], 0.8, []),
            ]
            result = HybridRavenSolver(vlm_client=object()).solve(
                groups,
                legacy_ranked=[
                    [6, 8, 7],
                    [6, 8, 1],
                    [6, 8, 6],
                    [8, 8, 7],
                    [6, 8, 3],
                    [3, 8, 7],
                    [6, 8, 4],
                    [6, 8, 5],
                    [6, 8, 8],
                    [7, 8, 7],
                ],
            )
        self.assertEqual(result.selected_answers, [6, 8, 7])

    def test_raven_subject_index_uses_available_rpc(self) -> None:
        agent = PreliminaryBaselineAgent(stub=None, channel=None)
        with patch.object(
            agent,
            "_call_struct",
            return_value=SimpleNamespace(value=4),
        ) as rpc:
            self.assertEqual(agent._raven_current_subject_index(), 4)
        self.assertEqual(rpc.call_args.args[0], "get_current_subject_index")

    def test_subject_log_redacts_base64_image(self) -> None:
        payload = "iVBORw0KGgo" + "A" * 1000
        summary = summarize_subject_for_log({"task_type": "raven", "task_data": payload})
        self.assertNotIn(payload, str(summary))
        self.assertIn("image/base64", summary["task_data"])

    def test_raven_pure_vision_submits_the_vision_answer(self) -> None:
        """纯视觉：三道题各问一次，拼成最终提交的三位答案。"""
        with tempfile.TemporaryDirectory() as tmp:
            canvas = Path(tmp) / "canvas.png"
            Image.new("RGB", (2376, 1200), "white").save(canvas)
            agent = _PureVisionAgent(str(canvas))
            client = _FakeVisionClient('{"answers": [3]}', answers=[3, 2, 8])
            agent.vlm_client = client

            with patch.dict(
                os.environ,
                {"RAVEN_PURE_VISION": "1", "RAVEN_VISION_PASSES": "1"},
            ):
                result = handle(agent, {}, {})

        self.assertEqual(result, {"answer": [3, 2, 8]})
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(agent._raven_last_diagnostics["canvas_size"], (2376, 1200))

    def test_raven_pre_submit_ensemble_votes_per_position(self) -> None:
        """test 只能提交一次：三路快答必须在提交前逐位收敛。"""
        from arenaagent.preliminary_baseline_agent.tasks.raven.pure_vision import solve_pure_vision

        runs = [
            ([5, 8, 6], {"per_question": []}),
            ([5, 8, 2], {"per_question": []}),
            ([8, 8, 2], {"per_question": []}),
        ]
        with patch.dict(os.environ, {"RAVEN_VISION_PASSES": "3"}), patch(
            "arenaagent.preliminary_baseline_agent.tasks.raven.pure_vision._solve_pure_vision_once",
            side_effect=runs,
        ) as solve_once:
            answers, diagnostics = solve_pure_vision(object(), "unused.png")

        self.assertEqual(answers, [5, 8, 2])
        self.assertEqual(diagnostics["vote_history"], [[5, 8, 6], [5, 8, 2], [8, 8, 2]])
        self.assertEqual(solve_once.call_count, 3)

    def test_raven_pure_vision_asks_one_question_per_request(self) -> None:
        """整张画布让三道题互相抢注意力；改成每题一次请求，每次带两块图（矩阵 + 候选）。"""
        with tempfile.TemporaryDirectory() as tmp:
            canvas = Path(tmp) / "canvas.png"
            Image.new("RGB", (2376, 1200), "white").save(canvas)
            agent = _PureVisionAgent(str(canvas))
            client = _FakeVisionClient('{"answers": [1]}', answers=[1, 2, 3])
            agent.vlm_client = client

            with patch.dict(
                os.environ,
                {"RAVEN_PURE_VISION": "1", "RAVEN_VISION_PASSES": "1"},
            ):
                handle(agent, {}, {})

        self.assertEqual(len(client.calls), 3)
        for call in client.calls:
            content = call[0]["content"]
            self.assertEqual(len(content), 3)  # 一段文字 + 矩阵图 + 候选图
            sizes = []
            for item in content[1:]:
                sent_url = item["image_url"]["url"]
                with Image.open(io.BytesIO(base64.b64decode(sent_url.split(",", 1)[1]))) as sent:
                    sizes.append(sent.size)
            self.assertTrue(all(size[0] < 2376 // 2 for size in sizes))
            # 矩阵在上、候选在下：切成两张后各自都比整块题图矮
            self.assertLess(sizes[0][1], 1200)
            self.assertLess(sizes[1][1], 1200)
        per_question = agent._raven_last_diagnostics["per_question"]
        self.assertEqual([item["question"] for item in per_question], [1, 2, 3])
        for item in per_question:
            self.assertEqual(len(item["matrix_size"]), 2)
            self.assertEqual(len(item["option_size"]), 2)

    def test_raven_pure_vision_downscales_panels_only_when_asked(self) -> None:
        """降采样是可选开关：默认发整块题图，设了 RAVEN_IMAGE_MAX_SIDE 才缩。"""
        with tempfile.TemporaryDirectory() as tmp:
            canvas = Path(tmp) / "canvas.png"
            Image.new("RGB", (2376, 1200), "white").save(canvas)
            agent = _PureVisionAgent(str(canvas))
            client = _FakeVisionClient('{"answers": [1, 2, 3]}')
            agent.vlm_client = client

            with patch.dict(
                os.environ,
                {
                    "RAVEN_IMAGE_MAX_SIDE": "600",
                    "RAVEN_PURE_VISION": "1",
                    "RAVEN_VISION_PASSES": "1",
                },
            ):
                handle(agent, {}, {})

        for item in client.calls[0][0]["content"][1:]:
            sent_url = item["image_url"]["url"]
            with Image.open(io.BytesIO(base64.b64decode(sent_url.split(",", 1)[1]))) as sent:
                self.assertLessEqual(max(sent.size), 600)

    def test_raven_pure_vision_splits_question_into_matrix_and_options(self) -> None:
        """矩阵与候选之间那条最宽的纯白横带就是分界；找不到时按固定比例兜底。"""
        from arenaagent.preliminary_baseline_agent.tasks.raven.pure_vision import split_question

        panel = Image.new("RGB", (300, 400), "white")
        draw = ImageDraw.Draw(panel)
        for y in (60, 160):  # 矩阵两行的格线
            draw.rectangle([20, y, 280, y + 60], outline="black", width=2)
        for y in (300,):  # 候选行的格线
            draw.rectangle([20, y, 280, y + 60], outline="black", width=2)
        matrix, options = split_question(panel)
        # 切点必须落在矩阵最后一行（y=220）与候选行（y=300）之间的空白带里
        self.assertGreater(matrix.height, 221)
        self.assertLess(matrix.height, 300)
        self.assertEqual(matrix.width, 300)
        self.assertEqual(matrix.height + options.height, 400)

        blank = Image.new("RGB", (300, 400), "white")
        matrix_only, options_only = split_question(blank)
        self.assertEqual(matrix_only.height + options_only.height, 400)
        self.assertEqual(matrix_only.height, int(400 * 0.62))

    def test_raven_pure_vision_injects_pixel_measurements(self) -> None:
        """每题请求里必须带上系统量出的形状/填充/面积，模型才比得出候选尺寸。"""
        with tempfile.TemporaryDirectory() as tmp:
            canvas = Path(tmp) / "canvas.png"
            full = Image.new("RGB", (2376, 1200), "white")
            full.paste(_synthetic_raven_panel(), (0, 0))
            full.save(canvas)
            agent = _PureVisionAgent(str(canvas))
            client = _FakeVisionClient('{"answers": [1]}', answers=[1, 2, 3])
            agent.vlm_client = client

            with patch.dict(
                os.environ,
                {"RAVEN_PURE_VISION": "1", "RAVEN_VISION_PASSES": "1"},
            ):
                handle(agent, {}, {})

        prompts = {}
        for call in client.calls:
            text = "".join(
                part.get("text", "") for part in call[0]["content"] if part.get("type") == "text"
            )
            match = re.search(r"第\s*(\d)\s*题", text)
            if match:
                prompts[int(match.group(1))] = text
        self.assertIn(1, prompts)
        self.assertIn("像素统计", prompts[1])
        self.assertIn("五边形", prompts[1])
        self.assertIn("矩阵（行,列）", prompts[1])

    def test_raven_measure_keeps_four_objects_and_position_slots(self) -> None:
        """Q3 的 3/4 图形格不得再被截成最多两个。"""
        from arenaagent.preliminary_baseline_agent.tasks.raven.measure import describe_shapes

        panel = _synthetic_raven_panel()
        draw = ImageDraw.Draw(panel)
        draw.rectangle([54, 812, 208, 966], fill="white", outline="black", width=3)
        for x, y in ((82, 840), (150, 840), (82, 908), (150, 908)):
            draw.ellipse([x, y, x + 32, y + 32], fill="gray", outline="black", width=2)

        description = describe_shapes(panel, 3)

        self.assertIn("1. 【共4个】", description)
        self.assertIn("[左上]", description)
        self.assertIn("[右下]", description)

    def test_raven_pure_vision_refuses_an_unparsable_answer(self) -> None:
        """拿不到三个 1~8 的编号就不要提交，避免把噪声当成答案。"""
        with tempfile.TemporaryDirectory() as tmp:
            canvas = Path(tmp) / "canvas.png"
            Image.new("RGB", (600, 300), "white").save(canvas)
            agent = _PureVisionAgent(str(canvas))
            agent.vlm_client = _FakeVisionClient("这三道题我无法判断。")

            with patch.dict(
                os.environ,
                {"RAVEN_PURE_VISION": "1", "RAVEN_VISION_PASSES": "1"},
            ):
                result = handle(agent, {}, {})

        self.assertEqual(result.get("result"), "failed")

    def test_canvas_crop_stays_in_memory_by_default(self) -> None:
        fake_groups = [[Image.new("RGB", (32, 32), "white") for _ in range(16)] for _ in range(3)]
        with patch(
            "arenaagent.vlm_agent.raven_skill.crop_group_image_to_pil_groups",
            return_value=fake_groups,
        ), patch("arenaagent.vlm_agent.raven_skill.crop_group_image_to_subplots") as disk_crop:
            result = group_image_to_raven_list(Image.new("RGB", (64, 64), "white"))
        self.assertEqual([len(group) for group in result or []], [16, 16, 16])
        disk_crop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
