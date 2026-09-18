from __future__ import annotations

import unittest
import json
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, ImageDraw

from arenaagent.preliminary_baseline_agent.preliminary_baseline_agent import PreliminaryBaselineAgent
from arenaagent.preliminary_baseline_agent.task_runtime import TaskContext
from arenaagent.preliminary_baseline_agent.tasks.raven.rules import induce_rules
from arenaagent.preliminary_baseline_agent.tasks.raven.ensemble import combine_question
from arenaagent.preliminary_baseline_agent.tasks.raven.experience import (
    record_successful_subject,
    relevant_experience_hints,
)
from arenaagent.preliminary_baseline_agent.tasks.raven.reasoners import (
    ask_text_verifier,
    ask_visual_reasoner,
)
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
    parse_model_votes,
    parse_single_model_vote,
    sanitize_ranked_triples,
)
from arenaagent.preliminary_baseline_agent.tasks.raven.vision import extract_question_observations
from arenaagent.agent_base import summarize_subject_for_log
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


class RavenRuntimeSafetyTests(unittest.TestCase):
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

    def test_three_questions_are_sent_as_parallel_grounded_requests(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                self.active = 0
                self.max_active = 0
                self.messages: list[list[dict]] = []

            def invoke(self, message, max_retries=2):
                del max_retries
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                    self.messages.append(message[0]["content"])
                prompt = message[0]["content"][0]["text"]
                question = int(prompt.split("QUESTION_NUMBER=", 1)[1].split(".", 1)[0])
                time.sleep(0.04)
                with self.lock:
                    self.active -= 1
                candidates = [
                    {"id": index, "confidence": 0.65 if index == question else 0.05, "mismatch": []}
                    for index in range(1, 9)
                ]
                return SimpleNamespace(
                    text=json.dumps(
                        {
                            "question": question,
                            "answer": question,
                            "candidates": candidates,
                            "rule": "auditable rule",
                            "evidence": ["image and CV agree"],
                        }
                    )
                )

        client = FakeClient()
        groups = [[Image.new("L", (32, 32), "white") for _ in range(16)] for _ in range(3)]
        summaries = [{"question": index, "cv_marker": f"summary-{index}"} for index in range(1, 4)]
        votes, _ = ask_visual_reasoner(client, groups, summaries)
        self.assertEqual([vote.answer for vote in votes], [1, 2, 3])
        self.assertEqual(len(client.messages), 3)
        self.assertGreaterEqual(client.max_active, 2)
        self.assertTrue(all(sum(item["type"] == "image_url" for item in content) == 1 for content in client.messages))
        self.assertTrue(all("STRUCTURED_EVIDENCE=" in content[0]["text"] for content in client.messages))

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

    def test_raven_uses_fast_path_then_escalates_without_faking_completion(self) -> None:
        class FakeAgent:
            subject_finished = False
            action_space = {"key": "answer"}
            _raven_image_temp_path = "ignored"

            @staticmethod
            def _get_param(params, name, default=None):
                return params.get(name, default)

        agent = FakeAgent()
        fake_groups = [[Image.new("L", (32, 32), "white") for _ in range(16)] for _ in range(3)]
        with patch("arenaagent.vlm_agent.raven_skill.resolve_raven_image_path", return_value="image.png"), patch(
            "arenaagent.vlm_agent.raven_skill.normalize_raven_image_list", return_value=fake_groups
        ), patch(
            "arenaagent.vlm_agent.raven_skill.get_raven_legacy_candidates",
            return_value=[[2, 2, 6], [2, 2, 5]],
        ), patch(
            "arenaagent.vlm_agent.raven_skill.get_raven_ranked_candidates",
            return_value=[[3, 2, 8], [2, 2, 6]],
        ) as hybrid:
            first = handle(agent, {"structure": []}, {})
            second = handle(agent, {"structure": []}, {})
        self.assertEqual(first, {"answer": [2, 2, 6]})
        self.assertEqual(second, {"answer": [3, 2, 8]})
        hybrid.assert_called_once()
        self.assertFalse(agent.subject_finished)

    def test_rejected_reasoned_answer_triggers_fresh_revision_not_next_enumeration(self) -> None:
        class FakeAgent:
            subject_finished = False
            action_space = {"key": "answer"}
            _raven_image_temp_path = "ignored"

            @staticmethod
            def _get_param(params, name, default=None):
                return params.get(name, default)

        agent = FakeAgent()
        fake_groups = [[Image.new("L", (32, 32), "white") for _ in range(16)] for _ in range(3)]
        with patch("arenaagent.vlm_agent.raven_skill.resolve_raven_image_path", return_value="image.png"), patch(
            "arenaagent.vlm_agent.raven_skill.normalize_raven_image_list", return_value=fake_groups
        ), patch(
            "arenaagent.vlm_agent.raven_skill.get_raven_legacy_candidates", return_value=[[3, 5, 5]]
        ), patch(
            "arenaagent.vlm_agent.raven_skill.get_raven_ranked_candidates",
            side_effect=[[[3, 1, 7]], [[3, 6, 6]]],
        ) as reasoner:
            first = handle(agent, {}, {})
            second = handle(agent, {}, {})
            third = handle(agent, {}, {})
        self.assertEqual(first, {"answer": [3, 5, 5]})
        self.assertEqual(second, {"answer": [3, 1, 7]})
        self.assertEqual(third, {"answer": [3, 6, 6]})
        self.assertEqual(reasoner.call_count, 2)
        self.assertIn([3, 1, 7], reasoner.call_args.kwargs["rejected_answers"])

    def test_runtime_refuses_second_ranked_combination_when_top_is_rejected(self) -> None:
        class FakeAgent:
            subject_finished = False
            action_space = {"key": "answer"}
            _raven_image_temp_path = "ignored"

            @staticmethod
            def _get_param(params, name, default=None):
                return params.get(name, default)

        agent = FakeAgent()
        fake_groups = [[Image.new("L", (32, 32), "white") for _ in range(16)] for _ in range(3)]
        with patch("arenaagent.vlm_agent.raven_skill.resolve_raven_image_path", return_value="image.png"), patch(
            "arenaagent.vlm_agent.raven_skill.normalize_raven_image_list", return_value=fake_groups
        ), patch(
            "arenaagent.vlm_agent.raven_skill.get_raven_legacy_candidates", return_value=[[3, 5, 5]]
        ), patch(
            "arenaagent.vlm_agent.raven_skill.get_raven_ranked_candidates",
            return_value=[[3, 5, 5], [3, 6, 6]],
        ):
            first = handle(agent, {}, {})
            second = handle(agent, {}, {})
        self.assertEqual(first, {"answer": [3, 5, 5]})
        self.assertEqual(second.get("result"), "failed")

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
