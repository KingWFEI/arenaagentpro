from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from google.protobuf import struct_pb2
from loguru import logger

from arenaagent.agent_base import parse_struct_to_data
from arenaagent.builder import Register
from arenaagent.generated.arena.message import basic_type_pb2
from arenaagent.preliminary_baseline_agent.task_registry import create_task_strategy, supported_task_types
from arenaagent.preliminary_baseline_agent.task_runtime import TaskContext, normalize_task_type
from arenaagent.preliminary_baseline_agent.tasks.base import TaskStrategy
from arenaagent.preliminary_baseline_agent.tasks.counting.review_client import (
    build_counting_review_client_from_env,
)
from arenaagent.preliminary_baseline_agent.tasks.counting.runtime import run_counting_subject
from arenaagent.preliminary_baseline_agent.tasks.jigsaw.runner import run_dedicated_jigsaw
from arenaagent.preliminary_baseline_agent.tasks.npc.runner import run_npc_fast_step
from arenaagent.preliminary_baseline_agent.tasks.npc.strategy import NpcStrategy
from arenaagent.preliminary_baseline_agent.tasks.raven.vision_client import (
    build_raven_vision_client_from_env,
)
from arenaagent.preliminary_baseline_agent.tasks.tidyroom.survey import run_scene_survey
from arenaagent.preliminary_baseline_agent.tasks.tidyroom.vlm_client import (
    build_tidyroom_vision_client_from_env,
)
from arenaagent.utils.configclass import configclass
from arenaagent.vlm_agent.vlm_agent import VLMAgent, VLMAgentCfg


@configclass
class PreliminaryBaselineAgentCfg(VLMAgentCfg):
    name: str = "preliminary_baseline_agent"
    counting_max_model_calls: int = 2
    counting_move_distance: float = 80.0
    counting_post_turn_settle_seconds: float = 0.12
    counting_capture_max_attempts: int = 3
    counting_image_max_width: int = 1280
    counting_perception_width: int = 1280
    counting_perception_height: int = 720
    counting_clock_closeups: int = 2
    counting_clock_image_max_width: int = 2000
    counting_max_recovery_submissions: int = 7
    counting_corner_move_distance: float = 80.0
    counting_occlusion_clearance: float = 70.0
    # Experimental corner start; the conservative spawn panorama is the scored default.
    counting_use_corner_route: bool = False


@Register("preliminary_baseline_agent")
class PreliminaryBaselineAgent(VLMAgent):
    """Shared VLM runtime with isolated strategy state for each preliminary task."""

    _TIDYROOM_MAX_LOCAL_STEPS = 64
    _NPC_MAX_LOCAL_STEPS = 8
    # test 环境中一次合法提交就会结算，answer_right=True 只表示请求被接受，并非泄露
    # 正确性。最多三次仅用于“尚未提交时”的网络/解析失败恢复；首个有效答案只提交一次。
    _RAVEN_MAX_INFERENCE_ATTEMPTS = 3
    _RAVEN_MIN_CALL_INTERVAL_SECONDS = 0.0
    _RAVEN_SUBJECT_SETTLE_WINDOW_SECONDS = 12.0
    _TIDYROOM_POST_TURN_SETTLE_SECONDS = 0.75
    _TIDYROOM_POST_TURN_MAX_ATTEMPTS = 10
    # 不再用固定数量门槛：鞋柜等视角本来就只看得见几件物品。未就绪帧由
    # 分割图非黑、非 RGB 副本，以及对象 ID 连续两帧一致来排除。
    _TIDYROOM_MIN_UNIFIED_OBJECTS = 1
    _TIDYROOM_SEGMENTATION_GAP_RATIO = 0.20
    _TIDYROOM_SEGMENTATION_MIN_GAP = 3
    _TIDYROOM_MIN_MAPPED_PIXEL_COVERAGE = 0.50
    # 新版任务的初始正向视野已经覆盖主要物品和目的地，只保留这一帧。
    _TIDYROOM_SURVEY_MAX_FRAMES = 1

    def __init__(
        self,
        stub,
        channel,
        cfg: PreliminaryBaselineAgentCfg | None = None,
        sleep_between_steps: float = 2.0,
    ) -> None:
        super().__init__(
            stub=stub,
            channel=channel,
            cfg=cfg or PreliminaryBaselineAgentCfg(),
            sleep_between_steps=sleep_between_steps,
        )
        self._task_strategy: TaskStrategy | None = None
        self._active_task_type = ""
        self._active_subject_key: tuple[str, str] | None = None
        self._task_context: TaskContext | None = None
        self._last_executed_action_name = ""
        self._last_executed_action: dict[str, Any] = {}
        self._spawn_xy: tuple[float, float] | None = None
        self._spawn_yaw: float | None = None
        self._tidyroom_post_turn_diagnostic_sequence = 0
        self._tidyroom_post_turn_perception_blocked = False
        self._tidyroom_retake_missing_frame = False
        self._tidyroom_survey_frames: list[str] = []
        self.raven_text_client = None
        self._raven_text_client_initialized = False
        self._raven_vlm_client_initialized = False
        self.npc_text_client = None
        self._npc_text_client_initialized = False
        self.counting_review_client = None
        self._counting_review_client_initialized = False
        self.tidyroom_vlm_client = None
        self._tidyroom_vlm_client_initialized = False
        self._jigsaw_attempt_subject_key: tuple[str, str] | None = None
        self._prefetched_subject: dict[str, Any] | None = None
        self._prefetched_subject_at: datetime | None = None
        self._skip_tongsim_character_init = False

    def init(self, opt: dict[str, Any]) -> None:
        """预取题型并保存真实出生坐标，避免瑞文创建无用的 3D 角色。"""
        try:
            prefetched = self._get_subject_from_task()
        except Exception as exc:
            logger.debug("Could not prefetch subject before runtime initialization: {}", exc)
            prefetched = {}
        if isinstance(prefetched, dict) and prefetched:
            self._prefetched_subject = prefetched
            self._prefetched_subject_at = datetime.now()
            self._skip_tongsim_character_init = normalize_task_type(prefetched) == "raven"
        if self._skip_tongsim_character_init:
            # Bound the correction path as well as the fast path. The Moonshot
            # account used by the 912 runtime only permits one in-flight request.
            try:
                visual_timeout = max(float(os.getenv("RAVEN_VISUAL_TIMEOUT_SECONDS", "20")), 1.0)
            except ValueError:
                visual_timeout = 20.0
            self.cfg.vlm_config.client_cfg.request_timeout_seconds = visual_timeout
            self.cfg.vlm_config.client_cfg.native_max_retries = 0
        super().init(opt)
        if self._skip_tongsim_character_init:
            # Raven uses a task-specific K3 whole-image client with thinking
            # disabled so one three-question request stays inside the score budget.
            self.vlm_client = build_raven_vision_client_from_env() or self.vlm_client
            self._raven_vlm_client_initialized = True
        raw_location = opt.get("spawn_loc")
        try:
            location = json.loads(raw_location) if isinstance(raw_location, str) else raw_location
            if isinstance(location, (list, tuple)) and len(location) >= 2:
                self._spawn_xy = float(location[0]), float(location[1])
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Could not parse agent spawn location for route planning: {}", raw_location)
        raw_rotation = opt.get("spawn_rot")
        try:
            rotation = json.loads(raw_rotation) if isinstance(raw_rotation, str) else raw_rotation
            if isinstance(rotation, (list, tuple)) and len(rotation) >= 3:
                self._spawn_yaw = float(rotation[2])
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Could not parse agent spawn rotation for counting route: {}", raw_rotation)
        self._seed_tidyroom_start_position()

    @property
    def active_task_type(self) -> str:
        return self._active_task_type

    def _should_handle_piece_transfer(self) -> bool:
        return False

    def _vlm_client_for_current_task(self):
        if self._active_task_type == "tidyroom" and self.tidyroom_vlm_client is not None:
            return self.tidyroom_vlm_client
        return super()._vlm_client_for_current_task()

    def _run_subject(self) -> None:
        """整理房间使用本地快速循环，其他四类任务保持原有服务循环。"""
        subject = self._prefetched_subject or self._get_subject_from_task()
        self._prefetched_subject = None
        task_type = normalize_task_type(subject)
        if task_type == "counting":
            run_counting_subject(self, subject)
            return
        if task_type == "raven":
            # 预取发生在"agent 一连上"的时刻，而赛题端要等 wait_after_first_agent_secs
            # 才通知开始答题；这中间的窗口里 get_subject 可能还是**上一题**的题图，
            # 于是我们解旧图、赛题端按新题判分（实测 slot 2 取到 slot 1 的图，三答全错）。
            # 这里的调用点已经在 is_ready_for_agent 之后，重新取一次才保证是本题。
            self._run_raven_subject_safely(self._get_subject_from_task())
            return
        if task_type == "npc":
            self._run_npc_subject_fast(subject)
            return
        if task_type != "tidyroom":
            super()._run_subject()
            return

        # action_space 已在 AgentBase.run 中读取。题目和任务响应在整理过程中
        # 不会变化，因此只读取一次，避免每个物理子动作之间重复执行四组 RPC。
        task_response = self._get_response_from_task()
        self.subject_finished = False
        logger.info("Using cached tidy-room fast loop (max_steps={})", self._TIDYROOM_MAX_LOCAL_STEPS)

        final_actions = {"submit_answer", "finish_task"}
        for local_step in range(1, self._TIDYROOM_MAX_LOCAL_STEPS + 1):
            logger.debug("Tidy-room fast-loop step {}", local_step)
            action_result = self.run_step(subject, task_response)
            self._apply_action(action_result)
            if self.subject_finished or self._last_executed_action_name in final_actions:
                break
            settle_delay = self._tidyroom_settle_delay()
            if settle_delay > 0:
                time.sleep(settle_delay)
        else:
            logger.error("Tidy-room fast loop exhausted {} steps", self._TIDYROOM_MAX_LOCAL_STEPS)
            emergency = {
                "action": "submit_answer",
                "parameters": {},
                "output": 0,
                "think": "本地快速循环达到安全步数上限，提交当前完成度。",
            }
            self._apply_action(self._execute_action_and_record(emergency))

        # 所有物品均已在提交前完成几何校验，可以立即通知赛题端评估。
        self._evaluate_subject()
    def _run_npc_subject_fast(
        self, 
        subject: dict[str, Any],
    ) -> None:
        """在单个本地循环中完成四名 NPC 访谈和最终文本判断。"""
        task_response = self._get_response_from_task()
        self.subject_finished = False

        final_actions = {"submit_answer", "finish_task"}
        accepted_final_answer = False

        logger.info(
            "Using NPC fast loop (max_steps={})",
            self._NPC_MAX_LOCAL_STEPS,
        )

        for local_step in range(1, self._NPC_MAX_LOCAL_STEPS + 1):
            logger.debug("NPC fast-loop step {}", local_step)

            action_result = self.run_step(subject, task_response)
            action_name = self._last_executed_action_name

            logger.debug(
                "NPC fast-loop step {} finished action={}",
                local_step,
                action_name,
            )

            # 最终答案只在这里向 Arena 上报一次。
            if action_name in final_actions:
                apply_response = self._apply_action(action_result)
                if apply_response.get("answer_right") is False:
                    answer_key = str(getattr(self, "action_space", {}).get("key") or "answer")
                    rejected_answer = str(action_result.get(answer_key) or "").strip()
                    strategy = self._task_strategy
                    if isinstance(strategy, NpcStrategy):
                        strategy.note_rejected_answer(rejected_answer)
                    self.subject_finished = False
                    logger.warning(
                        "NPC answer '{}' was rejected by Arena; retrying with remaining candidates",
                        rejected_answer,
                    )
                    continue
                self.subject_finished = True
                accepted_final_answer = True
                logger.info(
                    "NPC fast loop finished with final action {}",
                    action_name,
                )
                break

            # speak_to_npc 已经通过 speak_to RPC 真正完成了访谈，
            # 不再额外调用 update_action，也不进入 AgentBase 的 sleep。
            if action_name == "speak_to_npc":
                continue

            # 正常 NPC fast path 理论上不会出现其他动作。
            # 出现时停止快速循环，避免未知动作被无限重复。
            logger.error(
                "Unexpected action '{}' in NPC fast loop; aborting local loop",
                action_name,
            )
            break

        else:
            logger.error(
                "NPC fast loop exhausted {} steps without a final answer",
                self._NPC_MAX_LOCAL_STEPS,
            )

        if accepted_final_answer:
            self._evaluate_subject()
        else:
            logger.error("NPC fast loop ended without an answer accepted by Arena")

    _RAVEN_PAIR_RECORD = Path("logs/raven_subject_pairs.jsonl")
    _RAVEN_IMAGE_DIR = Path("logs/raven_subject_images")

    def _record_raven_pair(self, **payload: Any) -> None:
        """把 (subject 序号 / 题图 / 作答) 追加到 jsonl。

        为什么必须落盘：arena 的答案键只在服务端（`arena_offline/logs/arena_*.log` 里
        每个 subject 结算时打印一次），本地题图又会随临时目录被清理。09-19 那轮答对过
        528/687/254/861/272，图全丢了，现在没法把它们当验证样本；不落盘就会重演。

        没连上 arena 时直接跳过：单测会直接调 `_run_raven_subject_safely` 造数据，
        不设门就会把 mock 的 subject 序号和答案写进真实 manifest（实测过一次）。
        """
        if not getattr(self, "connected", False):
            return
        payload["ts"] = datetime.now().isoformat(timespec="seconds")
        payload["agent_id"] = self.agent_id
        try:
            self._RAVEN_PAIR_RECORD.parent.mkdir(parents=True, exist_ok=True)
            with self._RAVEN_PAIR_RECORD.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError as exc:  # noqa: BLE001 - 记录失败不能影响作答
            logger.warning("Could not append raven pair record: {}", exc)

    def _record_raven_subject_image(self, subject_index: int, subject: dict[str, Any]) -> None:
        """按内容 hash 永久保存本题题图（同名文件即同一张图，可反复覆盖）。"""
        raw_text = str(subject.get("task_data") or "")
        if not raw_text:
            return
        try:
            raw = base64.b64decode(raw_text, validate=False)
        except (ValueError, binascii.Error) as exc:
            logger.warning("Could not decode raven subject image: {}", exc)
            return
        digest = hashlib.sha256(raw).hexdigest()
        target = self._RAVEN_IMAGE_DIR / f"{digest[:12]}.png"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(raw)
        except OSError as exc:  # noqa: BLE001
            logger.warning("Could not save raven subject image: {}", exc)
        self._record_raven_pair(
            event="subject_image",
            subject_index=subject_index,
            sha256=digest,
            path=str(target),
            # 预取时刻：用来验证"取到的是不是上一题的图"（见 _run_subject 里的说明）
            prefetched_at=(
                self._prefetched_subject_at.isoformat(sep=" ", timespec="seconds")
                if self._prefetched_subject_at
                else None
            ),
        )

    def _run_raven_subject_safely(self, first_subject: dict[str, Any]) -> None:
        """瑞文纯视觉：提交前完成集成推理，首个有效答案只提交一次。

        纯视觉方案下没有本地候选排序可以回退，服务端判错时也不再用对错反馈
        去试下一个答案——那既把本地判分当 oracle，又要在限时里反复往返。
        """
        subject_index = self._raven_current_subject_index()
        logger.info(
            "Agent[{}] is solving Raven subject index {}: {}",
            self.agent_id,
            subject_index,
            {key: value for key, value in first_subject.items() if key != "task_data"},
        )
        self._record_raven_subject_image(subject_index, first_subject)
        self.action_space = parse_struct_to_data(
            self._call_struct(
                "get_action_space",
                {"agent_id": self.agent_id},
                struct_pb2.Struct.FromString,
            )
        )
        task_response = self._get_response_from_task()
        last_call_started = 0.0
        for attempt in range(1, self._RAVEN_MAX_INFERENCE_ATTEMPTS + 1):
            if attempt > 1:
                # 这里只会在上一轮没有产生可提交答案时触发。
                gap = self._RAVEN_MIN_CALL_INTERVAL_SECONDS - (time.monotonic() - last_call_started)
                if gap > 0:
                    time.sleep(gap)
            if self._raven_current_subject_index() != subject_index:
                # 视觉请求在飞行中时服务端可能已经换题；把旧答案提交到新图上等于
                # 答错一整题，直接放弃这一轮。
                logger.warning(
                    "Discarding stale Raven answer because subject moved on during reasoning (index {})",
                    subject_index,
                )
                self.subject_finished = False
                return
            last_call_started = time.monotonic()
            action = self.run_step(first_subject, task_response)
            if not isinstance(action, dict) or action.get("result") == "failed":
                # 视觉请求失败（限流、超时、解析不出）时不要提交：否则会把一次
                # 空作答记在这道题上，赛题端照样判错。
                logger.warning(
                    "Raven inference attempt {}/{} produced no usable answer: {}",
                    attempt,
                    self._RAVEN_MAX_INFERENCE_ATTEMPTS,
                    action,
                )
                continue
            apply_response = self._apply_action(action)
            logger.info(
                "Raven one-shot answer submitted after inference attempt {}/{} for subject index {} (response={})",
                attempt,
                self._RAVEN_MAX_INFERENCE_ATTEMPTS,
                subject_index,
                apply_response,
            )
            self._record_raven_pair(
                event="submit",
                subject_index=subject_index,
                attempt=attempt,
                answer=action.get("answer"),
                answer_right=apply_response.get("answer_right"),
            )
            self._evaluate_subject()
            if self._wait_for_raven_subject_to_settle():
                logger.info(
                    "Raven subject index {} settled after its one allowed submission", subject_index
                )
                # 题目已被赛题端判定完成，之后不必再空等 session 终态（见 AgentBase.run）。
                self.subject_settled = True
                return
            logger.info(
                "Raven subject index {} is not terminal yet; test submissions are one-shot, so no second answer will be sent",
                subject_index,
            )
            # AgentBase.run keeps the connection alive until the arena reaches a terminal state.
            return
        logger.error(
            "Raven subject index {} produced no usable answer after {} local inference attempts",
            subject_index,
            self._RAVEN_MAX_INFERENCE_ATTEMPTS,
        )

    def _wait_for_raven_subject_to_settle(self) -> bool:
        """等赛题端把当前这道题判结束。

        test 环境中提交后通常会结算；这个短轮询只用于识别已完成状态。即使暂未看到
        终态也绝不据此二次提交，外层会继续保持连接等待 arena 评测。
        """
        deadline = time.monotonic() + self._RAVEN_SUBJECT_SETTLE_WINDOW_SECONDS
        while time.monotonic() < deadline:
            try:
                if self._current_subject_finished():
                    return True
            except Exception as exc:  # noqa: BLE001 - 探测失败按未结束处理
                logger.debug("Raven settle probe failed: {}", exc)
                return False
            time.sleep(self.session_poll_seconds)
        return False

    def _raven_current_subject_index(self) -> int:
        """Read the current subject index through the RPC exposed by AgentBase."""
        response = self._call_struct(
            "get_current_subject_index",
            {"agent_id": self.agent_id},
            basic_type_pb2.Int32.FromString,
        )
        return int(getattr(response, "value", -1))

    def _ensure_task_strategy(self, subject: Any) -> TaskStrategy:
        task_type = normalize_task_type(subject)
        if task_type == "raven" and not self._raven_vlm_client_initialized:
            # The first subject can be unavailable during init when the agent
            # connects before the Arena flow becomes ready.  Build the bounded
            # non-thinking K3 client lazily as soon as Raven is actually known;
            # otherwise the generic startup client may have no request timeout.
            raven_client = build_raven_vision_client_from_env()
            if raven_client is not None:
                self.vlm_client = raven_client
            else:
                logger.warning(
                    "Raven bounded vision client unavailable; falling back to the primary client"
                )
            self._raven_vlm_client_initialized = True
        if task_type == "counting" and not self._counting_review_client_initialized:
            self.counting_review_client = build_counting_review_client_from_env()
            self._counting_review_client_initialized = True
        if task_type == "tidyroom" and not self._tidyroom_vlm_client_initialized:
            # 整理房间只做一次高价值单图盘点，使用独立的高速多模态模型；
            # 其余赛题继续沿用启动参数选择的主模型。
            self.tidyroom_vlm_client = build_tidyroom_vision_client_from_env() or self.vlm_client
            if self.tidyroom_vlm_client is self.vlm_client:
                logger.warning("Tidy-room fast vision client unavailable; falling back to primary client")
            self._tidyroom_vlm_client_initialized = True
        safe_subject = dict(subject) if isinstance(subject, dict) else {"subject": str(subject)}
        identity = str(
            safe_subject.get("subject_id")
            or safe_subject.get("task_id")
            or safe_subject.get("question_id")
            or safe_subject.get("subject")
            or safe_subject.get("task_prompt")
            or task_type
        )
        subject_key = (task_type, identity)
        if self._task_strategy is not None and subject_key == self._active_subject_key:
            return self._task_strategy

        if self._active_subject_key is not None:
            self._reset_shared_task_state()
        self._active_task_type = task_type
        self._active_subject_key = subject_key
        self._task_strategy = create_task_strategy(task_type)
        self._task_strategy.reset(safe_subject)
        self._seed_tidyroom_start_position()
        self._task_context = None
        logger.info(
            "Selected task strategy '{}' (supported={})",
            task_type,
            task_type in supported_task_types(),
        )
        return self._task_strategy

    def _reset_shared_task_state(self) -> None:
        """Prevent one subject's model/action history from leaking into the next."""
        self.history_messages = []
        self.last_json_parse_message = {}
        self._action_histories = []
        self._last_action_res = {}
        self._last_apply_resp = {}
        self._last_npc_reply = ""
        self._last_npc_subject = None
        self._movable_objects = []
        self._last_executed_action_name = ""
        self._tidyroom_post_turn_diagnostic_sequence = 0
        self._tidyroom_post_turn_perception_blocked = False
        self._tidyroom_retake_missing_frame = False
        self._tidyroom_survey_frames: list[str] = []

    def _remember_survey_frame(self, image_b64: str | None) -> None:
        """保留分割稳定后的初始正向画面，供单图盘点使用。"""
        if not image_b64:
            return
        self._tidyroom_survey_frames.append(image_b64)
        del self._tidyroom_survey_frames[: -self._TIDYROOM_SURVEY_MAX_FRAMES]

    def _tidyroom_survey_is_ready(self, strategy: TaskStrategy) -> bool:
        """初始正向画面到齐后立即允许盘点。"""
        required_frames = getattr(strategy.scanner, "turns_required", 0) + 1
        return bool(
            self._active_task_type == "tidyroom"
            and strategy.survey_is_due()
            and len(self._tidyroom_survey_frames) >= required_frames
        )

    def run_step(self, subject: Any, task_response: dict[str, Any]) -> dict[str, Any]:
        strategy = self._ensure_task_strategy(subject)
        safe_subject = dict(subject) if isinstance(subject, dict) else {"subject": str(subject)}
        strategy.before_step(safe_subject, task_response)

        if self._active_task_type == "npc" and isinstance(strategy, NpcStrategy):
            return run_npc_fast_step(self, safe_subject, task_response, strategy)

        if self._active_task_type == "jigsaw" and self._jigsaw_attempt_subject_key != self._active_subject_key:
            self._jigsaw_attempt_subject_key = self._active_subject_key
            try:
                run_dedicated_jigsaw(self, safe_subject)
            except Exception as exc:
                logger.exception("Dedicated jigsaw solver failed; using existing visual policy: {}", exc)
            else:
                return self._handle_finish(
                    {},
                    {"think": "拼图专用求解器完成三块拼图的放置。", "output": 0},
                )

        previous_tail = self._action_histories[-1] if self._action_histories else None
        action_result = super().run_step(subject, task_response)
        if self._action_histories and self._action_histories[-1] is not previous_tail:
            action_record = self._action_histories[-1]
            strategy.after_action(
                action=action_record.get("action") or {},
                result=action_record.get("result"),
                context=self._task_context,
            )
        return action_result

    def _acquire_camera_perception(
        self,
        subject: Any,
        **kwargs: Any,
    ) -> tuple[str | None, list[dict[str, Any]], list[dict[str, Any]]]:
        """转向后只接受分割区域与可见对象映射基本一致的完整感知。"""
        is_tidyroom = normalize_task_type(subject) == "tidyroom"
        is_post_turn_full_capture = bool(
            is_tidyroom
            and self._last_executed_action_name == "turn_in_degree"
            and kwargs.get("include_images", True)
        )
        is_initial_912_survey_capture = bool(
            is_tidyroom
            and kwargs.get("include_images", True)
            and self._task_strategy is not None
            and self._task_strategy.scanner.turns_completed == 0
            and not self._tidyroom_survey_frames
            and not is_post_turn_full_capture
        )
        if not (is_post_turn_full_capture or is_initial_912_survey_capture):
            return super()._acquire_camera_perception(subject, **kwargs)

        if is_post_turn_full_capture:
            self._tidyroom_post_turn_diagnostic_sequence += 1
        sequence = self._tidyroom_post_turn_diagnostic_sequence
        capture_label = "initial" if is_initial_912_survey_capture else f"postturn_{sequence:03d}"
        self._tidyroom_post_turn_perception_blocked = False
        started_at = time.monotonic()
        previous_unified_signature: tuple[str, ...] | None = None

        for attempt in range(1, self._TIDYROOM_POST_TURN_MAX_ATTEMPTS + 1):
            logger.info(
                "Tidy-room post-turn perception sequence={} attempt={}/{}: "
                "remain stationary and wait {:.2f}s before capture",
                sequence,
                attempt,
                self._TIDYROOM_POST_TURN_MAX_ATTEMPTS,
                self._TIDYROOM_POST_TURN_SETTLE_SECONDS,
            )
            time.sleep(self._TIDYROOM_POST_TURN_SETTLE_SECONDS)
            capture_started_at = time.monotonic()
            capture_kwargs = dict(kwargs)
            capture_kwargs.update(
                is_save=True,
                include_images=True,
                save_label=f"{capture_label}_attempt_{attempt:02d}",
            )
            result = super()._acquire_camera_perception(subject, **capture_kwargs)
            capture_finished_at = time.monotonic()
            diagnostics = dict(
                getattr(self.semantic_mapper, "last_perception_diagnostics", {}) or {}
            )
            consistent, reason, gap, threshold = self._tidyroom_perception_is_consistent(
                diagnostics
            )
            if consistent and diagnostics.get("source") == "unified":
                raw_signature = diagnostics.get("visible_object_ids") or ()
                signature = tuple(str(object_id) for object_id in raw_signature)
                if not signature:
                    # 兼容只提供数量、不提供 ID 集合的自定义 mapper。
                    signature = (f"count:{diagnostics.get('visible_object_count', 0)}",)
                if signature != previous_unified_signature:
                    consistent = False
                    reason = "unified_waiting_for_stable_frame"
                    previous_unified_signature = signature
            logger.info(
                "Tidy-room post-turn perception sequence={} attempt={}/{}: "
                "capture_start={:.3f}s capture_end={:.3f}s capture_duration={:.3f}s "
                "visible={} aligned={} segmentation_regions={} gap={} threshold={} "
                "mapped_pixels={}/{} mapped_pixel_coverage={:.4f} "
                "coverage_threshold={:.2f} status={} reason={} "
                "image_label={}",
                sequence,
                attempt,
                self._TIDYROOM_POST_TURN_MAX_ATTEMPTS,
                capture_started_at - started_at,
                capture_finished_at - started_at,
                capture_finished_at - capture_started_at,
                diagnostics.get("visible_object_count", len(result[1] or [])),
                diagnostics.get("aligned_object_count", 0),
                diagnostics.get("segmentation_region_count", 0),
                gap,
                threshold,
                diagnostics.get("mapped_pixel_count", 0),
                diagnostics.get("total_pixel_count", 0),
                float(diagnostics.get("mapped_pixel_coverage", 0.0) or 0.0),
                self._TIDYROOM_MIN_MAPPED_PIXEL_COVERAGE,
                "accepted" if consistent else "retry",
                reason,
                f"{capture_label}_attempt_{attempt:02d}",
            )
            if consistent:
                self._remember_survey_frame(result[0])
                self._tidyroom_retake_missing_frame = False
                return result

        self._tidyroom_post_turn_perception_blocked = True
        logger.error(
            "Tidy-room post-turn perception sequence={} exhausted {} attempts after "
            "{:.3f}s; discard inconsistent observation and block VLM/target action",
            sequence,
            self._TIDYROOM_POST_TURN_MAX_ATTEMPTS,
            time.monotonic() - started_at,
        )
        return None, [], []

    @classmethod
    def _tidyroom_perception_is_consistent(
        cls,
        diagnostics: dict[str, Any],
    ) -> tuple[bool, str, int, int]:
        """判定 raw segmentation 与最终 ID 映射是否发生明显脱节。"""
        if diagnostics.get("source") == "unified":
            visible = max(int(diagnostics.get("visible_object_count", 0)), 0)
            if not diagnostics.get("image_present"):
                return False, "unified_image_missing", 0, cls._TIDYROOM_MIN_UNIFIED_OBJECTS
            try:
                right_nonblack = float(diagnostics.get("right_nonblack_ratio", 0.0) or 0.0)
            except (TypeError, ValueError):
                right_nonblack = 0.0
            if right_nonblack < 0.05:
                return False, "unified_segmentation_black", visible, cls._TIDYROOM_MIN_UNIFIED_OBJECTS
            try:
                left_right_difference = float(
                    diagnostics.get("left_right_difference_ratio", 1.0) or 0.0
                )
            except (TypeError, ValueError):
                left_right_difference = 0.0
            if left_right_difference < 0.20:
                return (
                    False,
                    "unified_segmentation_still_rgb",
                    visible,
                    cls._TIDYROOM_MIN_UNIFIED_OBJECTS,
                )
            if visible < cls._TIDYROOM_MIN_UNIFIED_OBJECTS:
                return False, "unified_objects_not_ready", visible, cls._TIDYROOM_MIN_UNIFIED_OBJECTS
            return True, "unified_perception_ready", 0, cls._TIDYROOM_MIN_UNIFIED_OBJECTS
        required_keys = {
            "visible_object_count",
            "aligned_object_count",
            "segmentation_region_count",
            "mapped_pixel_coverage",
        }
        if not required_keys.issubset(diagnostics):
            # 兼容测试桩或不支持诊断字段的旧 mapper；真实 SemanticMapper 始终
            # 提供这三个字段。
            return True, "diagnostics_unavailable", 0, 0

        regions = max(int(diagnostics.get("segmentation_region_count", 0)), 0)
        aligned = max(int(diagnostics.get("aligned_object_count", 0)), 0)
        if regions <= 1:
            return False, "segmentation_not_ready", max(regions - aligned, 0), 1

        gap = max(regions - aligned, 0)
        threshold = max(
            cls._TIDYROOM_SEGMENTATION_MIN_GAP,
            int(math.ceil(regions * cls._TIDYROOM_SEGMENTATION_GAP_RATIO)),
        )
        if gap >= threshold:
            return False, "visible_objects_not_synchronized", gap, threshold
        try:
            coverage = float(diagnostics.get("mapped_pixel_coverage", 0.0) or 0.0)
        except (TypeError, ValueError):
            coverage = 0.0
        if not math.isfinite(coverage) or coverage < cls._TIDYROOM_MIN_MAPPED_PIXEL_COVERAGE:
            return False, "segmentation_pixel_coverage_too_low", gap, threshold
        return True, "counts_and_coverage_consistent", gap, threshold

    def _trim_history_messages(self) -> list[dict[str, Any]]:
        """Allow a task strategy to replace expensive full-message history with compact state."""
        strategy_limit = getattr(self._task_strategy, "history_message_limit", None)
        if strategy_limit == 0:
            self.history_messages = []
            return self.history_messages
        return super()._trim_history_messages()

    def _should_use_lightweight_perception(self, subject: Any) -> bool:
        """新版整理房间先完整采集并保存初始正向画面。"""
        task_type = normalize_task_type(subject)
        if task_type == "raven":
            # 瑞文题图在 task_data 中，不需要采集 3D 场景相机。
            return True
        if task_type != "tidyroom":
            return False
        if (
            self._task_strategy is not None
            and not self._tidyroom_survey_frames
        ):
            return False
        if self._last_executed_action_name == "turn_in_degree":
            last_degree = int(
                (self._last_executed_action.get("parameters") or {}).get("degree", 0)
            )
            if last_degree == 0 and not self._tidyroom_retake_missing_frame:
                return True
            # 转向后的首份数据必须同时读取 raw segmentation 与 visible_objects，
            # 才能发现可见对象缓存只返回少量 ID 的不同步问题。
            return False
        return True

    def _should_refresh_lightweight_scene(self, subject: Any) -> bool:
        """完整建图后只做目标级刷新，避免每一步重新枚举整幅视野。"""
        task_type = normalize_task_type(subject)
        if task_type == "raven":
            return False
        if task_type != "tidyroom" or self._task_strategy is None:
            return True
        return self._task_strategy.needs_scene_perception()

    def _lightweight_empty_scene_retry_delay(self, subject: Any) -> float:
        """首帧的等待与重试由完整感知入口负责。"""
        del subject
        return 0.0

    def _lightweight_initial_perception_delay(self, subject: Any) -> float:
        """首帧立即读取一次结构化结果，不再等待或执行盲转预热。"""
        del subject
        return 0.0

    def _should_force_vlm_after_empty_lightweight_perception(self, subject: Any) -> bool:
        """空结构化结果不交给未稳定的轻量感知直接调用 VLM。"""
        del subject
        return False

    def _on_forced_vlm(self, subject: Any, reason: str) -> None:
        if normalize_task_type(subject) == "tidyroom" and self._task_strategy is not None:
            self._task_strategy.note_forced_vlm(reason)

    def _execute_action_and_record(self, action: dict[str, Any]) -> dict[str, Any]:
        self._last_executed_action_name = str(action.get("action") or "").lower()
        self._last_executed_action = dict(action)
        return super()._execute_action_and_record(action)

    def _seed_tidyroom_start_position(self) -> None:
        """把出生点注入任务内调度器，避免第一件物品按 ID 随机选择。"""
        if self._spawn_xy is None or self._active_task_type != "tidyroom":
            return
        scheduler = getattr(self._task_strategy, "scheduler", None)
        if scheduler is not None and scheduler.estimated_agent_xy is None:
            scheduler.estimated_agent_xy = self._spawn_xy
            logger.info("Tidy-room route starts from spawn position {}", self._spawn_xy)

    def _tidyroom_settle_delay(self) -> float:
        """同步动作只保留必要的短等待；会自然下落的刚体继续使用配置值。"""
        configured = max(float(self.sleep_between_steps), 0.0)
        if configured == 0:
            return 0.0
        name = self._last_executed_action_name
        parameters = self._last_executed_action.get("parameters") or {}
        if name == "move_and_take_object":
            return min(configured, 0.05)
        if name == "turn_in_degree":
            # 等待统一放到转向后完整感知入口，保证直接调用 run_step 的测试和
            # 实际快速循环遵循相同时序，同时避免重复等待。
            return 0.0
        if name in {"move_to_location", "move_to_object", "move_forward", "move_backward"}:
            return min(configured, 0.02)
        if name in {"put_down_to_location", "put_down_sth_to_location"}:
            if bool(parameters.get("disable_physics")):
                return min(configured, 0.02)
            return configured
        if name in {"move_and_put_down", "move_and_put_down_object_in_container"}:
            return configured
        return min(configured, 0.05)

    def _apply_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """整理房间的中间物理动作已直接在 TongSim 执行，无需反复上报为答案。"""
        final_actions = {"submit_answer", "finish_task"}
        if self._active_task_type == "tidyroom" and self._last_executed_action_name not in final_actions:
            self._last_apply_resp = {"deferred": True}
            logger.debug(
                "Skip arena update_action for intermediate tidy-room action {}",
                self._last_executed_action_name,
            )
            return self._last_apply_resp
        return super()._apply_action(action)

    def _handle_submit_answer(self, params: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
        """Keep ZRQ's integer counting answer without changing other tasks."""
        if self._active_task_type != "counting":
            return super()._handle_submit_answer(params, action)
        key = self.action_space.get("key") or "action"
        answer: Any = action["output"]
        answer_type = str(self.action_space.get("type") or "").strip().lower()
        if answer_type in {"int", "integer"}:
            if isinstance(answer, str):
                normalized = answer.strip().upper()
                try:
                    answer = int(normalized)
                except ValueError:
                    pass
            elif isinstance(answer, float) and answer.is_integer():
                answer = int(answer)
        return {key: answer if answer_type in {"int", "integer"} else str(answer)}

    def _build_prompt_variables(
        self,
        subject: Any,
        task_response: dict[str, Any],
        api_info: Any,
        visible_objects_info: list[dict[str, Any]],
        object_in_hand: Any,
    ) -> dict[str, Any]:
        strategy = self._ensure_task_strategy(subject)
        safe_subject = dict(subject) if isinstance(subject, dict) else {"subject": str(subject)}
        world_aabbs_by_raw_id: dict[str, dict[str, Any]] = {}
        if self.tongsim is not None:
            for raw_id in strategy.refresh_raw_ids():
                try:
                    world_aabb = self.tongsim.get_object_world_aabb(raw_id)
                    if isinstance(world_aabb, dict):
                        world_aabbs_by_raw_id[raw_id] = world_aabb
                except Exception as exc:
                    logger.warning("Could not refresh world AABB for tidy-room object {}: {}", raw_id, exc)
        context = TaskContext(
            task_type=self._active_task_type,
            subject=safe_subject,
            task_response=dict(task_response or {}),
            visible_objects=list(visible_objects_info or []),
            object_in_hand=object_in_hand,
            movable_objects=list(self._movable_objects),
            action_histories=list(self._action_histories),
            last_action_result=self._last_action_res,
            raw_to_mapped_id={
                str(raw_id): str(mapped_id)
                for raw_id, mapped_id in (
                    self.semantic_mapper.object_id_map.items() if self.semantic_mapper is not None else []
                )
            },
            world_aabbs_by_raw_id=world_aabbs_by_raw_id,
        )
        self._task_context = context
        strategy.observe(context)

        logger.debug("_last_action_res {}", self._last_action_res)
        variables = {
            "api_info": api_info,
            "example_objects_info": visible_objects_info[0] if visible_objects_info else {},
            "task_type": self._active_task_type,
            "task_text": safe_subject.get("subject") or safe_subject.get("task_prompt") or "",
            "task_prompt": "",
            "visiable_objects_info": visible_objects_info,
            "object_in_hand": object_in_hand,
            "movable_objects": list(self._movable_objects),
            "movable_object_count": len(self._movable_objects),
            "response": task_response,
            "npc_reply": self._last_npc_reply,
            "npc_subject": self._last_npc_subject or {},
            "action_res": self._serialize_prompt_status(self._last_action_res),
            "apply_resp": self._serialize_prompt_status(self._last_apply_resp),
            "action_histories": self._trim_action_histories(),
        }
        return strategy.enrich_prompt(variables, context)

    def _parse_action_from_response(self, resp: Any) -> dict[str, Any]:
        action = super()._parse_action_from_response(resp)
        if not action or self._task_strategy is None:
            return action
        validated = self._task_strategy.validate_action(action, self._task_context)
        if validated != action:
            logger.info("Task strategy rewrote action from {} to {}", action, validated)
        return validated

    def _get_local_action(
        self,
        subject: Any,
        task_response: dict[str, Any],
        prompt_variables: dict[str, Any],
    ) -> dict[str, Any] | None:
        """优先执行单题本地策略；只有返回 None 才调用视觉模型。"""
        del task_response
        if (
            normalize_task_type(subject) == "tidyroom"
            and self._tidyroom_post_turn_perception_blocked
        ):
            self._tidyroom_post_turn_perception_blocked = False
            self._tidyroom_retake_missing_frame = True
            return {
                "action": "turn_in_degree",
                "parameters": {"degree": 0},
                "output": 0,
                "think": (
                    "分割区域与可见对象列表连续不同步；已丢弃异常观察，"
                    "保持初始朝向并重新采集稳定帧。"
                ),
            }
        if self._task_strategy is None or self._task_context is None:
            return None
        if (
            normalize_task_type(subject) == "tidyroom"
            and self._task_strategy.survey_is_due()
        ):
            if not self._tidyroom_survey_is_ready(self._task_strategy):
                self._tidyroom_retake_missing_frame = True
                return {
                    "action": "turn_in_degree",
                    "parameters": {"degree": 0},
                    "output": 0,
                    "think": "初始正向分割稳定帧尚未采集，保持朝向补采后再进行单图盘点。",
                }
            self._task_strategy.note_survey_attempted()
            run_scene_survey(self._task_strategy, self)
        local_action = self._task_strategy.next_local_action(self._task_context)
        if local_action is None:
            # 升级原因是在本地决策阶段产生的，调用 VLM 前刷新纯文本状态即可，不能重复 observe。
            refreshed = self._task_strategy.enrich_prompt(prompt_variables, self._task_context)
            prompt_variables.clear()
            prompt_variables.update(refreshed)
        return local_action

    def _load_task_spec_prompts(self) -> dict[str, str]:
        """Compatibility helper for older extensions that used the JSON prompt map."""
        return {task_type: create_task_strategy(task_type).load_prompt() for task_type in supported_task_types()}
