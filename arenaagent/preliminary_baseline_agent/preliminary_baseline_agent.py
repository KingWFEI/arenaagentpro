from __future__ import annotations

import json
import math
import time
from typing import Any

from google.protobuf import struct_pb2
from loguru import logger

from arenaagent.agent_base import parse_struct_to_data
from arenaagent.builder import Register
from arenaagent.generated.arena.message import basic_type_pb2
from arenaagent.preliminary_baseline_agent.task_registry import create_task_strategy, supported_task_types
from arenaagent.preliminary_baseline_agent.task_runtime import TaskContext, normalize_task_type
from arenaagent.preliminary_baseline_agent.tasks.base import TaskStrategy
from arenaagent.preliminary_baseline_agent.tasks.jigsaw.runner import run_dedicated_jigsaw
from arenaagent.preliminary_baseline_agent.tasks.npc.runner import run_npc_fast_step
from arenaagent.preliminary_baseline_agent.tasks.npc.strategy import NpcStrategy
from arenaagent.preliminary_baseline_agent.tasks.raven.text_client import (
    build_raven_text_client_from_env,
)
from arenaagent.utils.configclass import configclass
from arenaagent.vlm_agent.raven_skill import record_confirmed_raven_experience
from arenaagent.vlm_agent.vlm_agent import VLMAgent, VLMAgentCfg


@configclass
class PreliminaryBaselineAgentCfg(VLMAgentCfg):
    name: str = "preliminary_baseline_agent"


@Register("preliminary_baseline_agent")
class PreliminaryBaselineAgent(VLMAgent):
    """Shared VLM runtime with isolated strategy state for each preliminary task."""

    _TIDYROOM_MAX_LOCAL_STEPS = 64
    _NPC_MAX_LOCAL_STEPS = 6
    _TIDYROOM_POST_TURN_SETTLE_SECONDS = 0.25
    _TIDYROOM_POST_TURN_MAX_ATTEMPTS = 5
    _TIDYROOM_SEGMENTATION_GAP_RATIO = 0.20
    _TIDYROOM_SEGMENTATION_MIN_GAP = 3
    _TIDYROOM_MIN_MAPPED_PIXEL_COVERAGE = 0.50

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
        self._tidyroom_post_turn_diagnostic_sequence = 0
        self._tidyroom_post_turn_perception_blocked = False
        self.raven_text_client = None
        self._raven_text_client_initialized = False
        self._jigsaw_attempt_subject_key: tuple[str, str] | None = None

    def init(self, opt: dict[str, Any]) -> None:
        """保存赛题服务下发的真实出生坐标，供整理房间路线规划使用。"""
        super().init(opt)
        raw_location = opt.get("spawn_loc")
        try:
            location = json.loads(raw_location) if isinstance(raw_location, str) else raw_location
            if isinstance(location, (list, tuple)) and len(location) >= 2:
                self._spawn_xy = float(location[0]), float(location[1])
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Could not parse agent spawn location for route planning: {}", raw_location)
        self._seed_tidyroom_start_position()

    @property
    def active_task_type(self) -> str:
        return self._active_task_type

    def _should_handle_piece_transfer(self) -> bool:
        return False

    def _run_subject(self) -> None:
        """整理房间使用本地快速循环，其他四类任务保持原有服务循环。"""
        subject = self._get_subject_from_task()
        task_type = normalize_task_type(subject)
        if task_type == "raven":
            self._run_raven_subject_safely(subject)
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
                self._apply_action(action_result)
                self.subject_finished = True
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

        self._evaluate_subject()

    def _run_raven_subject_safely(self, first_subject: dict[str, Any]) -> None:
        """Discard a slow answer if the server moved to the next Raven subject.

        A remote vision call cannot be cancelled once it is in flight.  The
        server may force-evaluate the old subject meanwhile; submitting its
        result afterwards would otherwise apply that answer to the new image.
        """
        subject: dict[str, Any] = first_subject
        while not self._current_subject_finished():
            subject_index_before = self._raven_current_subject_index()
            logger.info(
                "Agent[{}] is running Raven subject index {}: {}",
                self.agent_id,
                subject_index_before,
                {key: value for key, value in subject.items() if key != "task_data"},
            )
            self.action_space = parse_struct_to_data(
                self._call_struct(
                    "get_action_space",
                    {"agent_id": self.agent_id},
                    struct_pb2.Struct.FromString,
                )
            )
            task_response = self._get_response_from_task()
            action = self.run_step(subject, task_response)
            subject_index_after = self._raven_current_subject_index()
            if subject_index_after != subject_index_before:
                logger.warning(
                    "Discarding stale Raven answer because subject changed {} -> {} during reasoning",
                    subject_index_before,
                    subject_index_after,
                )
                self.subject_finished = False
                subject = self._get_subject_from_task()
                continue
            self._apply_action(action)
            if self.subject_finished:
                logger.info("Raven subject finished by agent action.")
                break
            if self.sleep_between_steps > 0:
                time.sleep(self.sleep_between_steps)
            subject = self._get_subject_from_task()

        evaluation = self._evaluate_subject()
        record_confirmed_raven_experience(self, evaluation)

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
        if task_type == "raven" and not self._raven_text_client_initialized:
            self.raven_text_client = build_raven_text_client_from_env()
            self._raven_text_client_initialized = True
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
        is_post_turn_full_capture = bool(
            normalize_task_type(subject) == "tidyroom"
            and self._last_executed_action_name == "turn_in_degree"
            and kwargs.get("include_images", True)
        )
        if not is_post_turn_full_capture:
            return super()._acquire_camera_perception(subject, **kwargs)

        self._tidyroom_post_turn_diagnostic_sequence += 1
        sequence = self._tidyroom_post_turn_diagnostic_sequence
        self._tidyroom_post_turn_perception_blocked = False
        started_at = time.monotonic()

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
                save_label=f"postturn_{sequence:03d}_attempt_{attempt:02d}",
            )
            result = super()._acquire_camera_perception(subject, **capture_kwargs)
            capture_finished_at = time.monotonic()
            diagnostics = dict(
                getattr(self.semantic_mapper, "last_perception_diagnostics", {}) or {}
            )
            consistent, reason, gap, threshold = self._tidyroom_perception_is_consistent(
                diagnostics
            )
            logger.info(
                "Tidy-room post-turn perception sequence={} attempt={}/{}: "
                "capture_start={:.3f}s capture_end={:.3f}s capture_duration={:.3f}s "
                "visible={} aligned={} segmentation_regions={} gap={} threshold={} "
                "mapped_pixels={}/{} mapped_pixel_coverage={:.4f} "
                "coverage_threshold={:.2f} status={} reason={} "
                "image_label=postturn_{:03d}_attempt_{:02d}",
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
                sequence,
                attempt,
            )
            if consistent:
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
        """首帧/常规扫描只读结构化状态，约定的第二帧直接做完整采集。"""
        task_type = normalize_task_type(subject)
        if task_type == "raven":
            # 瑞文题图在 task_data 中，不需要采集 3D 场景相机。
            return True
        if task_type != "tidyroom":
            return False
        if self._last_executed_action_name == "turn_in_degree":
            # 转向后的首份数据必须同时读取 raw segmentation 与 visible_objects，
            # 才能发现可见对象缓存只返回少量 ID 的不同步问题。
            return False
        return not bool(
            self._task_strategy is not None
            and self._task_strategy.should_force_second_frame_vlm()
        )

    def _should_refresh_lightweight_scene(self, subject: Any) -> bool:
        """完整建图后只做目标级刷新，避免每一步重新枚举整幅视野。"""
        task_type = normalize_task_type(subject)
        if task_type == "raven":
            return False
        if task_type != "tidyroom" or self._task_strategy is None:
            return True
        return self._task_strategy.needs_scene_perception()

    def _lightweight_empty_scene_retry_delay(self, subject: Any) -> float:
        """轻量首帧不重试；转向后的等待与重试由完整感知入口负责。"""
        del subject
        return 0.0

    def _lightweight_initial_perception_delay(self, subject: Any) -> float:
        """首帧立即读取一次结构化结果，不再等待或执行盲转预热。"""
        del subject
        return 0.0

    def _should_force_vlm_after_empty_lightweight_perception(self, subject: Any) -> bool:
        """空结构化结果不原地重试也不立即调用 VLM，交给本地扫描器转向。"""
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
            return {
                "action": "turn_in_degree",
                "parameters": {"degree": 45},
                "output": 0,
                "think": (
                    "转向后的分割区域与可见对象列表连续不同步；已丢弃异常观察，"
                    "不调用 VLM、不抓取目标，安全转向 45 度后重新感知。"
                ),
            }
        if self._task_strategy is None or self._task_context is None:
            return None
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
