from __future__ import annotations

from copy import deepcopy
from typing import Any

from loguru import logger

from arenaagent.preliminary_baseline_agent.task_runtime import TaskContext
from arenaagent.preliminary_baseline_agent.tasks.base import TaskStrategy, action_succeeded
from arenaagent.preliminary_baseline_agent.tasks.tidyroom.planner import TidyRoomPlanner
from arenaagent.preliminary_baseline_agent.tasks.tidyroom.recovery import RecoveryPolicy
from arenaagent.preliminary_baseline_agent.tasks.tidyroom.scanner import RoomScanner
from arenaagent.preliminary_baseline_agent.tasks.tidyroom.scheduler import TargetScheduler
from arenaagent.preliminary_baseline_agent.tasks.tidyroom.verifier import TidyRoomVerifier
from arenaagent.preliminary_baseline_agent.tasks.tidyroom.vlm_policy import VLMEscalationPolicy
from arenaagent.preliminary_baseline_agent.tasks.tidyroom.world_model import TidyRoomWorldModel


class TidyRoomStrategy(TaskStrategy):
    """本地结构化数据主导、视觉模型按需介入的整理房间控制器。"""

    task_type = "tidyroom"
    history_message_limit = 0
    _MAX_VERIFICATION_WAITS = 1
    _FAILURES_BEFORE_DEFER = 2
    # 完成一圈机会式扫描后，若仍有目标缺失，只允许半圈补充搜索。
    # 角色移动后的观察会持续并入世界模型，不必再次原地转完整一圈。
    _MAX_LOCAL_SEARCH_TURNS = 4
    _LOCAL_SEARCH_DEGREES = 45
    # 目的地在这局房间里根本不存在时（模型把物品认成"鞋子"但屋里没有鞋架），
    # 转满几轮就该放弃这类目标，否则会把整道题的时间全耗在转圈上。
    _MAX_MISSING_DESTINATION_ROUNDS = 2
    _TRASH_SETTLE_OBSERVATIONS = 1
    _MAX_OFFICIAL_CONTAINER_ATTEMPTS = 2
    _MAX_PLACEMENT_VERIFICATION_FAILURES = 3
    _CONTAINER_NUDGE_DISTANCE = 15.0
    # 训练阶段验证：这些承载面使用相同的已验证落点，但跳过人物先走到
    # 家具旁的动作。若动作或几何校验失败，会自动恢复传统靠近后放置。
    _DIRECT_FORCE_PLACE_DESTINATIONS = frozenset({"dining_table"})

    def reset(self, subject: dict[str, Any]) -> None:
        super().reset(subject)
        self.world = TidyRoomWorldModel(subject)
        # 保留这三个公开属性，兼容已有实验代码和测试。
        self.declared_targets = self.world.declared_targets
        self.targets = self.world.targets
        self.scene_anchors = self.world.scene_anchors
        self.planner = TidyRoomPlanner()
        self.scanner = RoomScanner()
        self.scheduler = TargetScheduler()
        self.verifier = TidyRoomVerifier()
        self.recovery = RecoveryPolicy(vlm_threshold=3)
        self.vlm_policy = VLMEscalationPolicy()
        self.held_raw_id: str | None = None
        self.active_plan: dict[str, Any] | None = None
        self.awaiting_pick_raw_id: str | None = None
        self.awaiting_verification_raw_id: str | None = None
        self.verification_waits = 0
        self.placement_settle_waits = 0
        self.local_search_turns = 0
        self.local_search_reason: str | None = None
        self.local_search_signature: tuple[Any, ...] | None = None
        self._missing_destination_rounds: dict[str, int] = {}
        self.successful_takes = 0
        self.successful_puts = 0
        self.failed_actions = 0
        self.pickup_verification_failures = 0
        self.placement_verification_failures = 0
        self.local_action_count = 0
        self.vlm_call_count = 0
        self.vlm_reason: str | None = None
        self._second_frame_vlm_pending = False
        self._survey_done = False

    def survey_is_due(self) -> bool:
        """扫完一圈、手里还没有任何目标时，做一次全屋多图标注。

        912 不下发目标清单，逐帧问模型会得到互相矛盾的标签；这里在勘测结束时
        一次性问全，之后执行阶段不再需要模型。
        """
        return bool(
            not self.world.targets_declared_by_task
            and self.scanner.complete
            and not self.world.targets
            and not self._survey_done
        )

    def note_survey_attempted(self) -> None:
        self._survey_done = True

    def observe(self, context: TaskContext) -> None:
        super().observe(context)
        self.world.observe(context)
        # 规划器保存跨视角的物体和地板信息，角色转身后数据不会丢失。
        self.planner.observe(list(self.world.scene_objects.values()))
        self._reconcile_pick_state(self._object_in_hand_raw_id(context.object_in_hand))
        self._verify_pending_placement(context)
        if not self.scanner.complete and self._scan_requirements_met():
            self.scanner.finish_early("all_targets_and_required_destinations_mapped")
            logger.info(
                "Tidy-room scan stopped early: mapped all {} targets and required destinations {}",
                len(self.targets),
                sorted(self.world.required_destination_types()),
            )

    def tracked_raw_ids(self) -> tuple[str, ...]:
        raw_ids = list(self.world.tracked_raw_ids())
        if self.active_plan is not None:
            raw_ids.extend(
                [
                    str(self.active_plan.get("target_raw_id") or ""),
                    str(self.active_plan.get("destination_raw_id") or ""),
                ]
            )
        return tuple(raw_id for raw_id in dict.fromkeys(raw_ids) if raw_id)

    def refresh_raw_ids(self) -> tuple[str, ...]:
        """只刷新会影响当前动作校验的目标，家具几何沿用扫描阶段缓存。"""
        raw_ids: list[str] = []
        if self.awaiting_pick_raw_id is not None:
            raw_ids.append(self.awaiting_pick_raw_id)
        if self.held_raw_id is not None:
            raw_ids.append(self.held_raw_id)
        if self.awaiting_verification_raw_id is not None:
            raw_ids.append(self.awaiting_verification_raw_id)
        if self.active_plan is not None:
            raw_ids.extend(
                [
                    str(self.active_plan.get("target_raw_id") or ""),
                    str(self.active_plan.get("destination_raw_id") or ""),
                ]
            )
        return tuple(raw_id for raw_id in dict.fromkeys(raw_ids) if raw_id)

    def needs_scene_perception(self) -> bool:
        """目标及所需家具全部入图后，后续只查询动作相关对象。"""
        return not self._scan_requirements_met()

    def note_forced_vlm(self, reason: str) -> None:
        """接收公共运行时发起的完整视觉诊断。"""
        self._request_vlm(reason)

    def should_force_second_frame_vlm(self) -> bool:
        """首帧未形成组合并转向后，第二帧必须完整采集并咨询一次 VLM。

        前提是任务系统已经下发了目标清单——模型可以拿清单去比对自己看到的
        画面。912 不再下发清单，此时扫描才刚开始，模型没有可对照的物品，问
        它只会得到"继续转"这类无用回答（实测单次 70~130 秒）。这种情况留给
        扫描覆盖全屋之后的 local_plan_unavailable 一次性补充语义。
        """
        return bool(
            self._second_frame_vlm_pending
            and self.step_index >= 2
            and self.vlm_call_count == 0
            # 必须是任务系统下发的清单；勘测发现的目标不算，否则全屋标注之后
            # 又会被这个条件触发一次多余的诊断。
            and self.world.targets_declared_by_task
        )

    def next_local_action(self, context: TaskContext) -> dict[str, Any] | None:
        del context
        # 已经开始的搬运链拥有最高优先级。不能因为房间里还有另一个未观察
        # 目标，就在“抓起 -> 靠近 -> 放置 -> 校验”中间转向或调用 VLM。
        active_action = self._next_active_action()
        if active_action is not None:
            self.vlm_reason = None
            return active_action

        # 首帧没有可执行组合时已经转过 45 度。第二帧跳过本地短路，强制把
        # 当前同一帧的结构化信息、RGB 和分割图交给 VLM 一次。
        if self.should_force_second_frame_vlm():
            self._second_frame_vlm_pending = False
            return self._request_vlm("second_scene_diagnostic")

        active_raw_id = self._active_raw_id()
        self.vlm_reason = self.vlm_policy.reason(
            self.world,
            self.scanner,
            self.recovery,
            active_raw_id,
        )
        if self.vlm_reason == "local_recovery_exhausted":
            return self._request_vlm(self.vlm_reason)

        # 优先处理已经由结构化感知识别并且目的地可规划的物体。全局仍有
        # 未观察目标，不应阻塞眼前可确定完成的工作。
        pick_action = self._next_pick_action()
        if pick_action is not None:
            self.vlm_reason = None
            return pick_action

        # 首帧只使用一次结构化感知。无论结果为空，还是没有形成“待整理目标
        # + 对应目的地”的可执行组合，都立即转向 45 度；VLM 固定延后到转向
        # 后的第二帧，避免首帧分割缓存尚未就绪时浪费一次视觉调用。
        if self.step_index == 1:
            self._second_frame_vlm_pending = True
            scan_action = self.scanner.next_action()
            if scan_action is None:
                scan_action = self._turn_action(
                    self.scanner.turn_degrees,
                    "首帧没有形成可执行组合，转向 45 度后进入第二帧完整视觉诊断。",
                )
            return self._count_local(scan_action)

        # 扫描不再是开始任务前必须完成的阻塞阶段。当前视野没有形成
        # “目标 + 对应家具”的可执行计划时，才转 45 度补充一次视野；
        # 每次移动、抓取和放置后的新观察仍会持续并入同一个世界模型。
        scan_action = self.scanner.next_action()
        if scan_action is not None:
            self.vlm_reason = None
            return self._count_local(scan_action)

        self.vlm_reason = self.vlm_policy.reason(
            self.world,
            self.scanner,
            self.recovery,
            active_raw_id,
        )
        if self._is_local_search_reason(self.vlm_reason):
            search_action = self._next_local_search_action(self.vlm_reason)
            if search_action is not None:
                return self._count_local(search_action)
            # 刚放弃了一批够不着的目标，别再为它们问一次模型。
            if self._all_targets_settled():
                return self._count_local(self._submit_action())
        return self._request_vlm(self.vlm_reason or "local_plan_unavailable")

    def _all_targets_settled(self) -> bool:
        return bool(
            self.targets
            and (self.world.targets_declared_by_task or self.scanner.complete)
            and all(record.get("status") in {"done", "blocked"} for record in self.targets.values())
        )

    def _next_active_action(self) -> dict[str, Any] | None:
        """继续已开始的确定性搬运链，不被其他缺失目标打断。"""

        # 任务系统下发了完整清单时，清单做完即可提交。清单是模型逐帧发现的
        # 时候（912）不行：扫描没走完就可能还有一侧的房间压根没看过。
        if self._all_targets_settled():
            return self._count_local(self._submit_action())

        if self.awaiting_verification_raw_id is not None:
            return self._count_local(
                {
                    "action": "turn_in_degree",
                    "parameters": {"degree": 0},
                    "output": 0,
                    "think": "等待物理系统稳定并刷新目标物全局 AABB。",
                }
            )

        if self.held_raw_id is not None:
            if self.recovery.needs_vlm(self.held_raw_id):
                # 已持物也不能无限绕过失败阈值；把控制权交回升级策略一次。
                return None
            plan = self._ensure_plan(self.held_raw_id)
            if plan is not None:
                return self._count_local(self._put_action(plan, "结构化手持校验通过，执行本地规划的放置动作。"))

        return None

    def _next_pick_action(self) -> dict[str, Any] | None:
        """选择一个当前结构化数据足以处理的目标。"""
        raw_id = self._retry_or_schedule_target()
        if raw_id is None:
            return None
        plan = self._ensure_plan(raw_id)
        if plan is None:
            return None
        object_id = self.targets[raw_id].get("object_id")
        if object_id is None:
            return None
        return self._count_local(
            self._take_action(str(object_id), "本地调度器已选择目标，并锁定目的地与重试计划。")
        )

    def validate_action(self, action: dict[str, Any], context: TaskContext | None) -> dict[str, Any]:
        """VLM 只会走到这里；本方法继续用确定性约束保护最终动作。"""
        name = str(action.get("action") or "").lower()
        parameters = action.get("parameters") or {}
        consulted_reason = self.vlm_reason
        self.world.apply_semantic_hints(parameters, context)
        # 模型顺带标注的其余物品与家具也一并收下，避免为每一件再往返一次。
        # 它有时写在 parameters 里，有时写在动作顶层，两处都收。
        self.world.apply_scene_annotations(
            parameters.get("scene_annotations") or action.get("scene_annotations"),
            context,
        )
        requested_id = str(parameters.get("object_id") or "")
        self.recovery.mark_vlm_consulted(self._active_raw_id())
        self.vlm_reason = None
        if name == "turn_in_degree" and consulted_reason == "second_scene_diagnostic":
            # 第二帧 VLM 只决定“是否需要换方向”，扫描角度仍由本地扫描器统一
            # 管理。否则模型返回 15° 等任意角度时，after_action 会把它误计为
            # 一次完整的 45° 扫描，最终形成视野缺口。
            original_degree = parameters.get("degree")
            if original_degree != self.scanner.turn_degrees:
                action = deepcopy(action)
                parameters = dict(parameters)
                parameters["degree"] = self.scanner.turn_degrees
                action["parameters"] = parameters
                action["think"] = (
                    f"{str(action.get('think') or '').strip()} "
                    f"第二帧搜索转向由本地扫描器规范为 {self.scanner.turn_degrees}°"
                    f"（VLM 原始角度：{original_degree}°）。"
                ).strip()
        if name == "turn_in_degree" and self._is_local_search_reason(consulted_reason):
            # VLM 在结构化搜索耗尽后通常只会建议换个方向观察。执行这一次
            # 建议后重新给本地搜索一轮预算，避免下一帧再次请求 VLM。
            self._reset_local_search()

        if self.targets and self.world.unfinished_count() == 0:
            return self._submit_action()
        if self.held_raw_id is not None:
            plan = self._ensure_plan(self.held_raw_id)
            if plan is not None:
                return self._put_action(plan, "VLM 已完成异常判断；实际坐标由本地规划器校正。")

        pending_target_ids = {
            str(record["object_id"]): raw_id
            for raw_id, record in self.targets.items()
            if record.get("object_id") is not None and record.get("status") == "pending"
        }
        chosen_raw_id = pending_target_ids.get(requested_id)
        navigation_actions = {"move_to_object", "move_to_location", "move_forward", "move_backward"}
        if chosen_raw_id is None and (name in navigation_actions or name == "move_and_take_object"):
            chosen_raw_id = self._raw_id_for_mapped_id(self._first_pending_visible_id() or "", context)
        if name in {"finish_task", "submit_answer"} and self.world.unfinished_count() > 0:
            chosen_raw_id = self._raw_id_for_mapped_id(self._first_pending_visible_id() or "", context)

        if chosen_raw_id is not None:
            plan = self._ensure_plan(chosen_raw_id)
            if plan is not None:
                return self._take_action(
                    str(self.targets[chosen_raw_id]["object_id"]),
                    "VLM 选择了异常恢复目标；目的地和坐标仍由本地规划器锁定。",
                )
        if name in {"finish_task", "submit_answer"} and self.world.unfinished_count() > 0:
            return self._turn_action(60, "仍有未完成目标，继续观察缺失的目标或目的地。")
        return action

    def after_action(self, action: dict[str, Any], result: Any, context: TaskContext | None) -> None:
        name = str(action.get("action") or "").lower()
        if name == "turn_in_degree":
            if not self.scanner.complete:
                self.scanner.after_action(result)
            elif (
                self.local_search_reason is not None
                and int((action.get("parameters") or {}).get("degree", 0)) == self._LOCAL_SEARCH_DEGREES
            ):
                # 搜索转向无论成功与否都消耗一次预算，防止动作接口失败时
                # 在同一个方向永久循环。
                self.local_search_turns += 1
            return
        if name == "move_and_take_object":
            self._after_take(action, result, context)
            return
        if name in {"move_to_location", "move_to_object", "move_forward"} and self._after_placement_approach(result):
            return
        if name in {
            "move_and_put_down",
            "put_down_to_location",
            "put_down_sth_to_location",
            "move_and_put_down_object_in_container",
        }:
            self._after_put(action, result)

    def state_for_prompt(self) -> dict[str, Any]:
        plan_for_prompt = None
        if self.active_plan is not None:
            plan_for_prompt = {
                key: deepcopy(value)
                for key, value in self.active_plan.items()
                if key not in {"target_raw_id", "destination_raw_id", "target_info", "destination_info"}
            }
        return {
            "step_index": self.step_index,
            "target_count": len(self.declared_targets),
            "unfinished_count": self.world.unfinished_count(),
            "held_target_object_id": self.targets.get(self.held_raw_id, {}).get("object_id"),
            "successful_takes": self.successful_takes,
            "successful_verified_puts": self.successful_puts,
            "failed_actions": self.failed_actions,
            "pickup_verification_failures": self.pickup_verification_failures,
            "placement_verification_failures": self.placement_verification_failures,
            "local_action_count": self.local_action_count,
            "vlm_call_count": self.vlm_call_count,
            "vlm_reason": self.vlm_reason,
            "second_frame_vlm_pending": self._second_frame_vlm_pending,
            "local_search": {
                "reason": self.local_search_reason,
                "turns_completed": self.local_search_turns,
                "turns_limit": self._MAX_LOCAL_SEARCH_TURNS,
            },
            "scanner": self.scanner.state(),
            "recovery": self.recovery.state(),
            "active_plan": plan_for_prompt,
            "targets": [self._target_for_prompt(raw_id) for raw_id in self.declared_targets],
            "scene_anchors": [
                {key: deepcopy(value) for key, value in anchor.items() if key != "raw_id"}
                for anchor in self.scene_anchors.values()
            ],
        }

    def _ensure_plan(self, raw_id: str) -> dict[str, Any] | None:
        if self.active_plan is not None and self.active_plan.get("target_raw_id") == raw_id:
            self.planner.refresh_plan(self.active_plan, self.scene_anchors)
            return self.active_plan
        target = self.targets.get(raw_id)
        if target is None:
            return None
        destination_type = self.world.destination_type_for(target)
        if destination_type is None:
            return None
        self.active_plan = self.planner.build_plan(
            raw_id=raw_id,
            target=target,
            destination_type=destination_type,
            anchors=self.scene_anchors,
            slot_index=self._used_destination_slots(destination_type),
        )
        if self.active_plan is not None:
            # 目标被暂时延后再选中时，继续沿用累计失败序号，避免重建计划后
            # 又从同一个接近点和同一个放置槽位开始。
            self.active_plan["attempt"] = self.recovery.total_failures.get(raw_id, 0)
            self.active_plan["direct_force_place_disabled"] = bool(
                target.get("direct_force_place_disabled", False)
            )
            self.planner.refresh_plan(self.active_plan, self.scene_anchors)
            logger.info(
                "Tidy-room local plan target={} destination={} anchor={} move={} put={} support_z={}",
                target.get("object_id"),
                destination_type,
                self.active_plan.get("destination_object_id"),
                self.active_plan.get("move_target_location"),
                self.active_plan.get("put_target_location"),
                self.active_plan.get("support_z"),
            )
        return self.active_plan

    def _after_take(self, action: dict[str, Any], result: Any, context: TaskContext | None) -> None:
        requested_id = str((action.get("parameters") or {}).get("object_id") or "")
        raw_id = self._raw_id_for_mapped_id(requested_id, context)
        if not action_succeeded(result):
            self.failed_actions += 1
            if raw_id in self.targets:
                self.targets[raw_id]["status"] = "pending"
                self.recovery.record_failure(raw_id, "pickup_action_failed", self.active_plan)
            self.awaiting_pick_raw_id = None
            return
        if raw_id is not None:
            self._ensure_plan(raw_id)
            self.scheduler.mark_at_target(self.targets[raw_id])
            self.targets[raw_id]["status"] = "pickup_check"
            self.awaiting_pick_raw_id = raw_id

    def _after_put(self, action: dict[str, Any], result: Any) -> None:
        if self.active_plan is None:
            return
        raw_id = str(self.active_plan["target_raw_id"])
        direct_force_place = bool(self.active_plan.pop("direct_force_place_in_flight", False))
        if not action_succeeded(result):
            self.failed_actions += 1
            action_name = str(action.get("action") or "")
            error = str(result.get("error") or "") if isinstance(result, dict) else ""
            failure_reason = "placement_action_failed"
            if action_name == "move_and_put_down_object_in_container":
                failure_reason = "container_not_found" if "no container found" in error.lower() else failure_reason
                attempts = int(self.active_plan.get("official_container_attempts", 0)) + 1
                self.active_plan["official_container_attempts"] = attempts
                if attempts >= self._MAX_OFFICIAL_CONTAINER_ATTEMPTS:
                    # 专用接口无法识别这个垃圾桶时立即切换精确落点，不能再
                    # 围绕垃圾桶循环移动。
                    self.active_plan["container_coordinate_fallback"] = True
                    self.active_plan["placement_approached"] = True
                    logger.warning(
                        "Tidy-room container API failed {} times for target={}; use coordinate fallback",
                        attempts,
                        self.targets[raw_id].get("object_id"),
                    )
                else:
                    self.active_plan["placement_approached"] = False
            else:
                self.active_plan["placement_approached"] = False
                if direct_force_place:
                    # 直接放置不受支持时只试一次，随后恢复“先靠近再放置”。
                    self.active_plan["direct_force_place_disabled"] = True
                    self.targets[raw_id]["direct_force_place_disabled"] = True
            self.recovery.record_failure(raw_id, failure_reason, self.active_plan, {"error": error})
            self.planner.refresh_plan(self.active_plan, self.scene_anchors)
            return
        if self.held_raw_id is None:
            return
        if not direct_force_place:
            self.scheduler.mark_at_destination(self.active_plan)
        self.targets[raw_id]["status"] = "placement_check"
        self.awaiting_verification_raw_id = raw_id
        self.awaiting_pick_raw_id = None
        self.held_raw_id = None
        self.verification_waits = 0
        self.placement_settle_waits = 0
        parameters = action.get("parameters") or {}
        self.active_plan["last_put_location"] = deepcopy(
            parameters.get("put_target_location") or parameters.get("target_location") or {}
        )
        self.active_plan["last_put_action"] = str(action.get("action") or "")
        self.active_plan["last_put_was_direct"] = direct_force_place

    def _reconcile_pick_state(self, observed_hand_raw_id: str | None) -> None:
        if self.awaiting_pick_raw_id is not None:
            expected_raw_id = self.awaiting_pick_raw_id
            self.awaiting_pick_raw_id = None
            check = self.verifier.verify_pickup(expected_raw_id, observed_hand_raw_id)
            if check["valid"]:
                self.held_raw_id = expected_raw_id
                self.targets[expected_raw_id]["status"] = "held"
                self.successful_takes += 1
                self._reset_local_search()
                return
            self.targets[expected_raw_id]["status"] = "pending"
            self.held_raw_id = None
            self.pickup_verification_failures += 1
            self.recovery.record_failure(expected_raw_id, str(check["reason"]), self.active_plan, check)
            if observed_hand_raw_id in self.targets:
                self.held_raw_id = observed_hand_raw_id
                self.targets[observed_hand_raw_id]["status"] = "held"
                self.active_plan = None
            logger.warning("Tidy-room pickup verification failed target={} check={}", expected_raw_id, check)
            return
        if observed_hand_raw_id in self.targets:
            if self.held_raw_id != observed_hand_raw_id:
                self.successful_takes += 1
            self.held_raw_id = observed_hand_raw_id
            self.targets[observed_hand_raw_id]["status"] = "held"
        elif self.held_raw_id is not None and self.awaiting_verification_raw_id is None:
            lost_raw_id = self.held_raw_id
            self.held_raw_id = None
            self.targets[lost_raw_id]["status"] = "pending"
            self.recovery.record_failure(lost_raw_id, "object_lost_before_placement", self.active_plan)

    def _verify_pending_placement(self, context: TaskContext) -> None:
        raw_id = self.awaiting_verification_raw_id
        if raw_id is None or self.active_plan is None:
            return
        if (
            self.active_plan.get("destination_type") == "trash_bin"
            and self.placement_settle_waits < self._TRASH_SETTLE_OBSERVATIONS
        ):
            # 官方容器动作会启用物理模拟。至少跨过一个本地循环再读取 AABB，
            # 避免在罐子仍处于下落过程时误判成功或失败。
            self.placement_settle_waits += 1
            return
        target_aabb = context.world_aabbs_by_raw_id.get(raw_id)
        destination_raw_id = str(self.active_plan["destination_raw_id"])
        destination_aabb = context.world_aabbs_by_raw_id.get(destination_raw_id)
        if not target_aabb and self.targets.get(raw_id, {}).get("visible"):
            target_aabb = self.targets.get(raw_id, {}).get("object_info", {}).get("world_aabb")
        if not destination_aabb:
            destination_aabb = self.active_plan.get("destination_info", {}).get("world_aabb")
        check = self.verifier.verify_placement(target_aabb, destination_aabb, self.active_plan)
        if check["reason"] == "missing_aabb":
            self.verification_waits += 1
            if self.verification_waits <= self._MAX_VERIFICATION_WAITS:
                return
        self.awaiting_verification_raw_id = None
        self.verification_waits = 0
        self.placement_settle_waits = 0
        if check["valid"]:
            self.targets[raw_id]["status"] = "done"
            self.targets[raw_id]["verified_destination"] = self.active_plan["destination_type"]
            self.successful_puts += 1
            self.recovery.record_success(raw_id)
            logger.info(
                "Tidy-room placement verified target={} destination={} geometry={}",
                self.targets[raw_id].get("object_id"),
                self.active_plan["destination_type"],
                check,
            )
            self.active_plan = None
            return
        self.targets[raw_id]["status"] = "pending"
        self.targets[raw_id]["placement_verification_failures"] = (
            int(self.targets[raw_id].get("placement_verification_failures", 0)) + 1
        )
        self.placement_verification_failures += 1
        self.recovery.record_failure(raw_id, str(check["reason"]), self.active_plan, check)
        self.active_plan["placement_approached"] = False
        if self.active_plan.get("last_put_was_direct", False):
            # 动作虽然返回成功，但官方几何状态不正确时也关闭快速路径。
            self.active_plan["direct_force_place_disabled"] = True
            self.targets[raw_id]["direct_force_place_disabled"] = True
        self.planner.refresh_plan(self.active_plan, self.scene_anchors)
        logger.warning(
            "Tidy-room placement invalid target={} destination={} action={} reason={} "
            "actual_center={} destination_aabb={} next_move={} next_put={}",
            self.targets[raw_id].get("object_id"),
            self.active_plan.get("destination_type"),
            self.active_plan.get("last_put_action"),
            check.get("reason"),
            check.get("actual_center"),
            check.get("destination_aabb"),
            self.active_plan.get("move_target_location"),
            self.active_plan.get("put_target_location"),
        )
        target_placement_failures = int(
            self.targets[raw_id].get("placement_verification_failures", 0)
        )
        if target_placement_failures >= self._MAX_PLACEMENT_VERIFICATION_FAILURES:
            # 坐标纠正仍连续失败时停止重新抓放。保留 blocked 状态让最终提交
            # 体现真实完成度，避免一个异常物体耗尽 64 步或反复调用 VLM。
            self.targets[raw_id]["status"] = "blocked"
            logger.error(
                "Tidy-room blocks target={} after {} placement verification failures",
                self.targets[raw_id].get("object_id"),
                target_placement_failures,
            )
            self.active_plan = None
        elif self.recovery.consecutive_failures.get(raw_id, 0) >= self._FAILURES_BEFORE_DEFER:
            # 已释放的物体连续两次没有落入有效区域时，先让调度器选择其他
            # 物体。失败次数仍被保留；若之后再次选择它，会换接近方向和槽位。
            # 这样一个滚落物体不会耗尽整场比赛时间。
            logger.warning(
                "Tidy-room defers repeatedly failed target={} failures={}",
                self.targets[raw_id].get("object_id"),
                self.recovery.consecutive_failures.get(raw_id, 0),
            )
            self.active_plan = None

    def _retry_or_schedule_target(self) -> str | None:
        if self.active_plan is not None:
            raw_id = str(self.active_plan.get("target_raw_id") or "")
            if raw_id in self.targets and self.targets[raw_id]["status"] == "pending":
                return raw_id
        return self.scheduler.choose(
            self.world,
            self.planner,
            self.recovery.total_failures,
            self._DIRECT_FORCE_PLACE_DESTINATIONS,
        )

    def _scan_requirements_met(self) -> bool:
        """目标及它们实际需要的家具均已进入持续世界模型。"""
        if not self.targets:
            return False
        if not self.world.targets_declared_by_task:
            # 912 不下发清单，模型标注的只是它那一帧看到的部分。提前结束扫描
            # 会让房间另一侧的物品永远进不了世界模型，必须扫完整圈。
            return False
        if any(record.get("object_id") is None for record in self.targets.values()):
            return False
        if any(
            self.world.destination_type_for(record) is None
            for record in self.targets.values()
            if record.get("status") != "done"
        ):
            return False
        required = self.world.required_destination_types()
        return bool(required) and all(self.world.has_destination(item) for item in required)

    def _active_raw_id(self) -> str | None:
        if self.held_raw_id is not None:
            return self.held_raw_id
        if self.active_plan is not None:
            return str(self.active_plan.get("target_raw_id") or "") or None
        return None

    def _target_for_prompt(self, raw_id: str) -> dict[str, Any]:
        record = self.targets[raw_id]
        return {
            "object_id": record.get("object_id"),
            "category": record["category"],
            "semantic_label": record.get("semantic_label"),
            "destination_type": self.world.destination_type_for(record),
            "recommended_destination": self.world.destination_type_for(record),
            "status": record["status"],
            "visible": record["visible"],
            "object_info": record["object_info"],
        }

    def _raw_id_for_mapped_id(self, mapped_id: str, context: TaskContext | None) -> str | None:
        if context is not None:
            for raw_id, candidate_id in context.raw_to_mapped_id.items():
                if str(candidate_id) == mapped_id and raw_id in self.targets:
                    return raw_id
        for raw_id, record in self.targets.items():
            if str(record.get("object_id")) == mapped_id:
                return raw_id
        return None

    def _first_pending_visible_id(self) -> str | None:
        for raw_id in self.declared_targets:
            record = self.targets[raw_id]
            if record["status"] == "pending" and record["visible"] and record.get("object_id") is not None:
                return str(record["object_id"])
        return None

    def _used_destination_slots(self, destination_type: str) -> int:
        return sum(
            record.get("status") == "done" and record.get("verified_destination") == destination_type
            for record in self.targets.values()
        )

    def _count_local(self, action: dict[str, Any]) -> dict[str, Any]:
        self.local_action_count += 1
        return action

    def _next_local_search_action(self, reason: str) -> dict[str, Any] | None:
        """用廉价结构化转向代替只会让角色转身的 VLM 调用。"""
        signature = self._local_search_signature(reason)
        if signature != self.local_search_signature:
            self.local_search_signature = signature
            self.local_search_reason = reason
            # 原地发现新对象只更新搜索原因，不重置补扫预算。只有成功抓取
            # 带来真实位置变化后，_reset_local_search 才开始新一轮搜索。
        if self.local_search_turns >= self._MAX_LOCAL_SEARCH_TURNS:
            if reason.startswith("missing_destination:"):
                self._give_up_on_destination(reason)
            return None
        return self._turn_action(
            self._LOCAL_SEARCH_DEGREES,
            f"本地补充搜索 {self.local_search_turns + 1}/{self._MAX_LOCAL_SEARCH_TURNS}：{reason}。",
        )

    def _give_up_on_destination(self, reason: str) -> None:
        """目的地在这局房间里不存在时，把需要它的目标标成 blocked。

        模型会把物品认成"鞋子"，而屋里根本没有鞋架。不设上限的话就是：转满
        4 圈 → 问模型 → validate_action 重置预算 → 再转 4 圈，整道题的时间
        全耗在转圈上。转满若干轮就放弃这类目标，让调度器去做别的或者提交。
        """
        destination_type = reason.partition(":")[2]
        rounds = self._missing_destination_rounds.get(destination_type, 0) + 1
        self._missing_destination_rounds[destination_type] = rounds
        if rounds < self._MAX_MISSING_DESTINATION_ROUNDS:
            return
        blocked = 0
        for record in self.targets.values():
            if record.get("status") in {"done", "blocked"}:
                continue
            if self.world.destination_type_for(record) == destination_type:
                record["status"] = "blocked"
                blocked += 1
        if blocked:
            logger.warning(
                "Tidy-room gave up on {} target(s) needing {}: no such destination is visible in this room",
                blocked,
                destination_type,
            )

    def _local_search_signature(self, reason: str) -> tuple[Any, ...]:
        mapped_targets = sum(record.get("object_id") is not None for record in self.targets.values())
        destination_types = tuple(sorted(str(anchor.get("type") or "") for anchor in self.scene_anchors.values()))
        return reason, mapped_targets, destination_types

    @staticmethod
    def _is_local_search_reason(reason: str | None) -> bool:
        return bool(reason == "target_not_observed" or (reason or "").startswith("missing_destination:"))

    def _reset_local_search(self) -> None:
        self.local_search_turns = 0
        self.local_search_reason = None
        self.local_search_signature = None

    def _after_placement_approach(self, result: Any) -> bool:
        """记录携物靠近目的地的动作，下一步再执行可靠坐标放置。"""
        plan = self.active_plan
        if plan is None or self.held_raw_id is None:
            return False
        raw_id = str(plan["target_raw_id"])
        if action_succeeded(result):
            plan["placement_approached"] = True
        else:
            self.failed_actions += 1
            plan["placement_approached"] = False
            self.recovery.record_failure(raw_id, "placement_approach_failed", plan)
            if (
                plan.get("destination_type") == "trash_bin"
                and self.recovery.consecutive_failures.get(raw_id, 0) >= self._MAX_OFFICIAL_CONTAINER_ATTEMPTS
            ):
                plan["container_coordinate_fallback"] = True
                plan["placement_approached"] = True
            self.planner.refresh_plan(plan, self.scene_anchors)
        return True

    def _request_vlm(self, reason: str) -> None:
        """记录一次即将发生的真实 VLM 请求，并把控制权交回公共运行时。"""
        self.vlm_reason = reason
        self.vlm_call_count += 1
        logger.info("Tidy-room escalates to VLM because {}", reason)

    @staticmethod
    def _object_in_hand_raw_id(object_in_hand: Any) -> str | None:
        if isinstance(object_in_hand, (tuple, list)) and object_in_hand:
            return str(object_in_hand[0])
        if isinstance(object_in_hand, dict):
            value = object_in_hand.get("object_id")
            return str(value) if value is not None else None
        if isinstance(object_in_hand, str):
            return object_in_hand
        return None

    @staticmethod
    def _take_action(object_id: str, reason: str) -> dict[str, Any]:
        return {
            "action": "move_and_take_object",
            "parameters": {"object_id": object_id, "which_hand": 0},
            "output": 0,
            "think": reason,
        }

    def _put_action(self, plan: dict[str, Any], reason: str) -> dict[str, Any]:
        self.planner.refresh_plan(plan, self.scene_anchors)
        destination_type = str(plan.get("destination_type") or "目的地")
        if destination_type == "trash_bin" and plan.get("container_coordinate_fallback", False):
            return {
                "action": "put_down_to_location",
                "parameters": {
                    "target_location": deepcopy(plan["put_target_location"]),
                    "which_hand": 0,
                    "auto_rotate": True,
                    "force_release": True,
                    "disable_physics": False,
                    "hold_if_unreachable": False,
                    "force_locate": True,
                },
                "output": 0,
                "think": "容器接口两次未识别垃圾桶，停止绕行并回退到桶内精确落点。",
            }
        direct_force_place = (
            destination_type in self._DIRECT_FORCE_PLACE_DESTINATIONS
            and not plan.get("direct_force_place_disabled", False)
        )
        if direct_force_place:
            plan["direct_force_place_attempted"] = True
            plan["direct_force_place_in_flight"] = True
            return {
                "action": "put_down_to_location",
                "parameters": {
                    "target_location": deepcopy(plan["put_target_location"]),
                    "which_hand": 0,
                    "auto_rotate": True,
                    "force_release": True,
                    "disable_physics": True,
                    "hold_if_unreachable": False,
                    "force_locate": True,
                },
                "output": 0,
                "think": f"训练加速：保持 {destination_type} 的已验证落点，直接精确放置并在下一帧校验。",
            }
        if not plan.get("placement_approached", False):
            if destination_type == "trash_bin":
                official_attempts = int(plan.get("official_container_attempts", 0))
                if official_attempts == 0:
                    return {
                        "action": "move_to_object",
                        "parameters": {"object_id": str(plan["destination_object_id"])},
                        "output": 0,
                        "think": "让仿真导航直接靠近垃圾桶的有效交互距离。",
                    }
                return {
                    "action": "move_forward",
                    "parameters": {"distance": self._CONTAINER_NUDGE_DISTANCE},
                    "output": 0,
                    "think": "容器接口首次未识别垃圾桶，仅向前微调 15 厘米后重试。",
                }
            return {
                "action": "move_to_location",
                "parameters": {
                    "target_location": deepcopy(plan["move_target_location"]),
                    "stop_distance": 5.0,
                },
                "output": 0,
                "think": f"{reason} 先携物移动到 {destination_type} 的可达侧。",
            }
        if destination_type == "trash_bin":
            # 正常路径使用仿真提供的官方“放入容器”动作，让赛题侧建立真实
            # 的容器关系。精确坐标放置只保留给桌面和沙发。
            return {
                "action": "move_and_put_down_object_in_container",
                "parameters": {"which_hand": 0},
                "output": 0,
                "think": "已到达垃圾桶可操作侧，使用官方容器动作放入垃圾并等待物理落底。",
            }
        # 桌面和沙发仍使用本地规划的精确落点，并冻结以防圆形食物滚落。
        return {
            "action": "put_down_to_location",
            "parameters": {
                "target_location": deepcopy(plan["put_target_location"]),
                "which_hand": 0,
                "auto_rotate": True,
                "force_release": True,
                "disable_physics": True,
                "hold_if_unreachable": False,
                "force_locate": True,
            },
            "output": 0,
            "think": f"已靠近 {destination_type}，执行可验证的精确放置。",
        }

    @staticmethod
    def _turn_action(degree: int, reason: str) -> dict[str, Any]:
        return {
            "action": "turn_in_degree",
            "parameters": {"degree": degree},
            "output": 0,
            "think": reason,
        }

    @staticmethod
    def _submit_action() -> dict[str, Any]:
        return {
            "action": "submit_answer",
            "parameters": {},
            "output": 0,
            "think": "全部目标均已通过结构化放置校验，提交任务。",
        }
