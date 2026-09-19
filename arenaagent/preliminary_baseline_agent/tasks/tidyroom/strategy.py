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
    # 当前方案锁定出生朝向：缺目标或目的地时也不原地转向。
    _MAX_LOCAL_SEARCH_TURNS = 0
    _LOCAL_SEARCH_DEGREES = 45
    # 目的地在这局房间里根本不存在时（模型把物品认成"鞋子"但屋里没有鞋架），
    # 转满几轮就该放弃这类目标，否则会把整道题的时间全耗在转圈上。
    _MAX_MISSING_DESTINATION_ROUNDS = 2
    # 新版 912 由客户端把坐标放置回退到 put_down_sth(force_locate=True)；
    # 四类目的地都不会改变人物所在位置，调度器据此计算拾取顺序。
    _DIRECT_FORCE_PLACE_DESTINATIONS: frozenset[str] = frozenset(
        {"sofa", "dining_table", "trash_bin", "shoe_storage"}
    )

    def reset(self, subject: dict[str, Any]) -> None:
        super().reset(subject)
        self.world = TidyRoomWorldModel(subject)
        # 保留这三个公开属性，兼容已有实验代码和测试。
        self.declared_targets = self.world.declared_targets
        self.targets = self.world.targets
        self.scene_anchors = self.world.scene_anchors
        self.planner = TidyRoomPlanner()
        # 初始正向 120° 视野已经覆盖本题主要物品和目的地；不再为了盘点转向。
        self.scanner = RoomScanner(turns_required=0, turn_degrees=90)
        self.scheduler = TargetScheduler()
        self.verifier = TidyRoomVerifier()
        self.recovery = RecoveryPolicy(vlm_threshold=3)
        self.vlm_policy = VLMEscalationPolicy()
        self.held_raw_id: str | None = None
        self.active_plan: dict[str, Any] | None = None
        self.awaiting_pick_raw_id: str | None = None
        self.local_search_turns = 0
        self.local_search_reason: str | None = None
        self.local_search_signature: tuple[Any, ...] | None = None
        self._missing_destination_rounds: dict[str, int] = {}
        self.successful_takes = 0
        self.successful_puts = 0
        self.failed_actions = 0
        self.pickup_verification_failures = 0
        self.local_action_count = 0
        self.vlm_call_count = 0
        self.vlm_reason: str | None = None
        # 保留字段仅为兼容旧的状态输出；单帧流程永远不再启动第二视角诊断。
        self._second_frame_vlm_pending = False
        self._survey_done = False

    def survey_is_due(self) -> bool:
        """初始正向分割稳定后，做一次单图语义盘点。"""
        return bool(
            self.scanner.complete
            and not self._survey_done
        )

    def note_survey_attempted(self) -> None:
        self._survey_done = True

    def observe(self, context: TaskContext) -> None:
        super().observe(context)
        self.world.observe(context)
        self._reconcile_pick_state(self._object_in_hand_raw_id(context.object_in_hand))
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
        """只刷新当前动作相关的目标，家具几何沿用扫描阶段缓存。"""
        raw_ids: list[str] = []
        if self.awaiting_pick_raw_id is not None:
            raw_ids.append(self.awaiting_pick_raw_id)
        if self.held_raw_id is not None:
            raw_ids.append(self.held_raw_id)
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
        """单帧流程不会再为第二视角强制调用 VLM。"""
        return False

    def next_local_action(self, context: TaskContext) -> dict[str, Any] | None:
        del context
        # 已经开始的搬运链拥有最高优先级。不能因为房间里还有另一个未观察
        # 目标，就在“抓起 -> 靠近 -> 放置 -> 校验”中间转向或调用 VLM。
        active_action = self._next_active_action()
        if active_action is not None:
            self.vlm_reason = None
            return active_action

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

        # 初始扫描器在单帧模式中已完成，因此这里不会产生启动转向。
        # 保留通用分支，仅为兼容外部实验显式换入非零扫描器的情况。
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
        # 调度器只会返回“已有目的地 anchor”的目标。原先几何候选提升写在
        # _ensure_plan 中，导致没有 anchor 时调度器先返回 None，_ensure_plan
        # 永远进不去，几何回退成为死路径。餐桌可由“大桌面 + 邻近 chair”
        # 稳定识别，因此必须在调度之前先提升候选。
        for candidate_raw_id in self.world.pending_raw_ids():
            destination_type = self.world.destination_type_for(
                self.targets[candidate_raw_id]
            )
            self._promote_safe_geometric_destination(destination_type)
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
        # 模型顺带标注的其余物品与家具也一并收下，避免为每一件再往返一次。
        # 它有时写在 parameters 里，有时写在动作顶层，两处都收。
        self.world.apply_scene_annotations(
            parameters.get("scene_annotations") or action.get("scene_annotations"),
            context,
        )
        # 先登记 scene_annotations 中独立确认的家具，再处理当前物品引用的
        # destination_object_id；这样单个物品条目不能凭空创造目的地。
        self.world.apply_semantic_hints(parameters, context)
        requested_id = str(parameters.get("object_id") or "")
        self.recovery.mark_vlm_consulted(self._active_raw_id())
        self.vlm_reason = None
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
            degree = int((action.get("parameters") or {}).get("degree", 0))
            if not self.scanner.complete and degree == self.scanner.turn_degrees:
                self.scanner.after_action(result)
            elif (
                self.local_search_reason is not None
                and degree == self._LOCAL_SEARCH_DEGREES
            ):
                # 搜索转向无论成功与否都消耗一次预算，防止动作接口失败时
                # 在同一个方向永久循环。
                self.local_search_turns += 1
            return
        if name == "move_and_take_object":
            self._after_take(action, result, context)
            return
        if name in {
            "move_and_put_down",
            "put_down_to_location",
            "put_down_sth_to_location",
        }:
            self._after_put(result)

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
            "successful_puts": self.successful_puts,
            "failed_actions": self.failed_actions,
            "pickup_verification_failures": self.pickup_verification_failures,
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
        plan = self.planner.build_plan(
            raw_id=raw_id,
            target=target,
            destination_type=destination_type,
            anchors=self.scene_anchors,
            slot_index=self._used_destination_slots(destination_type),
        )
        if plan is None and self.world.promote_geometric_candidate(destination_type) is not None:
            # 这一类目的地没有可信标注：模型没标出来，或者标出来的家具已经被
            # 落点高度证伪。直接按几何挑一件，用下一次落点证伪比再问一次模型
            # 快得多——模型每轮要 20 秒，而且经常把同一件错家具再标一遍。
            plan = self.planner.build_plan(
                raw_id=raw_id,
                target=target,
                destination_type=destination_type,
                anchors=self.scene_anchors,
                slot_index=self._used_destination_slots(destination_type),
            )
        self.active_plan = plan
        if self.active_plan is not None:
            # 目标被暂时延后再选中时，继续沿用累计失败序号，避免重建计划后
            # 又从同一个接近点和同一个放置槽位开始。
            self.active_plan["attempt"] = self.recovery.total_failures.get(raw_id, 0)
            self.planner.refresh_plan(self.active_plan, self.scene_anchors)
            logger.info(
                "Tidy-room local plan target={} destination={} anchor={} put={} support_z={}",
                target.get("object_id"),
                destination_type,
                self.active_plan.get("destination_object_id"),
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
                error = str(result.get("error") or "") if isinstance(result, dict) else ""
                if "not pickup" in error.lower():
                    # 新版服务端用该错误明确表示这是环境装饰/建筑，不是可拾取
                    # 刚体。继续重试或再问视觉模型都不会改变服务端属性。
                    self.targets[raw_id]["status"] = "blocked"
                    self.targets[raw_id]["blocked_reason"] = "server_not_pickup"
                    logger.warning(
                        "Tidy-room blocks non-pickup object={} after authoritative server rejection",
                        requested_id,
                    )
                    self.active_plan = None
                else:
                    self.targets[raw_id]["status"] = "pending"
                    self.recovery.record_failure(raw_id, "pickup_action_failed", self.active_plan)
            self.awaiting_pick_raw_id = None
            return
        if raw_id is not None:
            self._ensure_plan(raw_id)
            self.scheduler.mark_at_target(self.targets[raw_id])
            self.targets[raw_id]["status"] = "pickup_check"
            self.awaiting_pick_raw_id = raw_id

    def _after_put(self, result: Any) -> None:
        if self.active_plan is None:
            return
        raw_id = str(self.active_plan["target_raw_id"])
        if not action_succeeded(result):
            self.failed_actions += 1
            error = str(result.get("error") or "") if isinstance(result, dict) else ""
            failure_reason = "placement_action_failed"
            self.recovery.record_failure(raw_id, failure_reason, self.active_plan, {"error": error})
            self.planner.refresh_plan(self.active_plan, self.scene_anchors)
            return
        if self.held_raw_id is None:
            return
        # put_down_sth_to_location 是把物品强制放到指定坐标：动作返回 success
        # 就说明它已经在该落点上，不再回读 AABB 做几何校验。省下的这一步在
        # 400 秒里很值钱，也避免读到放置前的旧坐标而把放好的物品反复重放。
        self.awaiting_pick_raw_id = None
        self.held_raw_id = None
        self._finish_placement(
            raw_id,
            log="Tidy-room placement done target={} destination={} (coordinate force place)",
        )

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
        elif self.held_raw_id is not None:
            lost_raw_id = self.held_raw_id
            self.held_raw_id = None
            self.targets[lost_raw_id]["status"] = "pending"
            self.recovery.record_failure(lost_raw_id, "object_lost_before_placement", self.active_plan)

    def _finish_placement(self, raw_id: str, log: str) -> None:
        """收尾一次放置：目标标记完成、计数、清掉活动计划。"""
        destination_type = str((self.active_plan or {}).get("destination_type") or "")
        self.targets[raw_id]["status"] = "done"
        self.targets[raw_id]["verified_destination"] = destination_type
        self.successful_puts += 1
        self.recovery.record_success(raw_id)
        logger.info(log, self.targets[raw_id].get("object_id"), destination_type)
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
        # 仍有可靠的纯几何候选时不允许把整类真实目标直接 blocked。尤其是
        # 初始模型把茶几/扶手椅当餐桌时，首帧缓存中的真餐桌仍可由邻近椅子
        # 找回，不能因为两轮 VLM 都标错就提前提交 done=0。
        if self._promote_safe_geometric_destination(destination_type):
            self._missing_destination_rounds[destination_type] = 0
            return
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

    def _promote_safe_geometric_destination(self, destination_type: str | None) -> bool:
        """为具有强几何证据的目的地补建 anchor；目前仅自动恢复餐桌。"""
        if not destination_type:
            return False
        if self.world.has_destination(destination_type):
            return True
        if destination_type != "dining_table":
            return False
        return self.world.promote_geometric_candidate(destination_type) is not None

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
        # 912 新版的家具坐标先验已提供精确落点；统一使用 force_locate，人物
        # 不需要先走到家具旁边，也不会触发旧版容器接口或导航回退。
        return {
            "action": "put_down_sth_to_location",
            "parameters": {
                "target_location": deepcopy(plan["put_target_location"]),
                "which_hand": 0,
                "auto_rotate": True,
                "force_locate": True,
            },
            "output": 0,
            "think": f"{reason} 按规划落点直接放置到 {destination_type}。",
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
            "think": "当前所有可执行目标均已处理或被服务端判定不可处理，提交任务。",
        }
