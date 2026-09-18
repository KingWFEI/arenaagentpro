from __future__ import annotations

from copy import deepcopy
from typing import Any

from loguru import logger

from arenaagent.preliminary_baseline_agent.task_runtime import TaskContext
from arenaagent.preliminary_baseline_agent.tasks.tidyroom.destinations import (
    infer_anchor_type,
    infer_target_category,
    normalize_destination_type,
    normalize_semantic_label,
    recommended_destination,
)


class TidyRoomWorldModel:
    """整理房间的持续世界状态，不依赖模型对历史截图的记忆。"""

    def __init__(self, subject: dict[str, Any]) -> None:
        self.declared_targets = [str(raw_id) for raw_id in subject.get("movable_object_id") or []]
        # 912 不下发清单，目标全靠模型在感知里发现。此时清单只是"目前看到的
        # 这些"，不能当成"房间里的全部"，因此也不能据它提前结束扫描。
        self.targets_declared_by_task = bool(self.declared_targets)
        self.targets: dict[str, dict[str, Any]] = {
            raw_id: self._new_target_record(raw_id) for raw_id in self.declared_targets
        }
        self.scene_objects: dict[str, dict[str, Any]] = {}
        self.scene_anchors: dict[str, dict[str, Any]] = {}

    def observe(self, context: TaskContext) -> None:
        """合并本帧可见数据和按原始 ID 查询到的全局 AABB。"""
        self._merge_declared_targets(context.movable_objects)
        visible_by_id = {str(item.get("object_id")): item for item in context.visible_objects}
        for object_id, info in visible_by_id.items():
            self.scene_objects[object_id] = deepcopy(info)
        self._update_targets(context, visible_by_id)
        self._update_anchors(context, visible_by_id)

    def tracked_raw_ids(self) -> tuple[str, ...]:
        """持续刷新全部目标和已经发现的目的地，而非只看当前截图。"""
        raw_ids = list(self.declared_targets)
        raw_ids.extend(str(anchor["raw_id"]) for anchor in self.scene_anchors.values())
        return tuple(dict.fromkeys(raw_ids))

    def pending_raw_ids(self) -> list[str]:
        return [raw_id for raw_id in self.declared_targets if self.targets[raw_id]["status"] == "pending"]

    def unfinished_count(self) -> int:
        return sum(record["status"] != "done" for record in self.targets.values())

    def required_destination_types(self) -> set[str]:
        return {
            destination_type
            for record in self.targets.values()
            if record["status"] != "done"
            if (destination_type := self.destination_type_for(record)) is not None
        }

    @staticmethod
    def destination_type_for(record: dict[str, Any]) -> str | None:
        """优先使用经过白名单约束的显式目的地，再回退到本地语义规则。"""
        explicit = normalize_destination_type(record.get("destination_type"))
        if explicit is not None:
            return explicit
        return recommended_destination(str(record.get("semantic_label") or record.get("category") or ""))

    # 判据来自实测：地板 751×807、沙发 112×410。家具总有一条边是薄的，
    # 结构面则是两条边都很宽、而且很高（墙）。
    _MAX_FURNITURE_FOOTPRINT = 400.0
    _MAX_FURNITURE_HEIGHT = 200.0

    def _is_furniture_sized(self, object_id: str) -> bool:
        info = self.scene_objects.get(object_id) or {}
        aabb = info.get("world_aabb") or {}
        low, high = aabb.get("min") or {}, aabb.get("max") or {}
        try:
            span_x = float(high.get("x", 0.0)) - float(low.get("x", 0.0))
            span_y = float(high.get("y", 0.0)) - float(low.get("y", 0.0))
            span_z = float(high.get("z", 0.0)) - float(low.get("z", 0.0))
        except (TypeError, ValueError):
            return True
        if span_x <= 0 or span_y <= 0 or span_z <= 0:
            # 没有几何信息时不做判断，交给后续的放置校验兜底。
            return True
        if min(span_x, span_y) > self._MAX_FURNITURE_FOOTPRINT:
            return False
        return span_z <= self._MAX_FURNITURE_HEIGHT

    @staticmethod
    def _raw_id_for_object_id(object_id: str, context: TaskContext) -> str | None:
        """模型看到的是服务端 ID，这里找回世界模型使用的原始 ID。"""
        return next(
            (
                raw_id
                for raw_id, mapped_id in context.raw_to_mapped_id.items()
                if str(mapped_id) == object_id
            ),
            None,
        )

    def has_destination(self, destination_type: str) -> bool:
        if destination_type == "dining_table":
            return any(anchor["type"] in {"dining_table", "table"} for anchor in self.scene_anchors.values())
        return any(anchor["type"] == destination_type for anchor in self.scene_anchors.values())

    def apply_semantic_hints(self, parameters: dict[str, Any], context: TaskContext | None) -> None:
        """接受开放式物品标签，但只允许四种规范目的地进入执行状态。"""
        if context is None:
            return
        target_object_id = str(parameters.get("object_id") or "")
        semantic_label = normalize_semantic_label(
            parameters.get("semantic_label") or parameters.get("target_category")
        )
        target_raw_id = self._raw_id_for_object_id(target_object_id, context)
        # 912 不再下发目标清单，目标改由模型在感知中指名。只有带 semantic_label
        # 的动作才算指名——像 move_to_object 那样单纯导航到一件家具时，模型不会
        # 给标签，不能把家具误登记成待整理物品。
        if (
            semantic_label is not None
            and target_raw_id is not None
            and target_raw_id not in self.targets
        ):
            self._merge_declared_targets([target_raw_id])
            # 立刻填上模型看到的 ID，否则本帧的 _raw_id_for_mapped_id 认不出它。
            self.targets[target_raw_id]["object_id"] = target_object_id
            logger.info(
                "Tidy-room discovered target raw_id={} from model object_id={}",
                target_raw_id,
                target_object_id,
            )
        destination_supplied = bool(str(parameters.get("destination_type") or "").strip())
        explicit_destination_type = normalize_destination_type(parameters.get("destination_type"))

        if destination_supplied and explicit_destination_type is None:
            logger.warning(
                "Ignored unsafe tidy-room destination type {!r}; allowed values are "
                "sofa/dining_table/trash_bin/shoe_storage",
                parameters.get("destination_type"),
            )

        if target_raw_id is not None and target_raw_id in self.targets:
            record = self.targets[target_raw_id]
            # 目的地一旦确定就不再改：模型逐帧标注时会摇摆，每次覆盖会让已经
            # 开始的搬运链中途换目的地，甚至把东西放到错的家具上。
            if record.get("destination_type") is None:
                if semantic_label is not None:
                    # category 保留为兼容字段，新代码以 semantic_label + destination_type 为准。
                    record["semantic_label"] = semantic_label
                    record["category"] = semantic_label
                resolved_destination_type = explicit_destination_type
                if resolved_destination_type is None and semantic_label is not None:
                    resolved_destination_type = recommended_destination(semantic_label)
                if resolved_destination_type is not None:
                    record["destination_type"] = resolved_destination_type
            logger.info(
                "Applied tidy-room semantic hint target={} label={} destination={}",
                target_object_id,
                record.get("semantic_label"),
                self.destination_type_for(record),
            )

        destination_object_id = str(parameters.get("destination_object_id") or "")
        # 家具 ID 必须和模型明确给出的合法目的地类型成对出现。不能用物品
        # 标签推导出的目的地去猜一个未知家具 ID，否则可能把书架当成餐桌。
        if destination_object_id and explicit_destination_type is not None:
            self._register_anchor(
                destination_object_id,
                explicit_destination_type,
                context,
                parameters.get("front_side"),
            )

    def apply_scene_annotations(self, annotations: Any, context: TaskContext | None) -> int:
        """接收模型对整幅画面做的一次性语义标注，返回应用的条数。

        912 不下发目标清单，每发现一件物品都要往返一次模型。让模型在回答时
        顺带把画面里其余物品和家具一起标出来，往返次数就从"每件物品一次"
        降到"每帧一次"。

        每条标注二选一：
          * 物品：{"object_id", "semantic_label", "destination_type", "destination_object_id"}
          * 家具：{"anchor_object_id", "destination_type"}
        """
        if context is None or not isinstance(annotations, list):
            return 0

        anchors: list[tuple[str, str, Any]] = []
        targets: list[tuple[dict[str, Any], str]] = []
        for entry in annotations:
            if not isinstance(entry, dict):
                continue
            anchor_object_id = str(entry.get("anchor_object_id") or "")
            object_id = anchor_object_id or str(entry.get("object_id") or "")
            # 解析不到的条目直接丢弃：模型偶尔会写上一个这帧没看到的 ID。
            if not object_id or self._raw_id_for_object_id(object_id, context) is None:
                continue
            destination_type = normalize_destination_type(entry.get("destination_type"))
            # 模型有时用 anchor_object_id 标注家具，有时直接把家具名写进
            # semantic_label（如 "sofa"）。后者不能再当物品，否则沙发会被登记
            # 成待整理目标，角色会去尝试搬走它。
            label_as_destination = normalize_destination_type(entry.get("semantic_label"))
            anchor_type = destination_type or label_as_destination
            # 目的地指向自己（"把 18 放进 18"）同样是模型在说"18 就是这个容器"。
            names_itself_as_destination = str(entry.get("destination_object_id") or "") == object_id
            # 只给 destination_type、不给物品标签，也是在命名家具。
            names_furniture_only = entry.get("semantic_label") is None and destination_type is not None
            is_anchor = (
                bool(anchor_object_id)
                or label_as_destination is not None
                or names_itself_as_destination
                or names_furniture_only
            )
            if is_anchor and anchor_type is not None:
                anchors.append((object_id, anchor_type, entry.get("front_side")))
            elif entry.get("object_id"):
                targets.append((entry, object_id))

        # 先家具后物品：同一件物体不可能既是待整理物品又是目的地，而模型偶尔
        # 会把垃圾桶同时写进两边，那会让角色去搬垃圾桶。
        for object_id, anchor_type, front_side in anchors:
            self._register_anchor(object_id, anchor_type, context, front_side)
        applied = len(anchors)
        for entry, object_id in targets:
            if object_id in self.scene_anchors:
                logger.info("Ignored tidy-room target {}: already a registered destination", object_id)
                continue
            self.apply_semantic_hints(entry, context)
            applied += 1
        if applied:
            logger.info(
                "Applied {} tidy-room scene annotations (targets={} anchors={})",
                applied,
                len(self.targets),
                len(self.scene_anchors),
            )
        return applied

    def _register_anchor(
        self,
        destination_object_id: str,
        destination_type: str,
        context: TaskContext,
        front_side_value: Any,
    ) -> None:
        """把一件家具登记成放置目的地。"""
        destination_raw_id = self._raw_id_for_object_id(destination_object_id, context)
        if destination_raw_id is None:
            return
        if not self._is_furniture_sized(destination_object_id):
            # 地板/墙/天花板都是巨大区域。模型见过把它们当成"鞋架""垃圾桶"，
            # 结果是把东西放到地上、校验还通过——因为落点确实在地板上。
            logger.info(
                "Ignored tidy-room destination {}: the region is too large to be furniture",
                destination_object_id,
            )
            return
        anchor = self.scene_anchors.get(destination_object_id)
        if anchor is None:
            anchor = {
                "raw_id": destination_raw_id,
                "object_id": destination_object_id,
                "type": destination_type,
                "visible": destination_object_id in self.scene_objects,
                "object_info": deepcopy(self.scene_objects.get(destination_object_id) or {}),
            }
            self.scene_anchors[destination_object_id] = anchor
        elif anchor.get("type") != destination_type:
            # 模型逐帧标注时会摇摆（同一件鞋架一会儿标 19 一会儿标 20）。
            # 每帧覆盖会让规划器拿到互相矛盾的落点，以第一次识别为准。
            logger.info(
                "Keeping tidy-room destination {} as {}; ignoring re-labelling as {}",
                destination_object_id,
                anchor.get("type"),
                destination_type,
            )
        anchor["raw_id"] = destination_raw_id
        if destination_object_id in self.scene_objects:
            anchor["object_info"] = deepcopy(self.scene_objects[destination_object_id])
            anchor["visible"] = True
        front_side = str(front_side_value or "").lower()
        if front_side in {"+x", "-x", "+y", "-y"}:
            anchor["front_side"] = front_side

    def _merge_declared_targets(self, movable_objects: list[Any]) -> None:
        for raw_value in movable_objects:
            raw_id = str(raw_value)
            if raw_id in self.targets:
                continue
            self.declared_targets.append(raw_id)
            self.targets[raw_id] = self._new_target_record(raw_id)

    def _update_targets(
        self,
        context: TaskContext,
        visible_by_id: dict[str, dict[str, Any]],
    ) -> None:
        for record in self.targets.values():
            record["visible"] = False
        for raw_id in self.declared_targets:
            record = self.targets[raw_id]
            mapped_id = context.raw_to_mapped_id.get(raw_id)
            if mapped_id is not None:
                mapped_id_str = str(mapped_id)
                record["object_id"] = mapped_id_str
                current_info = visible_by_id.get(mapped_id_str)
                record["visible"] = current_info is not None
                if current_info is not None:
                    record["object_info"] = deepcopy(current_info)
            refreshed_aabb = context.world_aabbs_by_raw_id.get(raw_id)
            if refreshed_aabb:
                record.setdefault("object_info", {})["world_aabb"] = deepcopy(refreshed_aabb)

    def _update_anchors(
        self,
        context: TaskContext,
        visible_by_id: dict[str, dict[str, Any]],
    ) -> None:
        for anchor in self.scene_anchors.values():
            anchor["visible"] = False
        target_set = set(self.declared_targets)
        for raw_id, mapped_id in context.raw_to_mapped_id.items():
            if raw_id in target_set:
                continue
            anchor_type = infer_anchor_type(raw_id)
            if anchor_type is None:
                continue
            mapped_id_str = str(mapped_id)
            current_info = visible_by_id.get(mapped_id_str)
            anchor = self.scene_anchors.setdefault(
                mapped_id_str,
                {
                    "raw_id": raw_id,
                    "object_id": mapped_id_str,
                    "type": anchor_type,
                    "visible": False,
                    "object_info": {},
                },
            )
            anchor["raw_id"] = raw_id
            anchor["type"] = anchor_type
            anchor["visible"] = current_info is not None
            if current_info is not None:
                anchor["object_info"] = deepcopy(current_info)
                self.scene_objects[mapped_id_str] = deepcopy(current_info)
            refreshed_aabb = context.world_aabbs_by_raw_id.get(raw_id)
            if refreshed_aabb:
                anchor.setdefault("object_info", {})["world_aabb"] = deepcopy(refreshed_aabb)
                self.scene_objects.setdefault(mapped_id_str, {})["world_aabb"] = deepcopy(refreshed_aabb)

    @staticmethod
    def _new_target_record(raw_id: str) -> dict[str, Any]:
        semantic_label = infer_target_category(raw_id)
        return {
            "object_id": None,
            # semantic_label 开放保存具体名称；destination_type 才是受限执行类别。
            "category": semantic_label,
            "semantic_label": semantic_label,
            "destination_type": recommended_destination(semantic_label),
            "status": "pending",
            "visible": False,
            "object_info": {},
            "direct_force_place_disabled": False,
            "placement_verification_failures": 0,
        }
