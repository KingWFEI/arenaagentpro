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

    def has_destination(self, destination_type: str) -> bool:
        if destination_type == "dining_table":
            return any(anchor["type"] in {"dining_table", "table"} for anchor in self.scene_anchors.values())
        return any(anchor["type"] == destination_type for anchor in self.scene_anchors.values())

    def apply_semantic_hints(self, parameters: dict[str, Any], context: TaskContext | None) -> None:
        """接受开放式物品标签，但只允许四种规范目的地进入执行状态。"""
        if context is None:
            return
        target_object_id = str(parameters.get("object_id") or "")
        target_raw_id = next(
            (
                raw_id
                for raw_id, mapped_id in context.raw_to_mapped_id.items()
                if str(mapped_id) == target_object_id and raw_id in self.targets
            ),
            None,
        )
        semantic_label = normalize_semantic_label(
            parameters.get("semantic_label") or parameters.get("target_category")
        )
        destination_supplied = bool(str(parameters.get("destination_type") or "").strip())
        explicit_destination_type = normalize_destination_type(parameters.get("destination_type"))

        if destination_supplied and explicit_destination_type is None:
            logger.warning(
                "Ignored unsafe tidy-room destination type {!r}; allowed values are "
                "sofa/dining_table/trash_bin/shoe_storage",
                parameters.get("destination_type"),
            )

        if target_raw_id is not None:
            record = self.targets[target_raw_id]
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
        if not destination_object_id or explicit_destination_type is None:
            return
        destination_type = explicit_destination_type
        destination_raw_id = next(
            (
                raw_id
                for raw_id, mapped_id in context.raw_to_mapped_id.items()
                if str(mapped_id) == destination_object_id
            ),
            None,
        )
        if destination_raw_id is None:
            return
        anchor = self.scene_anchors.setdefault(
            destination_object_id,
            {
                "raw_id": destination_raw_id,
                "object_id": destination_object_id,
                "type": destination_type,
                "visible": destination_object_id in self.scene_objects,
                "object_info": deepcopy(self.scene_objects.get(destination_object_id) or {}),
            },
        )
        anchor["raw_id"] = destination_raw_id
        anchor["type"] = destination_type
        if destination_object_id in self.scene_objects:
            anchor["object_info"] = deepcopy(self.scene_objects[destination_object_id])
            anchor["visible"] = True
        front_side = str(parameters.get("front_side") or "").lower()
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
