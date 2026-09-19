from __future__ import annotations

from copy import deepcopy
from math import hypot
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
from arenaagent.preliminary_baseline_agent.tasks.tidyroom.geometry import (
    center_xy,
    horizontal_area,
)
from arenaagent.preliminary_baseline_agent.tasks.tidyroom.furniture_priors import (
    FIXED_FURNITURE_PRIORS,
    PRIOR_MATCH_CENTER_TOLERANCE,
    PRIOR_MATCH_SPAN_TOLERANCE,
)

_NUT_MIN_SPAN_CM = 3.0
_NUT_MAX_SPAN_CM = 8.0
_NUT_MAX_ASPECT_RATIO = 1.8


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
        # 当局分割 ID -> 固定家具语义。视觉模型无权把这些建筑/
        # 家具再登记成待拾取目标。
        self.fixed_furniture_labels: dict[str, str] = {}
        self.fixed_room_prior_active = False
        self.rejected_anchor_ids: set[str] = set()

    def observe(self, context: TaskContext) -> None:
        """合并本帧可见数据和按原始 ID 查询到的全局 AABB。"""
        self._merge_declared_targets(context.movable_objects)
        visible_by_id = {str(item.get("object_id")): item for item in context.visible_objects}
        for object_id, info in visible_by_id.items():
            self.scene_objects[object_id] = deepcopy(info)
        self._apply_fixed_furniture_priors(context)
        self._update_targets(context, visible_by_id)
        self._update_anchors(context, visible_by_id)

    def tracked_raw_ids(self) -> tuple[str, ...]:
        """持续刷新全部目标和已经发现的目的地，而非只看当前截图。"""
        raw_ids = list(self.declared_targets)
        raw_ids.extend(
            str(anchor["raw_id"])
            for anchor in self.scene_anchors.values()
            if anchor.get("raw_id")
        )
        return tuple(dict.fromkeys(raw_ids))

    def fixed_furniture_for_prompt(self) -> dict[str, str]:
        """返回首帧已按坐标确认的家具，供盘点提示词排除误拾取。"""
        return dict(self.fixed_furniture_labels)

    @staticmethod
    def _aabb_signature(aabb: Any) -> tuple[float, float, float, float, float] | None:
        if not isinstance(aabb, dict):
            return None
        low, high = aabb.get("min") or {}, aabb.get("max") or {}
        try:
            min_x, min_y, min_z = float(low["x"]), float(low["y"]), float(low["z"])
            max_x, max_y, max_z = float(high["x"]), float(high["y"]), float(high["z"])
        except (KeyError, TypeError, ValueError):
            return None
        return (
            (min_x + max_x) / 2.0,
            (min_y + max_y) / 2.0,
            max_x - min_x,
            max_y - min_y,
            max_z - min_z,
        )

    def _match_fixed_furniture(self, prior: dict[str, Any]) -> str | None:
        expected = self._aabb_signature(prior.get("world_aabb"))
        if expected is None:
            return None
        matches: list[tuple[float, str]] = []
        for object_id, info in self.scene_objects.items():
            actual = self._aabb_signature(info.get("world_aabb"))
            if actual is None:
                continue
            center_error = hypot(actual[0] - expected[0], actual[1] - expected[1])
            span_error = max(abs(actual[index] - expected[index]) for index in (2, 3, 4))
            if (
                center_error <= PRIOR_MATCH_CENTER_TOLERANCE
                and span_error <= PRIOR_MATCH_SPAN_TOLERANCE
            ):
                matches.append((center_error + span_error, object_id))
        return min(matches)[1] if matches else None

    def _apply_fixed_furniture_priors(self, context: TaskContext) -> None:
        """用固定世界坐标绑定当局 ID，并预先建立可放置家具 anchor。"""
        matched = {
            prior_name: self._match_fixed_furniture(prior)
            for prior_name, prior in FIXED_FURNITURE_PRIORS.items()
        }
        # 不能把这份先验污染到其他场景：至少三件固定家具同时
        # 命中才认定是这间 912 客厅。一旦认定，后续局部视野也继续生效。
        if not self.fixed_room_prior_active:
            if sum(object_id is not None for object_id in matched.values()) < 3:
                return
            self.fixed_room_prior_active = True
            logger.info("Tidy-room activated fixed 912 furniture-coordinate priors")

        for prior_name, prior in FIXED_FURNITURE_PRIORS.items():
            object_id = matched[prior_name]
            if object_id is not None:
                label = str(prior.get("label") or prior_name)
                if self.fixed_furniture_labels.get(object_id) != label:
                    logger.info(
                        "Tidy-room bound fixed furniture prior {} to object_id={}",
                        label,
                        object_id,
                    )
                self.fixed_furniture_labels[object_id] = label

            destination_type = str(prior.get("destination_type") or "")
            if not destination_type:
                continue
            synthetic_id = f"fixed:{destination_type}"
            if object_id is not None:
                self.scene_anchors.pop(synthetic_id, None)
                raw_id = self._raw_id_for_object_id(object_id, context) or ""
                object_info = deepcopy(self.scene_objects.get(object_id) or {})
                anchor_id = object_id
                source = "fixed_coordinate_prior"
            else:
                # 当首帧分割漏掉家具时仍可用固定 AABB 规划隔空落点。
                raw_id = ""
                object_info = {
                    "object_id": synthetic_id,
                    "shape": "fixed_furniture_prior",
                    "world_aabb": deepcopy(prior["world_aabb"]),
                }
                anchor_id = synthetic_id
                source = "fixed_coordinate_prior_unbound"
            self.scene_anchors[anchor_id] = {
                "raw_id": raw_id,
                "object_id": anchor_id,
                "type": destination_type,
                "visible": object_id is not None,
                "object_info": object_info,
                "source": source,
            }

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
    _MAX_PICKUP_FOOTPRINT = 140.0
    _MAX_PICKUP_HEIGHT = 80.0

    def _is_furniture_sized(self, object_id: str, destination_type: str | None = None) -> bool:
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
        if span_z > self._MAX_FURNITURE_HEIGHT:
            return False
        horizontal_area = span_x * span_y
        if destination_type == "dining_table":
            # 本场景餐桌约 160x80；模型经常把周围约 60x60 的椅子也全部
            # 标成 dining_table。餐桌必须有明显的大水平承载面。
            if max(span_x, span_y) < 100.0 or horizontal_area < 6000.0 or span_z < 45.0:
                return False
        if destination_type == "sofa":
            # 主沙发至少有一条长边；脚凳、单椅及沙发上的分割小块不能作为
            # 抱枕目的地，否则规划点会落到家具外。
            if max(span_x, span_y) < 140.0 or horizontal_area < 10_000.0:
                return False
        if destination_type == "trash_bin":
            if max(span_x, span_y) > 100.0 or horizontal_area < 300.0:
                return False
        if destination_type == "shoe_storage":
            # 实测误识别的落地灯只有约 27x18 的占地、顶部却在 136cm。
            # 鞋架/鞋柜至少应有一条明显的水平承载边，并具有足够的底面积。
            if max(span_x, span_y) < 45.0 or horizontal_area < 800.0:
                return False
        return True

    def _is_plausible_pickup_sized(self, object_id: str) -> bool:
        """排除 NPC 和大型家具被视觉模型误登记为鞋、杯子等可拾取物。"""
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
            return True
        return bool(
            max(span_x, span_y) <= self._MAX_PICKUP_FOOTPRINT
            and span_z <= self._MAX_PICKUP_HEIGHT
        )

    def _item_spans(self, object_id: str) -> tuple[float, float, float] | None:
        """返回物体 AABB 三边长度；缺失或退化几何不参与语义纠错。"""
        info = self.scene_objects.get(object_id) or {}
        aabb = info.get("world_aabb") or {}
        low, high = aabb.get("min") or {}, aabb.get("max") or {}
        try:
            spans = (
                abs(float(high["x"]) - float(low["x"])),
                abs(float(high["y"]) - float(low["y"])),
                abs(float(high["z"]) - float(low["z"])),
            )
        except (KeyError, TypeError, ValueError):
            return None
        return spans if all(span > 0.0 for span in spans) else None

    def _correct_item_semantics_by_geometry(
        self,
        object_id: str,
        semantic_label: str | None,
        destination_type: str | None,
    ) -> tuple[str | None, str | None]:
        """用明显的尺度矛盾修正视觉标签，不猜测几何相近的物品。

        实测错误中，4×4×3 cm 的核桃被标成 pillow，而约 45×16×15 cm
        的长抱枕被标成 cup。两者与标签的尺度矛盾足够强，可以在执行前确定性
        纠正；普通罐瓶及其他开放式食物仍保留模型标签。
        """
        spans = self._item_spans(object_id)
        if semantic_label is None or destination_type is None or spans is None:
            return semantic_label, destination_type
        span_x, span_y, span_z = spans
        horizontal_long = max(span_x, span_y)
        horizontal_short = min(span_x, span_y)
        original = (semantic_label, destination_type)

        info = self.scene_objects.get(object_id) or {}
        color = str(info.get("color") or "").strip().lower()
        shape = str(info.get("shape") or "").strip().lower()
        trash_labels = {"trash", "garbage", "rubbish", "litter", "waste", "wrapper"}
        compact_brown_nut = bool(
            _NUT_MIN_SPAN_CM <= min(spans)
            and max(spans) <= _NUT_MAX_SPAN_CM
            and max(spans) / min(spans) <= _NUT_MAX_ASPECT_RATIO
            and color in {"brown", "dark brown"}
            and shape in {"irregular", "sphere", "spherical", "round", "unknown"}
        )
        if (
            destination_type == "trash_bin"
            and semantic_label in trash_labels
            and compact_brown_nut
        ):
            # 17:53 实测：棕色不规则核桃约 4.88×4.46×5.05cm，被视觉模型
            # 标成 trash。只对紧凑近球形的棕色小物体修正，避免泛化到纸屑等垃圾。
            semantic_label, destination_type = "food_walnut", "dining_table"

        if destination_type == "sofa" and "pillow" in semantic_label:
            if horizontal_long < 15.0 or span_z < 6.0:
                semantic_label, destination_type = "small_food", "dining_table"
        elif destination_type == "dining_table" and semantic_label in {
            "cup",
            "mug",
            "bottle",
            "can",
            "drink_can",
            "drink_container",
            "food_or_drink_container",
        }:
            # 圆柱抱枕横放时是一条长边加两个约 15cm 的截面；杯罐和饮料瓶
            # 即使倒地，截面通常也明显更小。
            if horizontal_long >= 32.0 and horizontal_short >= 12.0 and 8.0 <= span_z <= 35.0:
                semantic_label, destination_type = "pillow", "sofa"

        if (semantic_label, destination_type) != original:
            logger.warning(
                "Corrected tidy-room item semantics by AABB object_id={} spans={} "
                "from {}->{} to {}->{}",
                object_id,
                tuple(round(span, 2) for span in spans),
                original[0],
                original[1],
                semantic_label,
                destination_type,
            )
        return semantic_label, destination_type

    def discover_geometric_pillow_targets(self, context: TaskContext | None) -> int:
        """补回盘点模型漏掉的地面长抱枕。

        本场景真实长抱枕约 45×16×15cm；脚凳约 75×59×27cm。这里只接受
        窄范围的长条软物体，并排除已经在主沙发范围内的物体，不把通用小物
        枚举成目标。
        """
        if context is None:
            return 0
        known_object_ids = {
            str(record.get("object_id"))
            for record in self.targets.values()
            if record.get("object_id") is not None
        }
        discovered = 0
        for object_id in sorted(self.scene_objects):
            if object_id in known_object_ids or object_id in self.scene_anchors:
                continue
            spans = self._item_spans(object_id)
            if spans is None:
                continue
            span_x, span_y, span_z = spans
            horizontal_long = max(span_x, span_y)
            horizontal_short = min(span_x, span_y)
            is_bolster_sized = bool(
                32.0 <= horizontal_long <= 65.0
                and 12.0 <= horizontal_short <= 35.0
                and 8.0 <= span_z <= 30.0
            )
            if not is_bolster_sized or self._is_already_at_destination(object_id, "sofa"):
                continue
            raw_id = self._raw_id_for_object_id(object_id, context)
            if raw_id is None:
                continue
            self.apply_semantic_hints(
                {
                    "object_id": object_id,
                    "semantic_label": "pillow",
                    "destination_type": "sofa",
                },
                context,
            )
            if raw_id in self.targets:
                discovered += 1
                known_object_ids.add(object_id)
                logger.info(
                    "Tidy-room recovered omitted bolster target object_id={} spans={}",
                    object_id,
                    tuple(round(span, 2) for span in spans),
                )
        return discovered

    def _is_already_at_destination(self, object_id: str, destination_type: str) -> bool:
        """过滤目的地自带的装饰物，尤其是鞋架中不可拾取的展示鞋。"""
        target_aabb = (self.scene_objects.get(object_id) or {}).get("world_aabb") or {}
        target_low, target_high = target_aabb.get("min") or {}, target_aabb.get("max") or {}
        try:
            target_center_x = (float(target_low["x"]) + float(target_high["x"])) / 2.0
            target_center_y = (float(target_low["y"]) + float(target_high["y"])) / 2.0
            target_min_z = float(target_low["z"])
            target_max_z = float(target_high["z"])
        except (KeyError, TypeError, ValueError):
            return False

        # 鞋架场景中的展示鞋可能摆在格子前沿，水平 AABB 略微伸出家具；
        # 其余目的地只给较小容差，避免把落在家具旁的真正杂物过滤掉。
        margin = 70.0 if destination_type == "shoe_storage" else 20.0
        for anchor in self.scene_anchors.values():
            if anchor.get("type") != destination_type:
                continue
            anchor_aabb = (anchor.get("object_info") or {}).get("world_aabb") or {}
            low, high = anchor_aabb.get("min") or {}, anchor_aabb.get("max") or {}
            try:
                near_xy = bool(
                    float(low["x"]) - margin <= target_center_x <= float(high["x"]) + margin
                    and float(low["y"]) - margin <= target_center_y <= float(high["y"]) + margin
                )
                near_z = bool(
                    target_min_z <= float(high["z"]) + 40.0
                    and target_max_z >= float(low["z"]) - 10.0
                )
            except (KeyError, TypeError, ValueError):
                continue
            if near_xy and near_z:
                return True
        return False

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

    # 视觉模型被明确要求"不要把单椅标成沙发、不要把椅子标成餐桌"，于是它
    # 常把扶手椅标成 dining_table 交差。实测本场景：被标成餐桌的其实是一件
    # 111x90x84 的扶手椅，物品会穿过"桌面"落到座面上；真正的餐桌是
    # 160x80x76，旁边就摆着一把 shape=chair 的椅子。标注不可信时退回纯几何
    # 挑选，最终由落点高度证伪（见 strategy 的落地矛盾判定）。
    _CHAIR_NEIGHBOUR_RADIUS = 150.0

    def geometric_candidates(self, destination_type: str) -> list[str]:
        """按几何给这个目的地类型排序候选家具，不依赖模型标注。"""
        taken = {
            str(anchor.get("object_id"))
            for anchor in self.scene_anchors.values()
            if anchor.get("object_id") is not None
        }
        scored: list[tuple[float, str]] = []
        for object_id, info in self.scene_objects.items():
            if object_id in taken or object_id in self.rejected_anchor_ids:
                continue
            if str(info.get("shape") or "Unknown") == "Unknown":
                # 地板、墙、天花板都没有语义形状，纯按尺寸最容易混进来。
                continue
            if not self._is_furniture_sized(object_id, destination_type):
                continue
            score = horizontal_area(info.get("world_aabb"))
            if destination_type == "dining_table" and self._has_chair_neighbour(object_id):
                # 餐桌旁边一定有椅子；这一条就能把它和尺寸相近的扶手椅分开。
                score += 1e9
            scored.append((score, object_id))
        scored.sort(reverse=True)
        return [object_id for _, object_id in scored]

    def promote_geometric_candidate(self, destination_type: str) -> str | None:
        """没有可信标注时，把几何上最像的家具登记为目的地。"""
        candidates = self.geometric_candidates(destination_type)
        if not candidates:
            return None
        object_id = candidates[0]
        self.scene_anchors[object_id] = {
            "raw_id": object_id,
            "object_id": object_id,
            "type": destination_type,
            "visible": bool(self.scene_objects.get(object_id)),
            "object_info": deepcopy(self.scene_objects.get(object_id) or {}),
            "source": "geometric_candidate",
        }
        logger.info(
            "Tidy-room promoted geometric {} candidate object_id={}",
            destination_type,
            object_id,
        )
        return object_id

    def _has_chair_neighbour(self, object_id: str) -> bool:
        center = center_xy((self.scene_objects.get(object_id) or {}).get("world_aabb"))
        if center is None:
            return False
        for other_id, info in self.scene_objects.items():
            if other_id == object_id or str(info.get("shape") or "") != "chair":
                continue
            other = center_xy(info.get("world_aabb"))
            if other is None:
                continue
            if hypot(other[0] - center[0], other[1] - center[1]) <= self._CHAIR_NEIGHBOUR_RADIUS:
                return True
        return False

    def reject_anchor(self, object_id: str, reason: str) -> None:
        """永久拒绝本局中已经被几何执行证明不可靠的目的地。"""
        object_id = str(object_id or "")
        if not object_id:
            return
        removed = self.scene_anchors.pop(object_id, None)
        self.rejected_anchor_ids.add(object_id)
        logger.warning(
            "Rejected tidy-room destination object_id={} type={} reason={}",
            object_id,
            (removed or {}).get("type"),
            reason,
        )

    def apply_semantic_hints(self, parameters: dict[str, Any], context: TaskContext | None) -> None:
        """接受开放式物品标签，但只允许四种规范目的地进入执行状态。"""
        if context is None:
            return
        target_object_id = str(parameters.get("object_id") or "")
        if target_object_id in self.fixed_furniture_labels:
            logger.info(
                "Ignored tidy-room pickup object_id={} because fixed-coordinate prior identifies it as {}",
                target_object_id,
                self.fixed_furniture_labels[target_object_id],
            )
            return
        semantic_label = normalize_semantic_label(
            parameters.get("semantic_label") or parameters.get("target_category")
        )
        destination_supplied = bool(str(parameters.get("destination_type") or "").strip())
        explicit_destination_type = normalize_destination_type(parameters.get("destination_type"))
        semantic_label, explicit_destination_type = self._correct_item_semantics_by_geometry(
            target_object_id,
            semantic_label,
            explicit_destination_type,
        )
        if semantic_label is not None and explicit_destination_type is not None:
            inferred_destination_type = recommended_destination(semantic_label)
            # 已知物品必须遵循本地映射。对词表外标签保持开放（例如
            # ceramic_plate），但明确拒绝人物、建筑表面、植物和家具本体；
            # 否则恢复模型会把绿植或椅子登记成需要搬运的物品。
            environment_tokens = {
                "person",
                "human",
                "npc",
                "plant",
                "wall",
                "floor",
                "ceiling",
                "chair",
                "table",
                "sofa",
                "couch",
                "cabinet",
                "shelf",
                "lamp",
                "television",
                "tv",
                "refrigerator",
                "fridge",
                "furniture",
            }
            label_tokens = set(semantic_label.split("_"))
            is_environment_label = bool(label_tokens & environment_tokens)
            if (
                inferred_destination_type is not None
                and inferred_destination_type != explicit_destination_type
            ) or (inferred_destination_type is None and is_environment_label):
                logger.warning(
                    "Ignored incompatible tidy-room item object_id={} label={} destination={}",
                    target_object_id,
                    semantic_label,
                    explicit_destination_type,
                )
                return
        target_raw_id = self._raw_id_for_object_id(target_object_id, context)
        # 912 不再下发目标清单，目标改由模型在感知中指名。只有带 semantic_label
        # 的动作才算指名——像 move_to_object 那样单纯导航到一件家具时，模型不会
        # 给标签，不能把家具误登记成待整理物品。
        if (
            semantic_label is not None
            and target_raw_id is not None
            and target_raw_id not in self.targets
        ):
            if not self._is_plausible_pickup_sized(target_object_id):
                logger.warning(
                    "Ignored implausibly large tidy-room pickup object_id={} label={}",
                    target_object_id,
                    semantic_label,
                )
                return
            self._merge_declared_targets([target_raw_id])
            # 立刻填上模型看到的 ID，否则本帧的 _raw_id_for_mapped_id 认不出它。
            self.targets[target_raw_id]["object_id"] = target_object_id
            logger.info(
                "Tidy-room discovered target raw_id={} from model object_id={}",
                target_raw_id,
                target_object_id,
            )
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
        # 单个物品条目无权凭 destination_object_id 创造家具。家具必须先由
        # furniture/anchor_object_id 条目独立确认；否则模型的一次错配会直接
        # 变成可执行落点（本次日志就是把落地灯 20 当成鞋架）。
        if destination_object_id and explicit_destination_type is not None:
            anchor = self.scene_anchors.get(destination_object_id)
            if anchor is None or anchor.get("type") != explicit_destination_type:
                logger.warning(
                    "Ignored unconfirmed tidy-room destination object_id={} type={} for target={}",
                    destination_object_id,
                    explicit_destination_type,
                    target_object_id,
                )
            else:
                front_side = str(parameters.get("front_side") or "").lower()
                if front_side in {"+x", "-x", "+y", "-y"}:
                    anchor["front_side"] = front_side

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
        applied = 0
        for object_id, anchor_type, front_side in anchors:
            if self._register_anchor(object_id, anchor_type, context, front_side):
                applied += 1
        for entry, object_id in targets:
            if object_id in self.scene_anchors:
                logger.info("Ignored tidy-room target {}: already a registered destination", object_id)
                continue
            trusted_entry = deepcopy(entry)
            destination_object_id = str(trusted_entry.get("destination_object_id") or "")
            destination_type = normalize_destination_type(trusted_entry.get("destination_type"))
            if destination_object_id:
                anchor = self.scene_anchors.get(destination_object_id)
                if anchor is None or anchor.get("type") != destination_type:
                    logger.warning(
                        "Ignored unconfirmed scene destination object_id={} type={} for item={}",
                        destination_object_id,
                        trusted_entry.get("destination_type"),
                        object_id,
                    )
                    trusted_entry.pop("destination_object_id", None)
            destination_type = normalize_destination_type(trusted_entry.get("destination_type"))
            if destination_type is None:
                destination_type = recommended_destination(
                    str(trusted_entry.get("semantic_label") or "")
                )
            if destination_type and self._is_already_at_destination(object_id, destination_type):
                logger.info(
                    "Ignored tidy-room target {}: already at destination {}",
                    object_id,
                    destination_type,
                )
                continue
            self.apply_semantic_hints(trusted_entry, context)
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
    ) -> bool:
        """把一件家具登记成放置目的地。"""
        if destination_object_id in self.rejected_anchor_ids:
            logger.info(
                "Ignored previously rejected tidy-room destination {} as {}",
                destination_object_id,
                destination_type,
            )
            return False
        destination_raw_id = self._raw_id_for_object_id(destination_object_id, context)
        if destination_raw_id is None:
            return False
        fixed_label = self.fixed_furniture_labels.get(destination_object_id)
        fixed_destination_type = {
            "main_sofa": "sofa",
            "dining_table": "dining_table",
            "trash_bin": "trash_bin",
        }.get(fixed_label)
        if fixed_label is not None and fixed_destination_type != destination_type:
            logger.info(
                "Ignored tidy-room destination {} as {}: fixed-coordinate prior identifies it as {}",
                destination_object_id,
                destination_type,
                fixed_label,
            )
            return False
        if not self._is_furniture_sized(destination_object_id, destination_type):
            # 同时排除地板/墙等巨大区域，以及落地灯这类不符合特定家具
            # 几何形态的细长小底面。
            logger.info(
                "Ignored tidy-room destination {} as {}: implausible furniture geometry",
                destination_object_id,
                destination_type,
            )
            return False
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
        return True

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
            if mapped_id_str in self.rejected_anchor_ids:
                continue
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
        }
