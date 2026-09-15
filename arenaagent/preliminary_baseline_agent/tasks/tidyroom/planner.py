from __future__ import annotations

from copy import deepcopy
from math import hypot
from typing import Any

from arenaagent.preliminary_baseline_agent.tasks.tidyroom.geometry import (
    center_xy,
    contains_xy,
    expanded_contains_xy,
    horizontal_area,
    is_floor_region,
    normalize_aabb,
    normalize_location,
)

APPROACH_CLEARANCE = 60.0
FLOOR_SAFETY_MARGIN = 25.0
OBSTACLE_CLEARANCE = 35.0
MIN_BLOCKING_OBJECT_HEIGHT = 15.0


class TidyRoomPlanner:
    """根据家具、地板和障碍物 AABB 计算可达接近点与放置点。"""

    def __init__(self) -> None:
        self.scene_objects: dict[str, dict[str, Any]] = {}
        self.floor_regions: dict[str, dict[str, Any]] = {}

    def observe(self, visible_objects: list[dict[str, Any]]) -> None:
        for item in visible_objects:
            object_id = str(item.get("object_id") or "")
            if not object_id:
                continue
            self.scene_objects[object_id] = deepcopy(item)
            if is_floor_region(item):
                self.floor_regions[object_id] = deepcopy(item)

    def build_plan(
        self,
        raw_id: str,
        target: dict[str, Any],
        destination_type: str,
        anchors: dict[str, dict[str, Any]],
        slot_index: int,
    ) -> dict[str, Any] | None:
        anchor = self.select_anchor(destination_type, anchors, target.get("object_info") or {})
        if anchor is None:
            return None
        plan = {
            "target_raw_id": raw_id,
            "target_object_id": target.get("object_id"),
            "target_category": target.get("category"),
            "target_info": deepcopy(target.get("object_info") or {}),
            "destination_raw_id": anchor["raw_id"],
            "destination_object_id": anchor["object_id"],
            "destination_type": destination_type,
            "destination_info": deepcopy(anchor.get("object_info") or {}),
            "attempt": 0,
            "slot_index": slot_index,
            "last_failure": None,
        }
        if anchor.get("front_side"):
            plan["front_side"] = anchor["front_side"]
        self.refresh_plan(plan, anchors)
        return plan

    def refresh_plan(self, plan: dict[str, Any], anchors: dict[str, dict[str, Any]]) -> None:
        anchor = anchors.get(str(plan.get("destination_object_id") or ""))
        if anchor and anchor.get("object_info"):
            plan["destination_info"] = deepcopy(anchor["object_info"])
        if anchor and anchor.get("front_side"):
            plan["front_side"] = anchor["front_side"]
        destination_type = str(plan.get("destination_type") or "")
        if destination_type == "sofa":
            move_location, put_location, support_region, support_z = self._plan_sofa(plan, anchors)
        else:
            move_location, put_location, support_region, support_z = self._plan_flat_destination(plan)
        plan["move_target_location"] = move_location
        plan["put_target_location"] = put_location
        plan["support_region"] = support_region
        plan["support_z"] = support_z

    def select_anchor(
        self,
        destination_type: str,
        anchors: dict[str, dict[str, Any]],
        target_info: dict[str, Any],
    ) -> dict[str, Any] | None:
        candidates = [anchor for anchor in anchors.values() if anchor["type"] == destination_type]
        if not candidates and destination_type == "dining_table":
            candidates = [anchor for anchor in anchors.values() if anchor["type"] == "table"]
        if not candidates:
            return None
        if destination_type == "sofa":
            # 场景中的扶手椅和脚凳也带 Sofa 名称，水平占地最大的才是主沙发。
            return max(candidates, key=lambda anchor: horizontal_area(anchor.get("object_info", {}).get("world_aabb")))

        target_center = center_xy(target_info.get("world_aabb"))
        if target_center is None:
            return max(candidates, key=lambda anchor: horizontal_area(anchor.get("object_info", {}).get("world_aabb")))

        def distance(anchor: dict[str, Any]) -> float:
            anchor_center = center_xy(anchor.get("object_info", {}).get("world_aabb"))
            if anchor_center is None:
                return float("inf")
            return hypot(anchor_center[0] - target_center[0], anchor_center[1] - target_center[1])

        return min(candidates, key=distance)

    def _plan_flat_destination(
        self,
        plan: dict[str, Any],
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], float]:
        destination_info = plan.get("destination_info") or {}
        target_info = plan.get("target_info") or {}
        destination_aabb = normalize_aabb(destination_info.get("world_aabb"))
        target_aabb = normalize_aabb(target_info.get("world_aabb"))
        if destination_aabb is None:
            fallback = self._api_location(destination_info.get("place_location"))
            empty_region = {
                "min_x": fallback["X"],
                "max_x": fallback["X"],
                "min_y": fallback["Y"],
                "max_y": fallback["Y"],
            }
            return fallback, fallback, empty_region, fallback["Z"]

        minimum, maximum = destination_aabb
        target_half_x, target_half_y = self._target_half_extents(target_aabb)
        destination_type = str(plan.get("destination_type") or "")
        if destination_type == "trash_bin":
            region = {
                "min_x": minimum["x"] + target_half_x + 1.0,
                "max_x": maximum["x"] - target_half_x - 1.0,
                "min_y": minimum["y"] + target_half_y + 1.0,
                "max_y": maximum["y"] - target_half_y - 1.0,
            }
            support_z = minimum["z"]
        else:
            region = {
                "min_x": minimum["x"] + target_half_x + 8.0,
                "max_x": maximum["x"] - target_half_x - 8.0,
                "min_y": minimum["y"] + target_half_y + 8.0,
                "max_y": maximum["y"] - target_half_y - 8.0,
            }
            support_z = maximum["z"]

        if destination_type == "trash_bin":
            # 垃圾桶空间较窄，两件罐子不能使用“中心 + 轻微偏移”，否则刚体
            # 会相互重叠。直接分配左右两个间距更大的桶内槽位。
            put_x, put_y = self._container_slot_point(
                region,
                int(plan.get("slot_index", 0)),
                int(plan.get("attempt", 0)),
            )
        else:
            put_x, put_y = self._slot_point(
                region,
                int(plan.get("slot_index", 0)),
                int(plan.get("attempt", 0)),
            )
        put_z = self._put_origin_z(plan, target_aabb, support_z, destination_aabb)
        destination_center = ((minimum["x"] + maximum["x"]) / 2.0, (minimum["y"] + maximum["y"]) / 2.0)
        move_location = self._reachable_approach(
            destination_aabb,
            destination_center,
            str(plan.get("destination_object_id") or ""),
            str(plan.get("target_object_id") or ""),
            int(plan.get("attempt", 0)),
        )
        return move_location, {"X": put_x, "Y": put_y, "Z": put_z}, region, support_z

    def _plan_sofa(
        self,
        plan: dict[str, Any],
        anchors: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], float]:
        destination_info = plan.get("destination_info") or {}
        target_info = plan.get("target_info") or {}
        destination_aabb = normalize_aabb(destination_info.get("world_aabb"))
        target_aabb = normalize_aabb(target_info.get("world_aabb"))
        if destination_aabb is None:
            return self._plan_flat_destination(plan)

        minimum, maximum = destination_aabb
        width_x = maximum["x"] - minimum["x"]
        width_y = maximum["y"] - minimum["y"]
        depth_axis = "x" if width_x <= width_y else "y"
        long_axis = "y" if depth_axis == "x" else "x"
        sofa_center = ((minimum["x"] + maximum["x"]) / 2.0, (minimum["y"] + maximum["y"]) / 2.0)
        reference_center = self._nearest_reference_center("coffee_table", anchors, sofa_center)
        front_sign = self._front_sign_from_hint(str(plan.get("front_side") or ""), depth_axis)
        if front_sign is None:
            front_sign = self._front_sign(depth_axis, sofa_center, reference_center)
        target_half_x, target_half_y = self._target_half_extents(target_aabb)
        region = self._sofa_seat_region(
            minimum,
            maximum,
            depth_axis,
            front_sign,
            target_half_x,
            target_half_y,
        )
        put_x, put_y = self._slot_point(region, int(plan.get("slot_index", 0)), int(plan.get("attempt", 0)))
        support_z = minimum["z"] + 0.45 * (maximum["z"] - minimum["z"])
        put_z = self._put_origin_z(plan, target_aabb, support_z, destination_aabb)

        long_coordinate = put_y if long_axis == "y" else put_x
        if depth_axis == "x":
            approach_x = maximum["x"] + APPROACH_CLEARANCE if front_sign > 0 else minimum["x"] - APPROACH_CLEARANCE
            approach = {"X": approach_x, "Y": long_coordinate, "Z": 0.0}
        else:
            approach_y = maximum["y"] + APPROACH_CLEARANCE if front_sign > 0 else minimum["y"] - APPROACH_CLEARANCE
            approach = {"X": long_coordinate, "Y": approach_y, "Z": 0.0}
        attempt = int(plan.get("attempt", 0))
        if attempt > 0 or not self._is_reachable_point(
            approach,
            str(plan.get("destination_object_id") or ""),
            "",
        ):
            approach = self._reachable_approach(
                destination_aabb,
                sofa_center,
                str(plan.get("destination_object_id") or ""),
                str(plan.get("target_object_id") or ""),
                attempt,
            )
        return approach, {"X": put_x, "Y": put_y, "Z": put_z}, region, support_z

    def _reachable_approach(
        self,
        destination_aabb: tuple[dict[str, float], dict[str, float]],
        destination_center: tuple[float, float],
        destination_object_id: str,
        target_object_id: str,
        attempt: int = 0,
    ) -> dict[str, float]:
        minimum, maximum = destination_aabb
        center_x, center_y = destination_center
        candidates = (
            {"X": minimum["x"] - APPROACH_CLEARANCE, "Y": center_y, "Z": 0.0},
            {"X": maximum["x"] + APPROACH_CLEARANCE, "Y": center_y, "Z": 0.0},
            {"X": center_x, "Y": minimum["y"] - APPROACH_CLEARANCE, "Z": 0.0},
            {"X": center_x, "Y": maximum["y"] + APPROACH_CLEARANCE, "Z": 0.0},
        )
        valid = [
            point
            for point in candidates
            if self._is_reachable_point(point, destination_object_id, target_object_id)
        ]
        if not valid:
            valid = [
                point
                for point in candidates
                if self._inside_destination_floor(point["X"], point["Y"], margin=10.0)
            ]
        if not valid:
            return {"X": center_x, "Y": center_y, "Z": 0.0}
        floor_center = self._destination_floor_center(center_x, center_y)
        if floor_center is not None:
            valid.sort(key=lambda point: hypot(point["X"] - floor_center[0], point["Y"] - floor_center[1]))
        # 每次失败轮换一个已验证的接近点，避免在墙边重复同一路径。
        return valid[attempt % len(valid)]

    def _is_reachable_point(self, point: dict[str, float], destination_id: str, target_id: str) -> bool:
        if not self._inside_destination_floor(point["X"], point["Y"], FLOOR_SAFETY_MARGIN):
            return False
        ignored_ids = set(self.floor_regions) | {destination_id, target_id}
        for object_id, info in self.scene_objects.items():
            if object_id in ignored_ids or is_floor_region(info):
                continue
            aabb = normalize_aabb(info.get("world_aabb"))
            if aabb is None or aabb[1]["z"] < MIN_BLOCKING_OBJECT_HEIGHT:
                continue
            if expanded_contains_xy(aabb, point["X"], point["Y"], OBSTACLE_CLEARANCE):
                return False
        return True

    def _inside_destination_floor(self, x_value: float, y_value: float, margin: float) -> bool:
        return any(
            contains_xy(info.get("world_aabb"), x_value, y_value, margin)
            for info in self.floor_regions.values()
        )

    def _destination_floor_center(self, x_value: float, y_value: float) -> tuple[float, float] | None:
        containing = [
            info
            for info in self.floor_regions.values()
            if contains_xy(info.get("world_aabb"), x_value, y_value)
        ]
        if not containing:
            return None
        return center_xy(max(containing, key=lambda info: horizontal_area(info.get("world_aabb"))).get("world_aabb"))

    @staticmethod
    def _sofa_seat_region(  # noqa: PLR0917
        minimum: dict[str, float],
        maximum: dict[str, float],
        depth_axis: str,
        front_sign: int,
        target_half_x: float,
        target_half_y: float,
    ) -> dict[str, float]:
        center_x = (minimum["x"] + maximum["x"]) / 2.0
        center_y = (minimum["y"] + maximum["y"]) / 2.0
        region = {
            "min_x": minimum["x"] + target_half_x + 8.0,
            "max_x": maximum["x"] - target_half_x - 8.0,
            "min_y": minimum["y"] + target_half_y + 8.0,
            "max_y": maximum["y"] - target_half_y - 8.0,
        }
        if depth_axis == "x":
            if front_sign > 0:
                region["min_x"] = center_x
            else:
                region["max_x"] = center_x
        elif front_sign > 0:
            region["min_y"] = center_y
        else:
            region["max_y"] = center_y
        return region

    @staticmethod
    def _front_sign(
        depth_axis: str,
        sofa_center: tuple[float, float],
        reference_center: tuple[float, float] | None,
    ) -> int:
        if reference_center is None:
            return 1
        axis_index = 0 if depth_axis == "x" else 1
        return 1 if reference_center[axis_index] >= sofa_center[axis_index] else -1

    @staticmethod
    def _front_sign_from_hint(front_side: str, depth_axis: str) -> int | None:
        if front_side not in {f"+{depth_axis}", f"-{depth_axis}"}:
            return None
        return 1 if front_side.startswith("+") else -1

    @staticmethod
    def _nearest_reference_center(
        anchor_type: str,
        anchors: dict[str, dict[str, Any]],
        origin: tuple[float, float],
    ) -> tuple[float, float] | None:
        centers = [
            center
            for anchor in anchors.values()
            if anchor.get("type") == anchor_type
            if (center := center_xy(anchor.get("object_info", {}).get("world_aabb"))) is not None
        ]
        return min(centers, key=lambda point: hypot(point[0] - origin[0], point[1] - origin[1])) if centers else None

    @staticmethod
    def _target_half_extents(target_aabb: tuple[dict[str, float], dict[str, float]] | None) -> tuple[float, float]:
        if target_aabb is None:
            return 3.0, 3.0
        minimum, maximum = target_aabb
        return (maximum["x"] - minimum["x"]) / 2.0, (maximum["y"] - minimum["y"]) / 2.0

    @staticmethod
    def _slot_point(region: dict[str, float], slot_index: int, attempt: int) -> tuple[float, float]:
        center_x = (region["min_x"] + region["max_x"]) / 2.0
        center_y = (region["min_y"] + region["max_y"]) / 2.0
        width = max(region["max_x"] - region["min_x"], 0.0)
        depth = max(region["max_y"] - region["min_y"], 0.0)
        patterns = ((0.0, 0.0), (-0.3, 0.0), (0.3, 0.0), (0.0, -0.3), (0.0, 0.3))
        x_factor, y_factor = patterns[(slot_index + attempt) % len(patterns)]
        return center_x + x_factor * width, center_y + y_factor * depth

    @staticmethod
    def _container_slot_point(region: dict[str, float], slot_index: int, attempt: int) -> tuple[float, float]:
        center_x = (region["min_x"] + region["max_x"]) / 2.0
        center_y = (region["min_y"] + region["max_y"]) / 2.0
        width = max(region["max_x"] - region["min_x"], 0.0)
        depth = max(region["max_y"] - region["min_y"], 0.0)
        patterns = ((-0.42, 0.0), (0.42, 0.0), (0.0, -0.42), (0.0, 0.42))
        x_factor, y_factor = patterns[(slot_index + attempt) % len(patterns)]
        return center_x + x_factor * width, center_y + y_factor * depth

    @staticmethod
    def _put_origin_z(
        plan: dict[str, Any],
        target_aabb: tuple[dict[str, float], dict[str, float]] | None,
        support_z: float,
        destination_aabb: tuple[dict[str, float], dict[str, float]],
    ) -> float:
        target_place = normalize_location((plan.get("target_info") or {}).get("place_location"))
        destination_type = str(plan.get("destination_type") or "")
        if destination_type == "trash_bin":
            # put_down_to_location 在本场景中基本把传入 Z 当作物体底部。
            # 原实现按桶口对齐，导致罐子被固定在半空；现在放到桶底上方，
            # 随后开启物理模拟让它自然落底。
            return destination_aabb[0]["z"] + 2.0

        if destination_type == "dining_table":
            # 本场景的 put_down_to_location 会把传入 Z 直接作为物体底部。
            # 不能在重试时混用“移动后的 AABB”和“初始 place_location”计算
            # pivot 偏移，否则高度会在桌面上方和地板下方之间跳变。
            return support_z + 2.0

        if str(plan.get("target_category") or "") == "food_apple":
            # 苹果原始 AABB 会因初始旋转向下延伸，不能用原始 pivot 偏移
            # 推算放置高度，否则会悬在桌面上方约 16cm。
            return support_z + 2.0

        bottom_offset = 0.0
        if target_aabb is not None and target_place is not None:
            bottom_offset = target_aabb[0]["z"] - target_place["z"]
        z_adjustment = 0.0
        failure = plan.get("last_failure") or {}
        if failure.get("reason") == "below_surface":
            z_adjustment = min(5.0 * int(plan.get("attempt", 0)), 15.0)
        elif failure.get("reason") == "above_surface":
            z_adjustment = -min(3.0 * int(plan.get("attempt", 0)), 9.0)
        return support_z - bottom_offset + 2.0 + z_adjustment

    @staticmethod
    def _api_location(value: Any) -> dict[str, float]:
        location = normalize_location(value) or {"x": 0.0, "y": 0.0, "z": 0.0}
        return {"X": location["x"], "Y": location["y"], "Z": location["z"]}
