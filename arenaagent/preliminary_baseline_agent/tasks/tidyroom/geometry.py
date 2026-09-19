from __future__ import annotations

from typing import Any

Point3 = dict[str, float]
Aabb = tuple[Point3, Point3]
AABB_POINT_COUNT = 2
MIN_FLOOR_AREA = 20_000.0
MAX_FLOOR_THICKNESS = 10.0
MIN_FLOOR_TOP_Z = -20.0
MAX_FLOOR_TOP_Z = 15.0


def normalize_location(value: Any) -> Point3 | None:
    if not isinstance(value, dict):
        return None
    lowered = {str(key).lower(): item for key, item in value.items()}
    try:
        return {"x": float(lowered["x"]), "y": float(lowered["y"]), "z": float(lowered["z"])}
    except (KeyError, TypeError, ValueError):
        return None


def normalize_aabb(value: Any) -> Aabb | None:
    if isinstance(value, dict) and "world_aabb" in value:
        value = value["world_aabb"]
    if isinstance(value, dict):
        minimum = normalize_location(value.get("min"))
        maximum = normalize_location(value.get("max"))
    elif isinstance(value, (tuple, list)) and len(value) == AABB_POINT_COUNT:
        minimum = normalize_location(value[0])
        maximum = normalize_location(value[1])
    else:
        return None
    if minimum is None or maximum is None:
        return None
    return minimum, maximum


def center_xy(value: Any) -> tuple[float, float] | None:
    aabb = normalize_aabb(value)
    if aabb is None:
        return None
    minimum, maximum = aabb
    return (minimum["x"] + maximum["x"]) / 2.0, (minimum["y"] + maximum["y"]) / 2.0


def horizontal_area(value: Any) -> float:
    aabb = normalize_aabb(value)
    if aabb is None:
        return 0.0
    minimum, maximum = aabb
    return max(maximum["x"] - minimum["x"], 0.0) * max(maximum["y"] - minimum["y"], 0.0)


def is_floor_region(info: dict[str, Any]) -> bool:
    aabb = normalize_aabb(info.get("world_aabb"))
    if aabb is None:
        return False
    minimum, maximum = aabb
    thickness = maximum["z"] - minimum["z"]
    area = max(maximum["x"] - minimum["x"], 0.0) * max(maximum["y"] - minimum["y"], 0.0)
    return (
        area >= MIN_FLOOR_AREA
        and thickness <= MAX_FLOOR_THICKNESS
        and MIN_FLOOR_TOP_Z <= maximum["z"] <= MAX_FLOOR_TOP_Z
    )


def contains_xy(value: Any, x_value: float, y_value: float, margin: float = 0.0) -> bool:
    aabb = normalize_aabb(value)
    if aabb is None:
        return False
    minimum, maximum = aabb
    return (
        minimum["x"] + margin <= x_value <= maximum["x"] - margin
        and minimum["y"] + margin <= y_value <= maximum["y"] - margin
    )


def expanded_contains_xy(value: Any, x_value: float, y_value: float, expansion: float) -> bool:
    aabb = normalize_aabb(value)
    if aabb is None:
        return False
    minimum, maximum = aabb
    return (
        minimum["x"] - expansion <= x_value <= maximum["x"] + expansion
        and minimum["y"] - expansion <= y_value <= maximum["y"] + expansion
    )


def placement_check(
    target_aabb_value: Any,
    destination_aabb_value: Any,
    plan: dict[str, Any],
) -> dict[str, Any]:
    target_aabb = normalize_aabb(target_aabb_value)
    destination_aabb = normalize_aabb(destination_aabb_value)
    if target_aabb is None or destination_aabb is None:
        return {"valid": False, "reason": "missing_aabb"}

    target_minimum, target_maximum = target_aabb
    destination_minimum, destination_maximum = destination_aabb
    target_center = {
        "x": (target_minimum["x"] + target_maximum["x"]) / 2.0,
        "y": (target_minimum["y"] + target_maximum["y"]) / 2.0,
        "z": (target_minimum["z"] + target_maximum["z"]) / 2.0,
    }
    destination_type = str(plan.get("destination_type") or "")

    if destination_type == "trash_bin":
        # 官方评测关注的是“真正放进桶里”。除了中心点之外，还要求目标
        # AABB 基本完整位于桶内，并且底部已经落到桶底附近，拒绝悬空假成功。
        inside_xy = (
            target_minimum["x"] >= destination_minimum["x"] - 1.0
            and target_maximum["x"] <= destination_maximum["x"] + 1.0
            and target_minimum["y"] >= destination_minimum["y"] - 1.0
            and target_maximum["y"] <= destination_maximum["y"] + 1.0
        )
        inside_z = (
            target_minimum["z"] >= destination_minimum["z"] - 3.0
            and target_maximum["z"] <= destination_maximum["z"] + 2.0
        )
        resting_on_bottom = target_minimum["z"] <= destination_minimum["z"] + 8.0
        valid = inside_xy and inside_z and resting_on_bottom
        if valid:
            reason = "ok"
        elif not inside_xy:
            reason = "outside_xy"
        elif not inside_z:
            reason = "outside_z"
        else:
            reason = "not_resting_in_container"
        return {
            "valid": valid,
            "reason": reason,
            "actual_center": target_center,
            "actual_bottom_z": target_minimum["z"],
            "expected_bottom_z": destination_minimum["z"],
            "destination_aabb": destination_aabb_value,
        }

    region = plan.get("support_region") or {}
    inside_xy = (
        float(region.get("min_x", destination_minimum["x"])) <= target_center["x"]
        <= float(region.get("max_x", destination_maximum["x"]))
        and float(region.get("min_y", destination_minimum["y"])) <= target_center["y"]
        <= float(region.get("max_y", destination_maximum["y"]))
    )
    expected_support_z = float(plan.get("support_z", destination_maximum["z"]))
    tolerance_below = 12.0 if destination_type == "sofa" else 5.0
    tolerance_above = 15.0 if destination_type == "sofa" else 6.0
    bottom_z = target_minimum["z"]
    height_ok = expected_support_z - tolerance_below <= bottom_z <= expected_support_z + tolerance_above
    if inside_xy and height_ok:
        reason = "ok"
    elif not inside_xy:
        reason = "outside_xy"
    elif bottom_z < expected_support_z - tolerance_below:
        reason = "below_surface"
    else:
        reason = "above_surface"
    return {
        "valid": inside_xy and height_ok,
        "reason": reason,
        "actual_center": target_center,
        "actual_bottom_z": bottom_z,
        "expected_support_z": expected_support_z,
        "support_region": region,
        "destination_aabb": destination_aabb_value,
    }
