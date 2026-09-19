"""912 整理房间的固定家具坐标先验。

这些坐标来自同一赛题房间的多次结构化 AABB 采样。分割 object_id
可以改变，世界坐标不变，所以运行时按 AABB 中心和尺寸绑定当局 ID。
"""

from __future__ import annotations

from typing import Any


def _entry(
    label: str,
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
    destination_type: str | None = None,
) -> dict[str, Any]:
    return {
        "label": label,
        "destination_type": destination_type,
        "world_aabb": {
            "min": {"x": minimum[0], "y": minimum[1], "z": minimum[2]},
            "max": {"x": maximum[0], "y": maximum[1], "z": maximum[2]},
        },
    }


# key 只是场景资产名，绝不是运行时 object_id。
FIXED_FURNITURE_PRIORS: dict[str, dict[str, Any]] = {
    "plant": _entry(
        "plant",
        (251.9161, 567.0827, 0.0049),
        (338.2224, 646.4863, 100.9607),
    ),
    "armchair": _entry(
        "armchair",
        (486.1363, 539.6050, -0.5156),
        (597.7376, 630.0629, 83.4156),
    ),
    "ottoman": _entry(
        "ottoman",
        (350.5103, 175.4482, 1.6703),
        (409.4897, 250.5518, 28.6020),
    ),
    "main_sofa": _entry(
        "main_sofa",
        (226.5241, 169.1135, -0.4823),
        (338.4065, 579.4561, 98.9269),
        "sofa",
    ),
    "coffee_table": _entry(
        "coffee_table",
        (541.2239, 296.5392, 15.9612),
        (613.1496, 433.4986, 45.0344),
    ),
    "dining_table": _entry(
        "dining_table",
        (-297.0, 481.0, 3.0),
        (-137.0, 561.0, 78.7572),
        "dining_table",
    ),
    "tv_console": _entry(
        "tv_console",
        (792.7104, 292.5033, -0.2203),
        (835.1895, 533.1001, 37.2154),
    ),
    "trash_bin": _entry(
        "trash_bin",
        (800.3900, 253.2911, 3.8783),
        (827.5173, 280.5479, 38.8245),
        "trash_bin",
    ),
    "television": _entry(
        "television",
        (834.0103, 314.8070, 76.3724),
        (839.2478, 505.0789, 186.6224),
    ),
    "floor_lamp": _entry(
        "floor_lamp",
        (595.5012, 597.6069, 1.8868),
        (645.0281, 637.9233, 136.4018),
    ),
    "dining_chair_1": _entry(
        "dining_chair",
        (-280.9839, 402.2015, -0.3665),
        (-194.9604, 488.2896, 79.7715),
    ),
    "dining_chair_2": _entry(
        "dining_chair",
        (-187.5152, 413.9111, -0.3665),
        (-126.6942, 475.2563, 79.7715),
    ),
    "dining_chair_3": _entry(
        "dining_chair",
        (-189.4535, 571.1393, -0.3665),
        (-128.6324, 632.4844, 79.7715),
    ),
    "dining_chair_4": _entry(
        "dining_chair",
        (-282.4535, 571.1393, -0.3665),
        (-221.6324, 632.4844, 79.7715),
    ),
}


PRIOR_MATCH_CENTER_TOLERANCE = 12.0
PRIOR_MATCH_SPAN_TOLERANCE = 12.0
