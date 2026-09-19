from __future__ import annotations

import re


# 执行层只允许把物品送往这四类评分目的地。具体物品名称可以开放，
# 但不能让模型创造任意家具类型并直接驱动动作。
ALLOWED_DESTINATION_TYPES = frozenset({"sofa", "dining_table", "trash_bin", "shoe_storage"})


def normalize_semantic_label(value: object) -> str | None:
    """把 VLM 的开放式物品名称整理成稳定、可缓存的语义标签。"""
    label = re.sub(r"[^\w]+", "_", str(value or "").strip().lower(), flags=re.UNICODE).strip("_")
    return label[:80] or None


def normalize_destination_type(value: object) -> str | None:
    """只返回四种规范目的地；table 仅作为餐桌的输入兼容别名。"""
    normalized = normalize_semantic_label(value)
    aliases = {
        "table": "dining_table",
        "diningtable": "dining_table",
        "couch": "sofa",
        "dustbin": "trash_bin",
        "waste_bin": "trash_bin",
        "shoe_rack": "shoe_storage",
        "shoe_cabinet": "shoe_storage",
    }
    normalized = aliases.get(normalized or "", normalized)
    return normalized if normalized in ALLOWED_DESTINATION_TYPES else None


def infer_target_category(raw_id: str) -> str:
    value = raw_id.lower()
    rules = (
        (("pillow", "cushion"), "pillow"),
        (("drinkcontainer", "drink_container", "cup", "mug"), "drink_container"),
        (("apple",), "food_apple"),
        (("walnut",), "food_walnuts"),
        (("shoe", "slipper"), "shoes"),
        (("trash", "garbage"), "garbage"),
    )
    for keywords, category in rules:
        if any(keyword in value for keyword in keywords):
            return category
    return "movable_item"


def recommended_destination(category: str) -> str | None:
    """从已知或开放式语义标签推导目的地，无法确定时返回 None。"""
    label = normalize_semantic_label(category) or ""
    if label in {"pillow", "cushion", "bolster"} or any(
        word in label for word in ("pillow", "cushion")
    ):
        return "sofa"
    if label in {"shoe", "shoes", "slipper", "slippers", "sneaker", "sneakers", "boot", "boots"}:
        return "shoe_storage"
    if label in {"discarded_drink_can", "garbage", "trash", "rubbish", "litter", "waste"}:
        return "trash_bin"
    if any(word in label for word in ("discarded", "garbage", "trash", "rubbish", "litter", "waste")):
        return "trash_bin"
    if label in {
        "drink_container",
        "cup",
        "mug",
        "bottle",
        "can",
        "drink_can",
        "tableware",
        "food",
        "fruit",
        "snack",
        "meal",
        "walnut",
        "nuts",
    } or label.startswith(("food_", "fruit_", "drink_", "beverage_")):
        return "dining_table"
    return None


def infer_anchor_type(raw_id: str) -> str | None:
    value = raw_id.lower()
    if "dining" in value and "table" in value:
        return "dining_table"
    if "coffee" in value and "table" in value:
        return "coffee_table"
    rules = (
        (("trash", "garbage", "waste", "dustbin"), "trash_bin"),
        (("sofa", "couch"), "sofa"),
        (("refrigerator", "fridge"), "refrigerator"),
        (("table",), "table"),
    )
    for keywords, anchor_type in rules:
        if any(keyword in value for keyword in keywords):
            return anchor_type
    if "shoe" in value and any(keyword in value for keyword in ("rack", "cabinet", "shelf")):
        return "shoe_storage"
    return None
