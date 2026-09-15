from __future__ import annotations

from typing import Any

from arenaagent.preliminary_baseline_agent.tasks.tidyroom.geometry import normalize_aabb, placement_check


class TidyRoomVerifier:
    """只使用 TongSim 结构化状态验证抓取和放置，不根据截图猜测。"""

    @staticmethod
    def verify_pickup(expected_raw_id: str, observed_raw_id: str | None) -> dict[str, Any]:
        if observed_raw_id == expected_raw_id:
            return {"valid": True, "reason": "ok"}
        if observed_raw_id is None:
            return {"valid": False, "reason": "pickup_empty"}
        return {
            "valid": False,
            "reason": "wrong_object_in_hand",
            "observed_raw_id": observed_raw_id,
        }

    @staticmethod
    def verify_placement(
        target_aabb: Any,
        destination_aabb: Any,
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        if normalize_aabb(target_aabb) is None or normalize_aabb(destination_aabb) is None:
            return {"valid": False, "reason": "missing_aabb"}
        return placement_check(target_aabb, destination_aabb, plan)
