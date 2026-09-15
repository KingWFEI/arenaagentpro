from __future__ import annotations

from itertools import permutations
from math import hypot
from typing import Any

from arenaagent.preliminary_baseline_agent.tasks.tidyroom.geometry import center_xy
from arenaagent.preliminary_baseline_agent.tasks.tidyroom.planner import TidyRoomPlanner
from arenaagent.preliminary_baseline_agent.tasks.tidyroom.world_model import TidyRoomWorldModel


class TargetScheduler:
    """根据距离、可见性和失败次数选择下一件物品，避免模型随机挑选。"""

    def __init__(self) -> None:
        self.estimated_agent_xy: tuple[float, float] | None = None

    def choose(
        self,
        world: TidyRoomWorldModel,
        planner: TidyRoomPlanner,
        failure_counts: dict[str, int],
        direct_destination_types: frozenset[str] = frozenset(),
    ) -> str | None:
        candidates: dict[str, tuple[tuple[float, float] | None, tuple[float, float] | None, str]] = {}
        for raw_id in world.pending_raw_ids():
            target = world.targets[raw_id]
            if target.get("object_id") is None:
                continue
            destination_type = world.destination_type_for(target)
            if destination_type is None:
                continue
            anchor = planner.select_anchor(destination_type, world.scene_anchors, target.get("object_info") or {})
            if anchor is None:
                continue
            target_xy = center_xy(target.get("object_info", {}).get("world_aabb"))
            destination_xy = center_xy(anchor.get("object_info", {}).get("world_aabb"))
            candidates[raw_id] = target_xy, destination_xy, destination_type
        if not candidates:
            return None

        # 最多五个目标，直接穷举剩余顺序比逐步最近邻更可靠。代价模型同时
        # 考虑普通放置后角色位于家具旁，而 force_locate 放置不会移动角色。
        # 失败惩罚只施加给下一件，确保反复失败的目标先让位给其他物品。
        best: tuple[float, tuple[str, ...]] | None = None
        for order in permutations(sorted(candidates)):
            cost = self._route_cost(order, candidates, direct_destination_types)
            cost += 200.0 * failure_counts.get(order[0], 0)
            candidate = cost, order
            if best is None or candidate < best:
                best = candidate
        return best[1][0] if best is not None else None

    def _route_cost(
        self,
        order: tuple[str, ...],
        candidates: dict[
            str,
            tuple[tuple[float, float] | None, tuple[float, float] | None, str],
        ],
        direct_destination_types: frozenset[str],
    ) -> float:
        current = self.estimated_agent_xy
        total = 0.0
        for raw_id in order:
            target_xy, destination_xy, destination_type = candidates[raw_id]
            if target_xy is None:
                # 仍允许处理缺少 AABB 的唯一候选，但优先安排几何信息完整者。
                total += 10_000.0
                continue
            if current is not None:
                total += hypot(target_xy[0] - current[0], target_xy[1] - current[1])
            if destination_type in direct_destination_types:
                current = target_xy
                continue
            if destination_xy is None:
                total += 10_000.0
                current = target_xy
                continue
            total += hypot(target_xy[0] - destination_xy[0], target_xy[1] - destination_xy[1])
            current = destination_xy
        return total

    def mark_at_target(self, target: dict[str, Any]) -> None:
        location = center_xy(target.get("object_info", {}).get("world_aabb"))
        if location is not None:
            self.estimated_agent_xy = location

    def mark_at_destination(self, plan: dict[str, Any]) -> None:
        location = plan.get("move_target_location") or {}
        try:
            self.estimated_agent_xy = float(location["X"]), float(location["Y"])
        except (KeyError, TypeError, ValueError):
            pass
