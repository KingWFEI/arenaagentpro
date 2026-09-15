from __future__ import annotations

from arenaagent.preliminary_baseline_agent.tasks.tidyroom.recovery import RecoveryPolicy
from arenaagent.preliminary_baseline_agent.tasks.tidyroom.scanner import RoomScanner
from arenaagent.preliminary_baseline_agent.tasks.tidyroom.world_model import TidyRoomWorldModel


class VLMEscalationPolicy:
    """集中判断何时值得花费一次视觉模型调用。"""

    def reason(
        self,
        world: TidyRoomWorldModel,
        scanner: RoomScanner,
        recovery: RecoveryPolicy,
        active_raw_id: str | None,
    ) -> str | None:
        if not scanner.complete:
            return None
        if recovery.needs_vlm(active_raw_id):
            return "local_recovery_exhausted"
        for raw_id in world.pending_raw_ids():
            target = world.targets[raw_id]
            if target.get("semantic_label") == "movable_item":
                return "unknown_target_category"
            if target.get("object_id") is None:
                return "target_not_observed"
            destination_type = world.destination_type_for(target)
            if destination_type is None:
                return "unknown_target_destination"
            if not world.has_destination(destination_type):
                return f"missing_destination:{destination_type}"
        return None
