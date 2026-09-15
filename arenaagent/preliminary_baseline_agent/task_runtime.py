from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TaskContext:
    """Read-only-by-convention snapshot passed to one task strategy."""

    task_type: str
    subject: dict[str, Any]
    task_response: dict[str, Any]
    visible_objects: list[dict[str, Any]]
    object_in_hand: Any
    movable_objects: list[Any]
    action_histories: list[dict[str, Any]]
    last_action_result: Any = field(default_factory=dict)
    raw_to_mapped_id: dict[str, str] = field(default_factory=dict)
    world_aabbs_by_raw_id: dict[str, dict[str, Any]] = field(default_factory=dict)


def normalize_task_type(subject: Any) -> str:
    """Return the five-task routing key while tolerating stage-style subjects."""
    if not isinstance(subject, dict):
        return "unknown"

    raw = str(subject.get("task_type") or subject.get("stage") or "").strip().lower()
    aliases = {
        "tidy_room": "tidyroom",
        "tidy-room": "tidyroom",
        "tidyroom": "tidyroom",
        "jigsaw_room": "jigsaw",
        "counting_room": "counting",
        "npc_room": "npc",
        "raven_room": "raven",
    }
    return aliases.get(raw, raw or "unknown")
