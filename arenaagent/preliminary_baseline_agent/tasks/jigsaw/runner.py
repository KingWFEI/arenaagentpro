from __future__ import annotations

from typing import Any

from loguru import logger

from arenaagent.preliminary_baseline_agent.tasks.base import action_succeeded
from arenaagent.preliminary_baseline_agent.tasks.jigsaw.solver import infer_layout, solve_by_image_cost
from arenaagent.tongsim_interface import Rotation


def _raw_object_id(agent: Any, mapped_id: Any) -> str:
    mapper = agent.semantic_mapper
    try:
        raw_id = mapper.get_raw_id(int(mapped_id))
    except (TypeError, ValueError):
        raw_id = None
    if not raw_id:
        raise ValueError(f"jigsaw candidate {mapped_id} has no raw object ID")
    return str(raw_id)


def _require_success(action: str, result: Any) -> None:
    if not action_succeeded(result):
        raise RuntimeError(f"jigsaw {action} failed: {result}")


def run_dedicated_jigsaw(agent: Any, subject: dict[str, Any]) -> None:
    """Run YH's image/geometry solver using this agent's gRPC TongSim interface."""
    if agent.tongsim is None or agent.semantic_mapper is None or not agent.character_id:
        raise RuntimeError("jigsaw scene client is not initialized")

    image, _, objects = agent._acquire_camera_perception(subject, include_images=True, include_object_details=True)
    if not image:
        raise RuntimeError("jigsaw perception image is missing")

    # The shared perception layer exposes short mapped IDs to the model. The
    # solver can use them, but physical actions must use the original object IDs.
    enriched = []
    for obj in objects:
        item = dict(obj)
        raw_id = _raw_object_id(agent, item.get("object_id"))
        item["raw_object_id"] = raw_id
        basic_info = agent.tongsim.get_object_basic_info(raw_id)
        if isinstance(basic_info, dict):
            item["rotation"] = basic_info.get("rotation") or {}
        enriched.append(item)

    layout = infer_layout(enriched, subject.get("reference_bounding") or [])
    mapping = solve_by_image_cost(image, layout)
    logger.info("Dedicated jigsaw mapping: {}", mapping)

    candidates = {str(obj["object_id"]): obj for obj in layout.candidates}
    cells = {cell.name: cell for cell in layout.empty_cells}
    rotation = Rotation(**layout.target_rotation)
    def take_piece(placement: dict[str, str], which_hand: int, *, allow_failure: bool = False) -> bool:
        candidate = candidates[placement["object_id"]]
        object_id = candidate["raw_object_id"]
        approach = agent.tongsim.move_to_object(agent.character_id, object_id)
        if not action_succeeded(approach):
            if allow_failure:
                logger.warning("Could not approach jigsaw prefetch candidate {}: {}", object_id, approach)
                return False
            _require_success("approach candidate", approach)
        take = agent.tongsim.move_and_take_object(agent.character_id, object_id, which_hand=which_hand)
        if not action_succeeded(take):
            if allow_failure:
                logger.warning("Could not prefetch jigsaw candidate {} with hand {}: {}", object_id, which_hand, take)
                return False
            _require_success("take candidate", take)
        return True

    def place_piece(placement: dict[str, str], which_hand: int) -> None:
        candidate = candidates[placement["object_id"]]
        object_id = candidate["raw_object_id"]
        target = list(cells[placement["cell"]].location)
        _require_success(
            "approach destination",
            agent.tongsim.move_to_location(agent.character_id, target, stop_distance=30.0),
        )
        _require_success(
            "place candidate",
            agent.tongsim.put_down_to_location(
                agent.character_id,
                target_location=target,
                which_hand=which_hand,
                rotation=rotation,
                auto_rotate=False,
                force_locate=True,
            ),
        )
        after = agent.tongsim.get_object_basic_info(object_id)
        location = after.get("place_location") if isinstance(after, dict) else None
        if isinstance(location, dict) and all(axis in location for axis in ("X", "Y", "Z")):
            if max(abs(float(location[axis]) - target[index]) for index, axis in enumerate(("X", "Y", "Z"))) > 3.0:
                raise RuntimeError(f"jigsaw placement did not reach {placement['cell']}: {location}")
        logger.info("Placed jigsaw piece {} in {}", object_id, placement["cell"])

    # Retain YH's action order: take the first tile with hand 0, try to
    # prefetch the second with hand 1, then place both before taking the third.
    take_piece(mapping[0], which_hand=0)
    second_prefetched = len(mapping) > 1 and take_piece(mapping[1], which_hand=1, allow_failure=True)
    place_piece(mapping[0], which_hand=0)

    # Some TongSim implementations may report a failed prefetch even though
    # the object entered hand 1. Check only after releasing hand 0, since the
    # single-object get_object_in_hand response cannot describe both hands.
    if len(mapping) > 1 and not second_prefetched:
        held = agent.tongsim.get_object_in_hand(agent.character_id)
        second_id = candidates[mapping[1]["object_id"]]["raw_object_id"]
        second_prefetched = bool(held and str(held[0]) == second_id and int(held[1]) == 1)

    next_index = 1
    if second_prefetched:
        logger.info("Jigsaw second candidate was prefetched in hand 1")
        place_piece(mapping[1], which_hand=1)
        next_index = 2

    for placement in mapping[next_index:]:
        take_piece(placement, which_hand=0)
        place_piece(placement, which_hand=0)
