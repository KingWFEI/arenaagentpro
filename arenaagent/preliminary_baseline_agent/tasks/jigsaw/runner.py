from __future__ import annotations

import os
from pathlib import Path
from time import strftime
from typing import Any

from loguru import logger

from arenaagent.preliminary_baseline_agent.tasks.base import action_succeeded
from arenaagent.preliminary_baseline_agent.tasks.jigsaw.solver import (
    infer_layout,
    rotation_distance,
    save_diagnostic_montage,
    save_perception_snapshot,
    solve_by_image_cost,
)
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

    # The integrated client dials this competition RPC dynamically, because
    # its checked-in proto predates the server. Detect the live capability
    # from the returned perception, not from the generated stub's attributes.
    acquire_unified = getattr(agent.tongsim, "acquire_first_person_perception", None)
    perception = (
        acquire_unified(agent.character_id, width=1280, height=720)
        if acquire_unified is not None
        else None
    )
    legacy_rpc = perception is not None
    if legacy_rpc:
        image = perception.get("image")
        objects = perception.get("objects") or []
        enriched = []
        for obj in objects:
            item = dict(obj)
            item["raw_object_id"] = str(item["object_id"])
            enriched.append(item)
        logger.info("Using legacy TongSim jigsaw perception compatibility path")
    else:
        image, _, objects = agent._acquire_camera_perception(
            subject,
            include_images=True,
            include_object_details=True,
        )
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

    if not image:
        raise RuntimeError("jigsaw perception image is missing")

    layout = infer_layout(enriched, subject.get("reference_bounding") or [])
    diagnostics_dir = os.getenv("JIGSAW_DIAGNOSTICS_DIR", "").strip()
    diagnostic_cell = os.getenv("JIGSAW_DIAGNOSTIC_CELL", "").strip()
    diagnostic_stamp = strftime("%Y%m%d_%H%M%S")
    if diagnostics_dir:
        diagnostic_path = save_diagnostic_montage(
            image,
            layout,
            Path(diagnostics_dir) / f"jigsaw_{diagnostic_stamp}_before.jpg",
        )
        logger.info("Saved local jigsaw diagnostic montage to {}", diagnostic_path)
    mapping = solve_by_image_cost(image, layout)
    logger.info("Dedicated jigsaw mapping: {}", mapping)
    logger.info(
        "Jigsaw candidate rotations: {}; board rotations: {}; representative: {}",
        {str(obj["object_id"]): obj.get("rotation") for obj in layout.candidates},
        {str(obj["object_id"]): obj.get("rotation") for obj in layout.placed},
        layout.target_rotation,
    )

    candidates = {str(obj["object_id"]): obj for obj in layout.candidates}
    cells = {cell.name: cell for cell in layout.empty_cells}
    movable_ids = [str(obj["raw_object_id"]) for obj in (*layout.candidates, *layout.placed)]
    # Build the complete placement plan before touching the first piece.  The
    # arena evaluator uses the canonical wall-plane yaw (90 degrees). At that
    # yaw the tile's local X/Z face lies in world Y/Z; an in-face quarter-turn
    # is about its local Y axis (UE pitch), not roll (which tilts the face).
    target_yaw_setting = os.getenv("JIGSAW_TARGET_YAW", "90").strip().lower()
    if target_yaw_setting == "observed":
        base_rotation = dict(layout.target_rotation)
    else:
        base_rotation = {"roll": 0.0, "pitch": 0.0, "yaw": float(target_yaw_setting)}
    image_rotation_sign = float(os.getenv("JIGSAW_IMAGE_ROTATION_SIGN", "1"))

    placement_plan: list[dict[str, Any]] = []
    for solved in mapping:
        image_angle = int(solved.get("image_rotation_degrees", 0))
        if os.getenv("JIGSAW_IMAGE_ANGLE_MODE", "image").strip().lower() == "pose":
            # Retain the validated zero-angle fallback for an A/B evaluation.
            image_angle = 0
        pitch = float(base_rotation.get("pitch", 0.0)) + image_rotation_sign * image_angle
        pitch = (pitch + 180.0) % 360.0 - 180.0
        cell = cells[str(solved["cell"])]
        placement_plan.append(
            {
                "object_id": str(solved["object_id"]),
                "cell": cell.name,
                "target_location": list(cell.location),
                "target_rotation": {
                    "roll": float(base_rotation.get("roll", 0.0)),
                    "pitch": pitch,
                    "yaw": float(base_rotation.get("yaw", 90.0)),
                },
                "image_rotation_degrees": image_angle,
            }
        )

    # Place the visually distinctive top-middle horn first.  It then has the
    # longest settling time and can no longer be the just-released piece when
    # evaluation begins.
    placement_plan.sort(key=lambda item: item["cell"] != "top-middle")
    logger.info("Jigsaw complete placement plan: {}", placement_plan)
    normalization_rotation = Rotation(**base_rotation)

    def take_piece(placement: dict[str, Any], which_hand: int) -> None:
        candidate = candidates[placement["object_id"]]
        object_id = candidate["raw_object_id"]
        _require_success("approach candidate", agent.tongsim.move_to_object(agent.character_id, object_id))
        if legacy_rpc:
            take = agent.tongsim.move_and_take_object(
                agent.character_id,
                object_id,
                which_hand=which_hand,
                movable_object_ids=movable_ids,
            )
        else:
            take = agent.tongsim.move_and_take_object(agent.character_id, object_id, which_hand=which_hand)
        _require_success("take candidate", take)

    def place_piece(placement: dict[str, Any], which_hand: int) -> None:
        candidate = candidates[placement["object_id"]]
        object_id = candidate["raw_object_id"]
        target = list(placement["target_location"])
        target_rotation = Rotation(**placement["target_rotation"])
        _require_success(
            "approach destination",
            agent.tongsim.move_to_location(agent.character_id, target, stop_distance=30.0),
        )
        if legacy_rpc:
            put_result = agent.tongsim.put_down_sth(
                agent.character_id,
                target_location=target,
                target_rotation=target_rotation,
                auto_rotate=False,
                force_locate=True,
            )
        else:
            put_result = agent.tongsim.put_down_to_location(
                agent.character_id,
                target_location=target,
                which_hand=which_hand,
                rotation=target_rotation,
                auto_rotate=False,
                force_locate=True,
            )
        _require_success("place candidate", put_result)
        if (
            diagnostics_dir
            and legacy_rpc
            and (not diagnostic_cell or diagnostic_cell == placement["cell"])
        ):
            try:
                placed_perception = agent.tongsim.acquire_first_person_perception(
                    agent.character_id,
                    width=1280,
                    height=720,
                )
                placed_image = placed_perception.get("image")
                if placed_image:
                    snapshot_path = save_perception_snapshot(
                        placed_image,
                        Path(diagnostics_dir)
                        / f"jigsaw_{diagnostic_stamp}_after_{placement['cell']}_{object_id}.png",
                    )
                    logger.info("Saved post-place jigsaw snapshot to {}", snapshot_path)
            except Exception:
                logger.exception("Could not save post-place snapshot for jigsaw piece {}", object_id)
        # The legacy competition stub exposes only the combined perception RPC;
        # it does not implement get_object_basic_info.  Query post-place state
        # directly only on the newer API and take one combined diagnostic
        # snapshot below for the legacy API.
        after = agent.tongsim.get_object_basic_info(object_id) if not legacy_rpc else None
        if isinstance(after, dict):
            location = after.get("place_location") if isinstance(after, dict) else None
            if isinstance(location, dict) and all(axis in location for axis in ("X", "Y", "Z")):
                error = max(
                    abs(float(location[axis]) - target[index])
                    for index, axis in enumerate(("X", "Y", "Z"))
                )
                if error > 3.0:
                    raise RuntimeError(f"jigsaw placement did not reach {placement['cell']}: {location}")
            if diagnostics_dir:
                logger.info(
                    "Jigsaw post-place state: object_id={} cell={} target={} location={} rotation={}",
                    object_id,
                    placement["cell"],
                    target,
                    location,
                    after.get("rotation"),
                )
        logger.info("Placed jigsaw piece {} in {}", object_id, placement["cell"])

    def normalize_board_piece(board_piece: dict[str, Any]) -> None:
        """Re-place an Euler outlier without changing its image cell."""
        object_id = str(board_piece["raw_object_id"])
        location = board_piece.get("place_location") or {}
        target = [float(location[axis]) for axis in ("X", "Y", "Z")]
        _require_success("approach board piece", agent.tongsim.move_to_object(agent.character_id, object_id))
        if legacy_rpc:
            take = agent.tongsim.move_and_take_object(
                agent.character_id,
                object_id,
                which_hand=0,
                movable_object_ids=movable_ids,
            )
        else:
            take = agent.tongsim.move_and_take_object(agent.character_id, object_id, which_hand=0)
        _require_success("take board piece", take)
        _require_success(
            "approach board cell",
            agent.tongsim.move_to_location(agent.character_id, target, stop_distance=30.0),
        )
        if legacy_rpc:
            result = agent.tongsim.put_down_sth(
                agent.character_id,
                target_location=target,
                target_rotation=normalization_rotation,
                auto_rotate=False,
                force_locate=True,
            )
        else:
            result = agent.tongsim.put_down_to_location(
                agent.character_id,
                target_location=target,
                which_hand=0,
                rotation=normalization_rotation,
                auto_rotate=False,
                force_locate=True,
            )
        _require_success("normalize board piece", result)
        logger.info("Normalized jigsaw board piece {} at {}", object_id, target)

    # Newer servers can address a hand while force-locating, so prefetch two
    # pieces at a time. The competition's legacy RPC auto-selects the release
    # hand; keep that path single-handed to avoid swapping two image tiles.
    if legacy_rpc:
        for placement in placement_plan:
            take_piece(placement, which_hand=0)
            place_piece(placement, which_hand=0)
    else:
        for start in range(0, len(placement_plan), 2):
            pair = placement_plan[start : start + 2]
            take_piece(pair[0], which_hand=0)
            if len(pair) == 1:
                place_piece(pair[0], which_hand=0)
                continue
            second = candidates[pair[1]["object_id"]]
            second_take = agent.tongsim.move_and_take_object(
                agent.character_id,
                second["raw_object_id"],
                which_hand=1,
            )
            if action_succeeded(second_take):
                place_piece(pair[0], which_hand=0)
                place_piece(pair[1], which_hand=1)
            else:
                logger.warning("Second-hand jigsaw pickup failed; falling back to hand 0")
                place_piece(pair[0], which_hand=0)
                take_piece(pair[1], which_hand=0)
                place_piece(pair[1], which_hand=0)

    if diagnostics_dir and legacy_rpc:
        try:
            post_perception = agent.tongsim.acquire_first_person_perception(
                agent.character_id,
                width=1280,
                height=720,
            )
            post_objects = {
                str(item.get("object_id")): item
                for item in (post_perception.get("objects") or [])
                if isinstance(item, dict)
            }
            logger.info(
                "Jigsaw legacy post-place states: {}",
                {
                    str(candidate["raw_object_id"]): {
                        "expected_cell": next(
                            placement["cell"]
                            for placement in placement_plan
                            if placement["object_id"] == str(candidate["object_id"])
                        ),
                        "location": post_objects.get(str(candidate["raw_object_id"]), {}).get("place_location"),
                        "rotation": post_objects.get(str(candidate["raw_object_id"]), {}).get("rotation"),
                    }
                    for candidate in layout.candidates
                },
            )
        except Exception:
            # Diagnostics must never turn a valid placement run into a failed run.
            logger.exception("Could not capture legacy post-place jigsaw states")

    # The UE quaternion-to-Euler conversion occasionally represents an already
    # assembled tile as pitch +/-90 instead of the dominant yaw rotation. The
    # bundled evaluator subtracts four points for each such outlier. Correct only
    # outliers, not all six board pieces, to retain most of the time bonus.
    board_outliers = [
        piece
        for piece in layout.placed
        if rotation_distance(piece.get("rotation") or {}, layout.target_rotation) > 20.0
    ]
    logger.info("Jigsaw board rotation outliers to normalize: {}", [p["object_id"] for p in board_outliers])
    for board_piece in board_outliers:
        normalize_board_piece(board_piece)
