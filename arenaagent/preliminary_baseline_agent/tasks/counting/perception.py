from __future__ import annotations

from typing import Any

import grpc
from loguru import logger


def acquire_counting_perception(
    agent: Any,
    kwargs: dict[str, Any],
) -> tuple[str | None, list[dict[str, Any]], list[dict[str, Any]]] | None:
    """ZRQ's atomic image/object observation; return None for an older server."""
    if not hasattr(agent.tongsim, "acquire_first_person_perception"):
        return None
    try:
        width = max(int(kwargs.get("width", getattr(agent.cfg, "counting_perception_width", 1280))), 1)
        height = max(int(kwargs.get("height", getattr(agent.cfg, "counting_perception_height", 720))), 1)
        perception = agent.tongsim.acquire_first_person_perception(agent.character_id, width=width, height=height)
        objects = [dict(item) for item in perception.get("objects", []) if isinstance(item, dict)]
        image = perception.get("image") if kwargs.get("include_images", True) else None
        visible = [
            {"object_id": str(item.get("object_id")), "source_object_id": str(item.get("object_id"))}
            for item in objects
            if item.get("object_id") is not None
        ]
        agent.semantic_mapper.last_perception_diagnostics = {}
        logger.debug(
            "Counting unified perception image_size={} visible_objects={}",
            len(image) if isinstance(image, str) else 0,
            len(objects),
        )
        return image, visible, objects
    except grpc.RpcError as exc:
        if exc.code() != grpc.StatusCode.UNIMPLEMENTED:
            raise
        logger.warning("TongSim server lacks unified perception RPC; falling back to legacy split perception")
        return None
