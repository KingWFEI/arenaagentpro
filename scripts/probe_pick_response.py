"""Probe what the 912 pickup/placement RPCs actually return.

get_object_in_hand is gone on 912 and has_object_in_hand only reports a boolean,
so if any action response carries the grabbed object ID, tidy-room can use that
instead of inferring it.

    uv run python scripts/probe_pick_response.py
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.protobuf import struct_pb2  # noqa: E402

from arenaagent.agent_base import pack_data_to_struct, parse_struct_to_data  # noqa: E402
from arenaagent.tongsim_grpc_client import TongSimGrpcClient  # noqa: E402

ENDPOINT = "127.0.0.1:50060"
SPAWN_LOC = [282.0, -353.0, 20.0]


def call(client: TongSimGrpcClient, name: str, payload: dict) -> dict:
    rpc = client._channel.unary_unary(
        f"/tongsim.service.TongSimService/{name}",
        request_serializer=struct_pb2.Struct.SerializeToString,
        response_deserializer=struct_pb2.Struct.FromString,
    )
    return parse_struct_to_data(rpc(pack_data_to_struct(payload), metadata=client._metadata, timeout=60.0))


def wait_for_scene(client: TongSimGrpcClient, character_id: str, attempts: int = 8) -> dict:
    """Poll until the camera renders; the first frames after spawn come back empty."""
    perception: dict = {}
    for attempt in range(attempts):
        time.sleep(3.0)
        perception = call(
            client,
            "acquire_first_person_perception",
            {"character_id": character_id, "width": 1280, "height": 720},
        )
        objects = perception.get("objects") or []
        print(f"  第 {attempt + 1} 次采集: objects={len(objects)} error={perception.get('error')!r}")
        if objects:
            return perception
    return perception


def main() -> int:
    client = TongSimGrpcClient(endpoint=ENDPOINT)
    character_id = ""
    try:
        character_id = client.spawn_character(
            "", SPAWN_LOC, [0.0, 0.0, -90.0], "probe_pick_response", 120.0, 1280, 720
        )
        print(f"character_id={character_id!r}\n")
        if not character_id:
            return 1

        perception = wait_for_scene(client, character_id)
        objects = perception.get("objects", [])
        print(f"可见物体 {len(objects)} 个:")
        for item in objects:
            aabb = item.get("world_aabb") or {}
            lo, hi = aabb.get("min") or {}, aabb.get("max") or {}
            size = tuple(
                round(float(hi.get(a, 0.0)) - float(lo.get(a, 0.0)), 1) for a in ("x", "y", "z")
            )
            print(
                f"  id={str(item.get('object_id')):<4} shape={str(item.get('shape')):<12} "
                f"aabb尺寸(xyz)={size} color={item.get('color')}"
            )

        # 挑一个尺寸最小的可抓取道具；shape=Unknown 且 AABB 为零的是场景几何。
        def volume(item: dict) -> float:
            aabb = item.get("world_aabb") or {}
            lo, hi = aabb.get("min") or {}, aabb.get("max") or {}
            dims = [max(float(hi.get(a, 0.0)) - float(lo.get(a, 0.0)), 0.0) for a in ("x", "y", "z")]
            return dims[0] * dims[1] * dims[2]

        movable = [
            o
            for o in objects
            if o.get("object_id") is not None and volume(o) > 0 and o.get("shape") != "Unknown"
        ]
        if not movable:
            print("\n视野里没有可尝试的对象")
            return 0
        target = min(movable, key=volume)
        target_id = str(target["object_id"])
        print(f"\n选中最小物体 id={target_id} shape={target.get('shape')}")

        for name, payload in (
            ("move_to_object", {"character_id": character_id, "object_id": target_id}),
            ("move_and_take_object", {"character_id": character_id, "object_id": target_id, "which_hand": 0}),
        ):
            print(f"\n=== {name} ===")
            try:
                print(json.dumps(call(client, name, payload), ensure_ascii=False, indent=2)[:1200])
            except Exception as exc:  # noqa: BLE001
                print(f"  失败: {type(exc).__name__}: {str(exc)[:200]}")

        for name, payload in (
            ("has_object_in_hand", {"character_id": character_id}),
            ("acquire_first_person_perception", {"character_id": character_id, "width": 1280, "height": 720}),
        ):
            print(f"\n=== {name} ===")
            try:
                result = call(client, name, payload)
                if name == "acquire_first_person_perception":
                    result = {
                        "image": f"<base64 len={len(result.get('image') or '')}>",
                        "objects": [
                            {k: v for k, v in o.items() if k != "world_aabb"} for o in result.get("objects", [])
                        ],
                    }
                print(json.dumps(result, ensure_ascii=False, indent=2)[:1500])
            except Exception as exc:  # noqa: BLE001
                print(f"  失败: {type(exc).__name__}: {str(exc)[:200]}")
    finally:
        if character_id:
            with contextlib.suppress(Exception):
                client.destory_character(character_id)
        with contextlib.suppress(Exception):
            client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
