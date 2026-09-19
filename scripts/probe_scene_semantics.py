"""房间里的物体，感知到底给了多少语义？有没有"列出全部物体"的接口？

两件事：
  1. 把 unified perception 返回的物体原样打出来，看除了 shape/color/world_aabb
     还有没有 place_location 这类语义字段（有的话，餐桌识别就不必靠猜）。
  2. 再扫一批"列出全场景物体"的候选 RPC —— 如果有，agent 不必靠转圈探索
     就能找到远处的餐桌。

    uv run python scripts/probe_scene_semantics.py
"""

from __future__ import annotations

import contextlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import grpc  # noqa: E402
from google.protobuf import struct_pb2  # noqa: E402

from arenaagent.agent_base import pack_data_to_struct, parse_struct_to_data  # noqa: E402
from arenaagent.tongsim_grpc_client import TongSimGrpcClient  # noqa: E402

ENDPOINT = "127.0.0.1:50060"
SPAWN_LOC = [634.0, -40.0, 0.0]
SPAWN_ROT = [0.0, 0.0, 107.0]

CANDIDATES = [
    "get_all_objects",
    "fetch_scene_objects",
    "get_scene_objects",
    "list_objects",
    "fetch_all_objects",
    "get_all_object_ids",
    "fetch_scene_object_ids",
    "get_objects_in_scene",
    "get_object_list",
    "fetch_object_list",
    "get_scene_entity_ids",
    "query_scene_objects",
]


def rpc(client: TongSimGrpcClient, name: str, payload: dict):
    return client._channel.unary_unary(
        f"/tongsim.service.TongSimService/{name}",
        request_serializer=struct_pb2.Struct.SerializeToString,
        response_deserializer=struct_pb2.Struct.FromString,
    )(pack_data_to_struct(payload), metadata=client._metadata, timeout=30.0)


def box_size(item: dict) -> tuple[float, float, float]:
    aabb = item.get("world_aabb") or {}
    lo, hi = aabb.get("min") or {}, aabb.get("max") or {}
    return tuple(max(float(hi.get(a, 0.0)) - float(lo.get(a, 0.0)), 0.0) for a in ("x", "y", "z"))


def main() -> int:
    client = TongSimGrpcClient(endpoint=ENDPOINT)
    character_id = ""
    try:
        character_id = client.spawn_character(
            "", SPAWN_LOC, SPAWN_ROT, "probe_semantics", 120.0, 1280, 1280
        )
        print(f"character_id={character_id!r}")
        if not character_id:
            return 1
        time.sleep(6.0)

        result = rpc(
            client,
            "acquire_first_person_perception",
            {"character_id": character_id, "width": 1280, "height": 1280},
        )
        objects = [item for item in (result.get("objects") or []) if isinstance(item, dict)]
        print(f"\nobjects={len(objects)}")

        keys: set[str] = set()
        for item in objects:
            keys.update(item)
        print(f"物体字段全集: {sorted(keys)}")

        print("\n=== 每个物体的完整字段（按 id 排序）===")
        for item in sorted(objects, key=lambda i: int(i["object_id"]) if str(i["object_id"]).isdigit() else 999):
            size = tuple(round(v, 1) for v in box_size(item))
            print(f"  id={item['object_id']:>4} size={size}  {json.dumps({k: v for k, v in item.items() if k != 'world_aabb'}, ensure_ascii=False)}")

        print("\n=== 体积最大的 6 件的 AABB ===")
        ranked = sorted(objects, key=lambda i: math.prod(box_size(i)), reverse=True)
        for item in ranked[:6]:
            aabb = item.get("world_aabb") or {}
            lo, hi = aabb.get("min") or {}, aabb.get("max") or {}
            print(
                f"  id={item['object_id']:>4} shape={item.get('shape')} color={item.get('color')} "
                f"z=[{round(float(lo.get('z', 0)), 1)}, {round(float(hi.get('z', 0)), 1)}] "
                f"xy=[{round(float(lo.get('x', 0)), 1)},{round(float(lo.get('y', 0)), 1)}]~"
                f"[{round(float(hi.get('x', 0)), 1)},{round(float(hi.get('y', 0)), 1)}]"
            )

        print("\n=== 全场景物体接口探测 ===")
        for name in CANDIDATES:
            payload = {"character_id": character_id}
            try:
                response = rpc(client, name, payload)
                print(f"  {name:<26} OK -> {json.dumps(response, ensure_ascii=False)[:200]}")
            except grpc.RpcError as exc:
                print(f"  {name:<26} {exc.code().name}: {str(exc.details() or '')[:60]}")
            except Exception as exc:  # noqa: BLE001
                print(f"  {name:<26} {type(exc).__name__}: {str(exc)[:60]}")
    finally:
        if character_id:
            with contextlib.suppress(Exception):
                client.destory_character(character_id)
        with contextlib.suppress(Exception):
            client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
