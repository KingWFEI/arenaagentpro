"""Does put_down_sth place an object at a distance, or does the character have to walk there?

Tidy-room currently walks to the destination before placing. If force_locate really
ignores distance, that walk can be dropped entirely.

    uv run python scripts/probe_remote_placement.py

Pick up a small prop, then ask the server to drop it on a piece of furniture the
character is NOT standing next to, and check where it actually ended up.
"""

from __future__ import annotations

import contextlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.protobuf import struct_pb2  # noqa: E402

from arenaagent.agent_base import pack_data_to_struct, parse_struct_to_data  # noqa: E402
from arenaagent.tongsim_grpc_client import TongSimGrpcClient  # noqa: E402

ENDPOINT = "127.0.0.1:50060"
SPAWN_LOC = [634.0, -40.0, 0.0]
SPAWN_ROT = [0.0, 0.0, 107.0]


def call(client: TongSimGrpcClient, name: str, payload: dict) -> dict:
    rpc = client._channel.unary_unary(
        f"/tongsim.service.TongSimService/{name}",
        request_serializer=struct_pb2.Struct.SerializeToString,
        response_deserializer=struct_pb2.Struct.FromString,
    )
    return parse_struct_to_data(rpc(pack_data_to_struct(payload), metadata=client._metadata, timeout=120.0))


def box_size(item: dict) -> tuple[float, float, float]:
    aabb = item.get("world_aabb") or {}
    lo, hi = aabb.get("min") or {}, aabb.get("max") or {}
    return tuple(max(float(hi.get(a, 0.0)) - float(lo.get(a, 0.0)), 0.0) for a in ("x", "y", "z"))


def centre(item: dict) -> list[float]:
    aabb = item.get("world_aabb") or {}
    lo, hi = aabb.get("min") or {}, aabb.get("max") or {}
    return [round((float(lo.get(a, 0.0)) + float(hi.get(a, 0.0))) / 2, 1) for a in ("x", "y", "z")]


def perceive(client: TongSimGrpcClient, character_id: str) -> dict[str, dict]:
    result = call(
        client,
        "acquire_first_person_perception",
        {"character_id": character_id, "width": 1280, "height": 1280},
    )
    return {
        str(item["object_id"]): item
        for item in (result.get("objects") or [])
        if isinstance(item, dict) and item.get("object_id") is not None
    }


def main() -> int:
    client = TongSimGrpcClient(endpoint=ENDPOINT)
    character_id = ""
    try:
        character_id = client.spawn_character(
            "", SPAWN_LOC, SPAWN_ROT, "probe_remote_place", 120.0, 1280, 1280
        )
        print(f"character_id={character_id!r}")
        if not character_id:
            return 1

        objects: dict[str, dict] = {}
        for attempt in range(8):
            time.sleep(3.0)
            objects = perceive(client, character_id)
            print(f"  采集 {attempt + 1}: objects={len(objects)}")
            if len(objects) > 3:
                break
        else:
            if not objects:
                print(
                    "\n!! 八次采集都是空的：客户端进程还在但已不渲染场景，"
                    "这是需要重启仿真客户端的故障状态，不是逻辑问题。"
                )

        # 可抓取的小道具：体积小、有真实包围盒、形状已知
        movables = [
            item
            for item in objects.values()
            if item.get("shape") != "Unknown" and 0 < math.prod(box_size(item)) < 200_000
        ]
        if not movables:
            print("视野里没有可尝试抓取的小物体，换个视角或直接重跑")
            return 1
        prop = min(movables, key=lambda item: math.prod(box_size(item)))
        prop_id = str(prop["object_id"])
        print(f"\n选中小道具 id={prop_id} shape={prop.get('shape')} 尺寸={box_size(prop)}")

        # 目标落点：一件明显不在身边的大型家具的顶面中心
        furnitures = [
            item
            for item in objects.values()
            if item.get("shape") != "Unknown" and math.prod(box_size(item)) > 300_000
        ]
        if not furnitures:
            print("视野里没有可用的大家具，无法构造远距离落点")
            return 1
        furniture = max(furnitures, key=lambda item: math.prod(box_size(item)))
        cx, cy, cz = centre(furniture)
        target = {"X": cx, "Y": cy, "Z": round(cz, 1)}
        print(f"目标家具 id={furniture['object_id']} shape={furniture.get('shape')} 落点={target}")

        print("\n=== 抓起道具 ===")
        print(" move_to_object ->", call(client, "move_to_object", {"character_id": character_id, "object_id": prop_id}))
        time.sleep(2.0)
        print(
            " move_and_take_object ->",
            call(client, "move_and_take_object", {"character_id": character_id, "object_id": prop_id, "which_hand": 0}),
        )
        time.sleep(2.0)
        hand = call(client, "has_object_in_hand", {"character_id": character_id})
        print(f" has_object_in_hand -> {json.dumps(hand, ensure_ascii=False)}")
        if not hand.get("has_object"):
            print("!! 没抓起来，后面的结论不成立")
            return 1

        # 把镜头对准目标家具，这样放完之后能直接在同一视野里找到道具。
        print("\n=== 让目标家具进入视野 ===")
        call(client, "look_at_location", {"character_id": character_id, "target_location": target})
        time.sleep(2.0)
        before = perceive(client, character_id)
        print(f" 目标家具 {furniture['object_id']} 可见: {furniture['object_id'] in before}")

        print("\n=== 不移动，直接请求放到远处 ===")
        payload = {
            "character_id": character_id,
            "target_location": target,
            "target_rotation": None,
            "auto_rotate": True,
            "force_locate": True,
        }
        print(" put_down_sth ->", call(client, "put_down_sth", payload))
        time.sleep(3.0)
        hand_after = call(client, "has_object_in_hand", {"character_id": character_id})
        print(f" 之后 has_object_in_hand -> {json.dumps(hand_after, ensure_ascii=False)}")

        print("\n=== 道具现在在哪 ===")
        time.sleep(2.0)
        after = perceive(client, character_id)
        moved = after.get(prop_id)
        if moved is None:
            print(f" 道具 {prop_id} 不在视野里")
            # 转一圈再找一次
            for _ in range(8):
                call(client, "turn_in_degree", {"character_id": character_id, "degree": 45.0})
                time.sleep(1.0)
                moved = perceive(client, character_id).get(prop_id)
                if moved is not None:
                    print(" 转一圈后找到了")
                    break
        if moved is None:
            print(" 转完整圈仍未找到该道具")
        else:
            distance = math.dist(centre(moved), [target["X"], target["Y"], target["Z"]])
            print(f" 道具 {prop_id} 位置={centre(moved)}  距目标 {distance:.1f}")
        print(f" 目标家具 {furniture['object_id']} 中心={[cx, cy, cz]}")
        print("\n判定：道具位置 ≈ 目标落点，且角色全程只转身没移动 → 隔空放置成立")
    finally:
        if character_id:
            with contextlib.suppress(Exception):
                client.destory_character(character_id)
        with contextlib.suppress(Exception):
            client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
