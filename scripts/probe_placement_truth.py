"""隔空放置到底是"没放成"还是"读到了旧坐标"？一次问清楚。

背景：tidyroom 实跑日志里，同一个物体多次尝试放置后，校验读到的
actual_center 每次完全一致（比如 target=34 两次都是 517.0/234.0/10.34）。
两种解释：
  A. 放置其实成功了，但校验读到的是物体被拿起之前的缓存 AABB；
  B. 放置被静默拒绝，物体根本没动。
区分方法：放完之后立刻查 has_object_in_hand —— 手空了说明物体确实被释放；
再朝落点看一眼重新感知，看物体到底在哪。

顺带验证：物体被拿在手里时，感知报的 AABB 是不是拿起前的旧值。

    uv run python scripts/probe_placement_truth.py
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


def hand(client: TongSimGrpcClient, character_id: str) -> str:
    with contextlib.suppress(Exception):
        return json.dumps(
            call(client, "has_object_in_hand", {"character_id": character_id}), ensure_ascii=False
        )
    return "?"


def drop(client: TongSimGrpcClient, character_id: str, point: dict) -> dict:
    payload = {
        "character_id": character_id,
        "target_location": point,
        "target_rotation": None,
        "auto_rotate": True,
        "force_locate": True,
    }
    return call(client, "put_down_sth", payload)


def locate(client: TongSimGrpcClient, character_id: str, prop_id: str, point: dict) -> dict | None:
    """朝落点看一眼，再在全新帧里读物体位置。"""
    call(client, "look_at_location", {"character_id": character_id, "target_location": point})
    time.sleep(2.5)
    return perceive(client, character_id).get(prop_id)


def main() -> int:
    client = TongSimGrpcClient(endpoint=ENDPOINT)
    character_id = ""
    try:
        character_id = client.spawn_character(
            "", SPAWN_LOC, SPAWN_ROT, "probe_place_truth", 120.0, 1280, 1280
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

        print("\n=== 场景里体积最大的 12 件（识别沙发/餐桌/垃圾桶）===")
        ranked = sorted(objects.values(), key=lambda i: math.prod(box_size(i)), reverse=True)
        for item in ranked[:12]:
            print(
                f"  id={item['object_id']:>4} shape={str(item.get('shape')):<12} "
                f"color={str(item.get('color')):<10} center={centre(item)} size={tuple(round(v, 1) for v in box_size(item))}"
            )

        movables = [
            item
            for item in objects.values()
            if item.get("shape") != "Unknown" and 0 < math.prod(box_size(item)) < 200_000
        ]
        if not movables:
            print("没有可抓的小物体")
            return 1
        prop = min(movables, key=lambda item: math.prod(box_size(item)))
        prop_id = str(prop["object_id"])
        before = prop
        print(f"\n选中小道具 id={prop_id} shape={prop.get('shape')} 拿起前 center={centre(prop)}")

        print("\n=== 抓起 ===")
        print(" move_to_object ->", call(client, "move_to_object", {"character_id": character_id, "object_id": prop_id}))
        time.sleep(2.0)
        print(
            " move_and_take_object ->",
            call(client, "move_and_take_object", {"character_id": character_id, "object_id": prop_id, "which_hand": 0}),
        )
        time.sleep(2.0)
        print(f" has_object_in_hand -> {hand(client, character_id)}")

        print("\n=== 关键问题：拿在手里时，感知报的是旧坐标吗？===")
        held = perceive(client, character_id).get(prop_id)
        if held is None:
            print("  道具不在视野里（无法判断）")
        else:
            print(f"  感知里 id={prop_id} shape={held.get('shape')} center={centre(held)}")
            print(f"  拿起前的位置           center={centre(before)}")
            print(f"  → {'坐标没跟着手走，是拿起前的旧值' if centre(held) == centre(before) else '坐标跟着手更新了'}")

        furnitures = [
            item
            for item in objects.values()
            if item.get("shape") != "Unknown" and math.prod(box_size(item)) > 300_000
        ]
        if not furnitures:
            print("没有可用大家具")
            return 1
        furniture = max(furnitures, key=lambda item: math.prod(box_size(item)))
        cx, cy, cz = centre(furniture)
        target = {"X": cx, "Y": cy, "Z": round(float((furniture.get("world_aabb") or {}).get("max", {}).get("z", cz)), 1) + 2.0}
        print(
            f"\n=== 落点：家具 id={furniture['object_id']} shape={furniture.get('shape')} "
            f"center={[cx, cy, cz]} 落点={target} ==="
        )
        with contextlib.suppress(Exception):
            position = call(client, "get_character_position", {"character_id": character_id})
            print(f" 角色在 {json.dumps(position, ensure_ascii=False)}（不移动）")

        print("\n put_down_sth ->", drop(client, character_id, target))
        time.sleep(3.0)
        print(f" 之后 has_object_in_hand -> {hand(client, character_id)}")

        after = locate(client, character_id, prop_id, target)
        if after is None:
            print(f" 道具 {prop_id} 在新视野里找不到")
        else:
            distance = math.dist(centre(after), [target["X"], target["Y"], target["Z"]])
            print(f" 道具 {prop_id} 现在 center={centre(after)} 距落点 {distance:.1f}")
            print(f" → {'放置成立' if distance < 40 else '物体没到落点'}")

        print("\n=== 再测一次：放进垃圾桶类容器（落点取容器内部中心）===")
        bins = [
            item
            for item in objects.values()
            if 3_000 < math.prod(box_size(item)) < 60_000
            and box_size(item)[2] >= 25.0
            and max(box_size(item)[0], box_size(item)[1]) < 60.0
        ]
        if not bins:
            print(" 视野里没有更像垃圾桶的容器，跳过")
        else:
            container = bins[0]
            lo = (container.get("world_aabb") or {}).get("min", {})
            hi = (container.get("world_aabb") or {}).get("max", {})
            inner = {
                "X": (float(lo.get("x", 0)) + float(hi.get("x", 0))) / 2,
                "Y": (float(lo.get("y", 0)) + float(hi.get("y", 0))) / 2,
                "Z": float(lo.get("z", 0)) + 2.0,
            }
            print(f" 容器 id={container['object_id']} size={tuple(round(v,1) for v in box_size(container))} 落点={inner}")
            print(" 先把它抓回来 ->", call(client, "move_and_take_object", {"character_id": character_id, "object_id": prop_id, "which_hand": 0}))
            time.sleep(2.0)
            print(f" has_object_in_hand -> {hand(client, character_id)}")
            print(" put_down_sth ->", drop(client, character_id, inner))
            time.sleep(3.0)
            print(f" 之后 has_object_in_hand -> {hand(client, character_id)}")
            inside = locate(client, character_id, prop_id, inner)
            if inside is None:
                print(f" 道具 {prop_id} 不在视野里")
            else:
                distance = math.dist(centre(inside), [inner["X"], inner["Y"], inner["Z"]])
                print(f" 道具 {prop_id} 现在 center={centre(inside)} 距容器中心 {distance:.1f}")
    finally:
        if character_id:
            with contextlib.suppress(Exception):
                client.destory_character(character_id)
        with contextlib.suppress(Exception):
            client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
