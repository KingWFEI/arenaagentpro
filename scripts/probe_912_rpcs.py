"""Sweep every candidate TongSim RPC against the running server and report which exist.

The vendored proto disagrees with the 912 server in both directions, so proto
declarations cannot be trusted. UNIMPLEMENTED is returned before argument
validation, so a bogus-but-well-formed payload still reveals whether a method is
registered.

    uv run python scripts/probe_912_rpcs.py
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import grpc  # noqa: E402
from google.protobuf import struct_pb2  # noqa: E402

from arenaagent.agent_base import pack_data_to_struct  # noqa: E402
from arenaagent.tongsim_grpc_client import TongSimGrpcClient  # noqa: E402

ENDPOINT = "127.0.0.1:50060"

# name -> extra payload keys merged with character_id
CANDIDATES: dict[str, dict] = {
    # --- 生成的 proto 里有 ---
    "acquire_first_person_image": {},
    "acquire_first_person_segmantic_image": {},
    "acquire_first_person_perception": {"width": 1280, "height": 720},
    "fetch_first_person_visible_objects": {},
    "get_object_basic_info": {"object_id": "4"},
    "get_object_world_aabb": {"object_id": "4"},
    "get_object_id_by_name": {"name": "Bed"},
    "get_object_in_hand": {},
    "look_at_location": {"target_location": {"X": 0.0, "Y": 0.0, "Z": 0.0}},
    "look_at_object": {"object_id": "4"},
    "point_at_object": {"object_id": "4"},
    "move_to_location": {"target_location": {"X": 0.0, "Y": 0.0, "Z": 0.0}, "stop_distance": 30.0},
    "move_forward": {"distance": 0.0},
    "move_to_object": {"object_id": "4"},
    "move_and_take_object": {"object_id": "4", "which_hand": 0},
    "turn_in_degree": {"degree": 0.0},
    "put_down_to_location": {"target_location": {"X": 0.0, "Y": 0.0, "Z": 0.0}},
    "pour_water": {"object_id": "4", "location": {"X": 0.0, "Y": 0.0, "Z": 0.0}},
    "slice_food": {"object_id": "4", "location": {"X": 0.0, "Y": 0.0, "Z": 0.0}},
    "wash_hands": {"faucet_object_id": "4"},
    "wash_object_in_hand": {"faucet_object_id": "4"},
    "sit_down_to_object": {"object_id": "4"},
    "mop_floor": {"dirt_id": "4"},
    "rest": {},
    "speak_to_npc": {"target": "x", "content": "x"},
    "move_and_put_down_object_in_container": {"which_hand": 0},
    "move_and_put_down": {"move_target_location": {}, "put_target_location": {}},
    "set_object_pose": {"object_id": "4", "location": {}, "rotation": {}},
    "open_door": {"object_id": "4"},
    "close_door": {"object_id": "4"},
    "interact": {"object_id": "4"},
    "heartbeat": {},
    # --- 生成的 proto 里没有，但可能存在于 912 ---
    "has_object_in_hand": {},
    "move_to_npc": {"npc_name": "x"},
    "transfer_puzzle_piece": {"piece_object_id": "4"},
    "move_and_take_puzzle_piece": {"piece_object_id": "4"},
    "put_down_sth": {"target_location": {"X": 0.0, "Y": 0.0, "Z": 0.0}},
    "set_pickup_whitelist": {"object_ids": ["4"]},
    "get_character_location": {},
    "get_character_pose": {},
    "get_all_objects": {},
    "get_object_rotation": {"object_id": "4"},
    "reset_scene": {},
}


def main() -> int:
    client = TongSimGrpcClient(endpoint=ENDPOINT)
    character_id = ""
    results: dict[str, str] = {}
    try:
        with contextlib.suppress(Exception):
            character_id = client.spawn_character(
                "", [282.0, -353.0, 20.0], [0.0, 0.0, -90.0], "probe_rpc_sweep", 120.0, 720, 720
            )
        print(f"character_id={character_id!r}")

        for name, extra in CANDIDATES.items():
            payload = dict(extra)
            if character_id:
                payload["character_id"] = str(character_id)
            rpc = client._channel.unary_unary(
                f"/tongsim.service.TongSimService/{name}",
                request_serializer=struct_pb2.Struct.SerializeToString,
                response_deserializer=struct_pb2.Struct.FromString,
            )
            try:
                rpc(pack_data_to_struct(payload), metadata=client._metadata, timeout=15.0)
                results[name] = "OK"
            except grpc.RpcError as exc:
                code = exc.code()
                if code == grpc.StatusCode.UNIMPLEMENTED:
                    results[name] = "UNIMPLEMENTED"
                else:
                    results[name] = f"{code.name}: {str(exc.details() or '')[:70]}"
            except Exception as exc:  # noqa: BLE001
                results[name] = f"{type(exc).__name__}: {str(exc)[:70]}"
    finally:
        if character_id:
            with contextlib.suppress(Exception):
                client.destory_character(character_id)
        with contextlib.suppress(Exception):
            client.close()

    def dump(title: str, names: list[str]) -> None:
        print(f"\n=== {title} ({len(names)}) ===")
        for name in names:
            print(f"  {name:<42} {results[name]}")

    names = list(results)
    dump("已实现（存在）", [n for n in names if results[n] == "OK"])
    dump(
        "已实现但参数被拒（同样说明存在）",
        [n for n in names if results[n] != "OK" and results[n] != "UNIMPLEMENTED"],
    )
    dump("不存在（UNIMPLEMENTED）", [n for n in names if results[n] == "UNIMPLEMENTED"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
