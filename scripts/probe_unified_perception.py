"""One-off probe: dump the raw shape of the 912 server's unified perception RPC.

Run against a live UE client + tongsim_server + task system:

    uv run python scripts/probe_unified_perception.py

Prints the exact response structure so semantic_mapper can be adapted without guessing.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.protobuf import struct_pb2  # noqa: E402

from arenaagent.agent_base import pack_data_to_struct, parse_struct_to_data  # noqa: E402
from arenaagent.tongsim_grpc_client import TongSimGrpcClient  # noqa: E402


ENDPOINT = "127.0.0.1:50060"
SPAWN_LOC = [282.0, -353.0, 20.0]
SPAWN_ROT = [0.0, 0.0, -90.0]
DESIRED_NAME = "probe_unified_perception"
WIDTH = 1280
HEIGHT = 720


def shorten(value: object, limit: int = 160) -> object:
    if isinstance(value, str) and len(value) > limit:
        return f"<str len={len(value)} head={value[:40]!r}>"
    if isinstance(value, dict):
        return {k: shorten(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [shorten(v, limit) for v in value[:5]] + ([f"...(+{len(value) - 5})"] if len(value) > 5 else [])
    return value


def main() -> int:
    client = TongSimGrpcClient(endpoint=ENDPOINT)
    character_id = ""
    try:
        character_id = client.spawn_character(
            "", SPAWN_LOC, SPAWN_ROT, DESIRED_NAME, 120.0, 1000, 1000
        )
        print(f"spawn_character -> character_id={character_id!r}")
        if not character_id:
            print("!! spawn failed; is the UE client and 912 tongsim_server running?")
            return 1

        channel = client._channel
        rpc = channel.unary_unary(
            "/tongsim.service.TongSimService/acquire_first_person_perception",
            request_serializer=struct_pb2.Struct.SerializeToString,
            response_deserializer=struct_pb2.Struct.FromString,
        )
        response = rpc(
            pack_data_to_struct(
                {"character_id": str(character_id), "width": WIDTH, "height": HEIGHT}
            ),
            metadata=client._metadata,
        )
        perception = parse_struct_to_data(response)

        print("\n=== 顶层键 ===")
        for key, value in perception.items():
            kind = type(value).__name__
            extra = f" len={len(value)}" if isinstance(value, (str, list, dict)) else ""
            print(f"  {key:<24} {kind}{extra}")

        objects = perception.get("objects")
        print(f"\n=== objects（共 {len(objects) if isinstance(objects, list) else 'N/A'} 个）===")
        if isinstance(objects, list) and objects:
            for index, item in enumerate(objects[:3]):
                print(f"\n  --- objects[{index}] ---")
                print(json.dumps(shorten(item, 400), ensure_ascii=False, indent=4))
                if isinstance(item, dict):
                    print(f"  >>> 字段: {sorted(item)}")
        else:
            print("  (空或不存在)")

        print("\n=== 完整原始响应（图片截断）===")
        print(json.dumps(shorten(perception), ensure_ascii=False, indent=2)[:4000])

        image_b64 = perception.get("image")
        if isinstance(image_b64, str) and image_b64:
            import base64
            import io

            from PIL import Image

            raw = base64.b64decode(image_b64.split(",", 1)[-1])
            photo = Image.open(io.BytesIO(raw))
            out_path = Path.cwd() / "logs" / "probe_unified_perception.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            photo.save(out_path, format="JPEG", quality=92)
            print(f"\n=== 图片 ===")
            print(f"  解码尺寸: {photo.size[0]}x{photo.size[1]}  像素格式={photo.mode}  字节={len(raw)}")
            print(f"  已保存到: {out_path}")

        print("\n=== has_object_in_hand（proto 未声明，手动拼路径）===")
        try:
            rpc_hand = channel.unary_unary(
                "/tongsim.service.TongSimService/has_object_in_hand",
                request_serializer=struct_pb2.Struct.SerializeToString,
                response_deserializer=struct_pb2.Struct.FromString,
            )
            hand = parse_struct_to_data(
                rpc_hand(
                    pack_data_to_struct({"character_id": str(character_id)}),
                    metadata=client._metadata,
                )
            )
            print(f"  {json.dumps(hand, ensure_ascii=False)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  调用失败: {type(exc).__name__}: {str(exc)[:160]}")

        print("\n=== get_object_in_hand（旧接口，预期 UNIMPLEMENTED）===")
        try:
            print(f"  {client._call('get_object_in_hand', {'character_id': str(character_id)})}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {type(exc).__name__}: {str(exc)[:120]}")
    finally:
        if character_id:
            with contextlib.suppress(Exception):
                client.destory_character(character_id)
        with contextlib.suppress(Exception):
            client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
