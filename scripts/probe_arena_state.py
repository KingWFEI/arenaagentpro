"""问一下 arena 现在处于什么状态：能不能接新 agent、这一轮是否还在跑。

用来区分两种"agent 连上去没反应"：
- arena 还在等 agent（is_ready_for_agent=true，只是没连上）
- 这一轮 10 个 subject 已经全部结算完（task-flow 结束，必须重启 arena_offline）

    uv run python scripts/probe_arena_state.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import grpc  # noqa: E402
from google.protobuf import struct_pb2  # noqa: E402

from arenaagent.agent_base import pack_data_to_struct  # noqa: E402
from arenaagent.generated.arena.message import basic_type_pb2  # noqa: E402

TARGET = "127.0.0.1:50051"
PROBE_ID = "arena-state-probe"

CALLS = (
    ("is_ready_for_agent", basic_type_pb2.Bool.FromString, {}),
    ("get_num_subjects", basic_type_pb2.Int32.FromString, {}),
    ("get_current_subject_index", basic_type_pb2.Int32.FromString, {}),
)


def main() -> int:
    channel = grpc.insecure_channel(
        TARGET,
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ],
    )
    for method, deserializer, extra in CALLS:
        payload = {"agent_id": PROBE_ID, **extra}
        request = pack_data_to_struct(payload)
        rpc = channel.unary_unary(
            f"/arena.agent.TongTestAgentService/{method}",
            request_serializer=request.SerializeToString,
            response_deserializer=deserializer,
        )
        try:
            response = rpc(request, timeout=5)
        except grpc.RpcError as exc:
            print(f"{method:<26} 调用失败: {exc.code()} {exc.details()}")
            continue
        value = getattr(response, "value", None)
        print(f"{method:<26} {value}  ({struct_pb2.Struct})")
    channel.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
