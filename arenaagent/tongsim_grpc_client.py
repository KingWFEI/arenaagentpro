from __future__ import annotations

import contextlib
import threading
import uuid
from typing import Any

import grpc
from loguru import logger

from arenaagent.agent_base import pack_data_to_struct, parse_struct_to_data
from arenaagent.generated.tongsim.tongsim_service_pb2_grpc import TongSimServiceStub
from arenaagent.tongsim_interface import Rotation, TongSimInterface


_MAX_MSG_BYTES = 50 * 1024 * 1024
_GRPC_OPTIONS = [
    ("grpc.max_send_message_length", _MAX_MSG_BYTES),
    ("grpc.max_receive_message_length", _MAX_MSG_BYTES),
]
_CLIENT_ID_METADATA_KEY = "x-tongsim-client-id"


class TongSimGrpcClient(TongSimInterface):
    """TongSimInterface implementation that proxies all calls to a remote TongSimService."""

    def __init__(self, endpoint: str = "127.0.0.1:50060", heartbeat_interval_secs: float = 2.0) -> None:
        self._channel = grpc.insecure_channel(endpoint, options=_GRPC_OPTIONS)
        self._stub = TongSimServiceStub(self._channel)
        self._endpoint = endpoint
        self._client_id = uuid.uuid4().hex
        self._metadata = ((_CLIENT_ID_METADATA_KEY, self._client_id),)
        self._heartbeat_interval_secs = max(float(heartbeat_interval_secs), 0.5)
        self._heartbeat_stop = threading.Event()
        self._heartbeat_supported = True
        self._heartbeat_compat_logged = False
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"tongsim-heartbeat-{self._client_id[:8]}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _call(self, method_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        rpc = getattr(self._stub, method_name)
        return parse_struct_to_data(rpc(pack_data_to_struct(payload), metadata=self._metadata))

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.is_set():
            try:
                self.heartbeat()
            except grpc.RpcError as exc:
                if exc.code() == grpc.StatusCode.UNIMPLEMENTED:
                    self._heartbeat_supported = False
                    if not self._heartbeat_compat_logged:
                        logger.warning(
                            "TongSim server {} does not implement heartbeat RPC; disabling heartbeat for client {}. "
                            "Disconnect-triggered auto cleanup requires the upgraded tongsim-server.",
                            self._endpoint,
                            self._client_id,
                        )
                        self._heartbeat_compat_logged = True
                    return
                logger.debug("TongSim heartbeat failed for client {}: {}", self._client_id, exc)
            except Exception as exc:  # pragma: no cover - runtime guard
                logger.debug("TongSim heartbeat failed for client {}: {}", self._client_id, exc)

            if self._heartbeat_stop.wait(self._heartbeat_interval_secs):
                return

    def heartbeat(self) -> None:
        if not self._heartbeat_supported:
            return
        self._call("heartbeat", {})

    # ------------------------------------------------------------------ #
    # Character lifecycle
    # ------------------------------------------------------------------ #

    def spawn_character(
        self,
        asset_name,
        loc,
        rot,
        desired_name,
        fov: float = 120.0,
        width: int = 720,
        height: int = 1000,
        camera_name_suffix: str | None = None,
        camera_stream_id: str | None = None,
        spawn_extra_camera: bool = False,
    ) -> str:
        result = self._call(
            "spawn_character",
            {
                "asset_name": asset_name,
                "loc": list(loc),
                "rot": list(rot),
                "desired_name": desired_name,
                "fov": fov,
                "width": width,
                "height": height,
                "camera_name_suffix": camera_name_suffix,
                "camera_stream_id": camera_stream_id,
                "spawn_extra_camera": spawn_extra_camera,
            },
        )
        return result.get("character_id", "")

    def destory_character(self, character_id=None) -> dict:
        return self._call("destory_character", {"character_id": str(character_id) if character_id is not None else ""})

    def close(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=self._heartbeat_interval_secs + 1.0)

        with contextlib.suppress(Exception):
            self._call("close", {})
        self._channel.close()

    # ------------------------------------------------------------------ #
    # Perception
    # ------------------------------------------------------------------ #

    def acquire_first_person_image(self, character_id, encode_base64: bool = True):
        # Always returns base64 str from server; _decode_image() in SemanticMapper handles str.
        result = self._call("acquire_first_person_image", {"character_id": str(character_id)})
        return result.get("image")

    def acquire_first_person_segmantic_image(self, character_id, encode_base64: bool = True):
        result = self._call("acquire_first_person_segmantic_image", {"character_id": str(character_id)})
        return result.get("image")

    def fetch_first_person_visible_objects(self, character_id) -> list:
        result = self._call("fetch_first_person_visible_objects", {"character_id": str(character_id)})
        return result.get("objects", [])

    def get_object_basic_info(self, object_id: str) -> dict[str, Any]:
        return self._call("get_object_basic_info", {"object_id": object_id})

    def get_object_world_aabb(self, object_id: str) -> dict[str, Any]:
        return self._call("get_object_world_aabb", {"object_id": object_id})

    def get_object_id_by_name(self, name: str) -> str | None:
        result = self._call("get_object_id_by_name", {"name": name})
        return result.get("object_id") or None

    def get_object_in_hand(self, character_id) -> tuple[str, int] | None:
        result = self._call("get_object_in_hand", {"character_id": str(character_id)})
        obj_id = result.get("object_id")
        if obj_id is None:
            return None
        return (str(obj_id), int(result.get("hand_idx", 0)))

    def set_object_pose(self, object_id: str, location, rotation: Rotation) -> bool:
        loc = location
        if hasattr(location, "X"):
            loc = {"X": location.X, "Y": location.Y, "Z": location.Z}
        rot = {"roll": rotation.roll, "yaw": rotation.yaw, "pitch": rotation.pitch}
        result = self._call("set_object_pose", {"object_id": object_id, "location": loc, "rotation": rot})
        return bool(result.get("result", False))

    # ------------------------------------------------------------------ #
    # View control
    # ------------------------------------------------------------------ #

    def look_at_location(
        self, character_id, target_location, is_cancel: bool = False, execute_immediately: bool = False
    ):
        return self._call(
            "look_at_location",
            {
                "character_id": str(character_id),
                "target_location": target_location,
                "is_cancel": is_cancel,
                "execute_immediately": execute_immediately,
            },
        )

    def look_at_object(self, character_id, object_id: str, is_cancel: bool = False):
        return self._call(
            "look_at_object",
            {
                "character_id": str(character_id),
                "object_id": object_id,
                "is_cancel": is_cancel,
            },
        )

    def point_at_object(self, character_id, object_id: str, is_cancel: bool = False, which_hand: int = 0):
        return self._call(
            "point_at_object",
            {
                "character_id": str(character_id),
                "object_id": object_id,
                "is_cancel": is_cancel,
                "which_hand": which_hand,
            },
        )

    # ------------------------------------------------------------------ #
    # Movement
    # ------------------------------------------------------------------ #

    def move_to_location(self, character_id, target_location, stop_distance: float = 0.5):
        return self._call(
            "move_to_location",
            {
                "character_id": str(character_id),
                "target_location": target_location,
                "stop_distance": stop_distance,
            },
        )

    def move_forward(self, character_id, distance: float):
        return self._call(
            "move_forward",
            {
                "character_id": str(character_id),
                "distance": float(distance),
            },
        )

    def move_to_object(self, character_id, object_id: str):
        return self._call(
            "move_to_object",
            {
                "character_id": str(character_id),
                "object_id": object_id,
            },
        )

    def move_and_take_object(self, character_id, object_id: str, which_hand: int = 0):
        return self._call(
            "move_and_take_object",
            {
                "character_id": str(character_id),
                "object_id": object_id,
                "which_hand": which_hand,
            },
        )

    def turn_in_degree(self, character_id, degree):
        return self._call(
            "turn_in_degree",
            {
                "character_id": str(character_id),
                "degree": float(degree),
            },
        )

    # ------------------------------------------------------------------ #
    # Object manipulation
    # ------------------------------------------------------------------ #

    def put_down_to_location(
        self,
        character_id,
        target_location,
        which_hand: int = 0,
        disable_physics: bool = False,
        hold_if_unreachable: bool = False,
        force_release: bool = True,
        auto_rotate: bool = False,
        rotation: Rotation | None = None,
        force_locate: bool = False,
    ):
        rot_dict = None
        if rotation is not None:
            rot_dict = {"roll": rotation.roll, "yaw": rotation.yaw, "pitch": rotation.pitch}
        return self._call(
            "put_down_to_location",
            {
                "character_id": str(character_id),
                "target_location": target_location,
                "which_hand": which_hand,
                "disable_physics": disable_physics,
                "hold_if_unreachable": hold_if_unreachable,
                "force_release": force_release,
                "auto_rotate": auto_rotate,
                "rotation": rot_dict,
                "force_locate": force_locate,
            },
        )

    def pour_water(self, character_id, object_id: str, location, which_hand: int = 0):
        return self._call(
            "pour_water",
            {
                "character_id": str(character_id),
                "object_id": object_id,
                "location": location,
                "which_hand": which_hand,
            },
        )

    def slice_food(self, character_id, object_id: str, location):
        return self._call(
            "slice_food",
            {
                "character_id": str(character_id),
                "object_id": object_id,
                "location": location,
            },
        )

    def wash_hands(self, character_id, faucet_object_id: str):
        return self._call(
            "wash_hands",
            {
                "character_id": str(character_id),
                "faucet_object_id": faucet_object_id,
            },
        )

    def wash_object_in_hand(self, character_id, faucet_object_id: str):
        return self._call(
            "wash_object_in_hand",
            {
                "character_id": str(character_id),
                "faucet_object_id": faucet_object_id,
            },
        )

    def move_and_put_down_object_in_container(self, character_id, which_hand: int = 0):
        return self._call(
            "move_and_put_down_object_in_container",
            {
                "character_id": str(character_id),
                "which_hand": which_hand,
            },
        )

    def move_and_put_down(
        self,
        character_id,
        move_target_location,
        put_target_location,
        which_hand: int = 0,
        put_rotation: Rotation | None = None,
    ):
        rot_dict = None
        if put_rotation is not None:
            rot_dict = {"roll": put_rotation.roll, "yaw": put_rotation.yaw, "pitch": put_rotation.pitch}
        return self._call(
            "move_and_put_down",
            {
                "character_id": str(character_id),
                "move_target_location": move_target_location,
                "put_target_location": put_target_location,
                "which_hand": which_hand,
                "put_rotation": rot_dict,
            },
        )

    # ------------------------------------------------------------------ #
    # Interaction
    # ------------------------------------------------------------------ #
    def sit_down_to_object(self, character_id, object_id: str):
        return self._call(
            "sit_down_to_object",
            {
                "character_id": str(character_id),
                "object_id": object_id,
            },
        )

    def mop_floor(self, character_id, dirt_id: str):
        return self._call(
            "mop_floor",
            {
                "character_id": str(character_id),
                "dirt_id": dirt_id,
            },
        )

    def rest(self, character_id):
        return self._call("rest", {"character_id": str(character_id)})

    def speak_to_npc(self, character_id, target: str, content: str):
        return self._call(
            "speak_to_npc",
            {
                "character_id": str(character_id),
                "target": target,
                "content": content,
            },
        )
