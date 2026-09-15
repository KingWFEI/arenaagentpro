import hashlib
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable

import grpc
from google.protobuf import struct_pb2
from google.protobuf.json_format import MessageToDict, ParseDict

from arenaagent.generated.arena.agent.arena_agent_service_pb2_grpc import (
    TongTestAgentServiceStub,
)
from arenaagent.generated.arena.message import agent_msg_pb2, basic_type_pb2, session_msg_pb2
from arenaagent.utils.configclass import configclass
from loguru import logger


def summarize_subject_for_log(value: Any, key: str = "") -> Any:
    """Keep useful subject metadata without dumping image/Base64 payloads."""
    if isinstance(value, dict):
        return {item_key: summarize_subject_for_log(item, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list):
        return [summarize_subject_for_log(item, key) for item in value]
    if isinstance(value, bytes):
        digest = hashlib.sha256(value).hexdigest()[:12]
        return f"<binary bytes={len(value)} sha256={digest}>"
    if isinstance(value, str):
        is_image_payload = value.startswith("data:image") or (
            len(value) >= 128 and value.startswith(("iVBORw0KGgo", "/9j/", "R0lGOD", "UklGR", "Qk"))
        )
        if is_image_payload or (key in {"task_data", "image", "images"} and len(value) >= 128):
            digest = hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:12]
            return f"<image/base64 chars={len(value)} sha256={digest}>"
        if len(value) > 2000:
            digest = hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:12]
            return f"{value[:240]}...<truncated chars={len(value)} sha256={digest}>"
    return value

@configclass
class AgentCfg:
    name: str = "agent"
    sleep_between_steps: float = 0.5
    agent_body_asset_name: str = ""
    log_dir: str = "logs"

class AgentBase(ABC):
    def __init__(
        self,
        stub: TongTestAgentServiceStub,
        channel: grpc.Channel,
        cfg,
        sleep_between_steps: float = 2.0,
    ) -> None:
        self.stub = stub
        self.channel = channel
        self.cfg = cfg.copy() if cfg is not None else AgentCfg()
        self.sleep_between_steps = sleep_between_steps
        self.agent_id = uuid.uuid4().hex[:10]
        self.action_space: dict[str, Any] = {}
        self.connected = False
        self.character_id_in_simulation = ""
        self.subject_finished = False
        
    @abstractmethod
    def run_step(self, subject: str, task_response: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("AgentBase.run_step must be implemented by subclasses.")
    
    @abstractmethod
    def init(self, opt: dict[str, Any]):
        """
        init agent before connect to task
        """
        raise NotImplementedError("AgentBase.init must be implemented by subclasses.")

    @abstractmethod
    def deinit(self):
        """
        deinit agent after disconnect from task
        """
        raise NotImplementedError("AgentBase.deinit must be implemented by subclasses.")

    def load(self, params: dict[str, Any]) -> None:
        logger.debug("Loading agent config name={} keys={}", params.get("name"), sorted(params))
        try:
            self.cfg.from_dict(params)  # type: ignore[arg-type]
        except Exception:
            logger.warning("Agent config format may be invalid; falling back to defaults for {}", params.get("name"))
            pass

        self.sleep_between_steps = float(self.cfg.sleep_between_steps)
        if not self._connect():
            return
        
        init_info = self._get_agent_spawn_info(f"{self.cfg.name}.{self.agent_id}")
        logger.debug("get agent spawn info {}", init_info)
        self.init(init_info)
        
    def run(self) -> None:
        if not self.connected:
            return

        self.action_space = parse_struct_to_data(
            self._call_struct("get_action_space", {"agent_id": self.agent_id}, struct_pb2.Struct.FromString)
        )

        while not self._task_ready():
            time.sleep(self.sleep_between_steps)

        self._run_subject()
        # self._evaluate_task()

        while True:
            session_status = self._get_task_status().get("session_status")
            logger.debug(f"Current session status: {session_status}")
            if session_status in (
                session_msg_pb2.SessionStatus.Value("FINISHED"),
                session_msg_pb2.SessionStatus.Value("TERMINATED"),
                session_msg_pb2.SessionStatus.Value("ERROR"),
            ):
                logger.info(f"finished work Current session status: {session_status}")
                break
            time.sleep(1)

        self._disconnect()

    def _run_subject(self) -> None:
        # subject = self._get_subject_from_task()
        # logger.info("Agent[{}] is running the subject: {}", self.agent_id, subject)

        while not self._current_subject_finished():
            # logger.debug("Agent[{}] fetching task response...", self.agent_id)
            subject = self._get_subject_from_task()
            logger.info(
                "Agent[{}] is running the subject: {}",
                self.agent_id,
                summarize_subject_for_log(subject),
            )

            self.action_space = parse_struct_to_data(
                self._call_struct("get_action_space", {"agent_id": self.agent_id}, struct_pb2.Struct.FromString)
            )
            task_response = self._get_response_from_task()
            action = self.run_step(subject, task_response)
            apply_resp = self._apply_action(action)
            if self.subject_finished:
                logger.info("Subject finished by agent action.")
                break
            if self.sleep_between_steps > 0:
                time.sleep(self.sleep_between_steps)

        self._evaluate_subject()

    def _handle_finish(self, params: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
        self.subject_finished = True
        action_new = {}
        action_new[self.action_space["key"]] = str(action.get("think", "")) + str(action["output"])
        return action_new

    # tongtest grpc接口调用
    def _connect(self) -> bool:
        request = {"agent_id": self.agent_id}
        resp = self._call_struct("connect", request, basic_type_pb2.Bool.FromString)
        self.connected = bool(getattr(resp, "value", False))
        if not self.connected:
            logger.error("Agent failed to connect with agent_id=%s", self.agent_id)
        return self.connected

    def _disconnect(self) -> None:
        if not self.connected:
            return
        
        self.deinit()
        self._call_struct("disconnect", {"agent_id": self.agent_id}, basic_type_pb2.Bool.FromString)
        self.connected = False

    def _get_agent_spawn_info(self, agent_name: str) -> dict[str, Any]:
        payload = {"agent_name": agent_name}
        payload.setdefault("agent_id", self.agent_id)
        resp = self._call_struct("get_agent_spawn_info", payload, agent_msg_pb2.AgentMsg.FromString)
        if resp is None:
            return {}
        # Convert proto to plain dict; fallback to attributes if conversion fails.
        try:
            info = MessageToDict(resp, preserving_proto_field_name=True)
            if "camera_fov" not in info and "fov" in info:
                info["camera_fov"] = info["fov"]
            if "camera_width" not in info and "width" in info:
                info["camera_width"] = info["width"]
            if "camera_height" not in info and "height" in info:
                info["camera_height"] = info["height"]
            return info
        except Exception:
            fov = getattr(resp, "fov", "")
            width = getattr(resp, "width", "")
            height = getattr(resp, "height", "")
            return {
                "name": getattr(resp, "name", ""),
                "ue_server_ip": getattr(resp, "ue_server_ip", ""),
                "ue_server_port": getattr(resp, "ue_server_port", ""),
                "ue_proto_server_port": getattr(resp, "ue_proto_server_port", ""),
                "spawn_loc": getattr(resp, "spawn_loc", ""),
                "spawn_rot": getattr(resp, "spawn_rot", ""),
                "character_name": getattr(resp, "character_name", ""),
                "fov": fov,
                "width": width,
                "height": height,
                "camera_fov": fov,
                "camera_width": width,
                "camera_height": height,
            }

    def _apply_action(self, action: dict[str, Any]) -> dict[str, Any]:
        payload = {"action": action}
        payload.setdefault("agent_id", self.agent_id)
        resp = self._call_struct("update_action", payload, struct_pb2.Struct.FromString)
        return parse_struct_to_data(resp)

    def _get_subject_from_task(self) -> dict[str, Any]:
        resp = self._call_struct("get_subject", {"agent_id": self.agent_id}, struct_pb2.Struct.FromString)
        return parse_struct_to_data(resp)

    def _get_num_subjects_from_task(self) -> int:
        resp = self._call_struct("get_num_subjects", {"agent_id": self.agent_id}, basic_type_pb2.Int32.FromString)
        return getattr(resp, "value", 0)

    def _evaluate_subject(self) -> dict[str, Any]:
        resp = self._call_struct("evaluate_subject", {"agent_id": self.agent_id}, struct_pb2.Struct.FromString)
        return parse_struct_to_data(resp)

    def _evaluate_task(self) -> dict[str, Any]:
        resp = self._call_struct("evaluate_task", {"agent_id": self.agent_id}, struct_pb2.Struct.FromString)
        return parse_struct_to_data(resp)

    def _get_subject_score(self) -> dict[str, Any]:
        resp = self._call_struct("get_subject_score", {"agent_id": self.agent_id}, struct_pb2.Struct.FromString)
        return parse_struct_to_data(resp)

    def _get_session_status(self) -> dict[str, Any]:
        resp = self._call_struct(
            "get_session_status",
            {"agent_id": self.agent_id},
            session_msg_pb2.QuerySessionStatusResult.FromString,
        )
        return {"session_status": getattr(resp, "session_status", None)}

    def _get_task_status(self) -> dict[str, Any]:
        resp = self._call_struct(
            "get_task_status",
            {"agent_id": self.agent_id},
            session_msg_pb2.QuerySessionStatusResult.FromString,
        )
        return {"session_status": getattr(resp, "session_status", None)}

    def _task_ready(self) -> bool:
        resp = self._call_struct("is_ready_for_agent", {"agent_id": self.agent_id}, basic_type_pb2.Bool.FromString)
        return bool(getattr(resp, "value", False))

    def _current_subject_finished(self) -> bool:
        resp = self._call_struct(
            "is_current_subject_finished",
            {"agent_id": self.agent_id},
            basic_type_pb2.Bool.FromString,
        )
        return bool(getattr(resp, "value", False))

    def _get_response_from_task(self) -> dict[str, Any]:
        resp = self._call_struct("get_response", {"agent_id": self.agent_id}, struct_pb2.Struct.FromString)
        return parse_struct_to_data(resp)

    def _get_agent_id(self) -> str:
        resp = self._call_struct(
            "get_agent_id_in_simulation",
            {"agent_id": self.agent_id},
            basic_type_pb2.String.FromString,
        )
        return getattr(resp, "value", "")

    def _call_struct(self, method: str, payload: dict[str, Any], deserializer: Callable) -> Any:
        if self.channel is None:
            raise RuntimeError("gRPC channel missing in AgentBase.")
        request = pack_data_to_struct(payload)
        rpc = self.channel.unary_unary(
            f"/arena.agent.TongTestAgentService/{method}",
            request_serializer=request.SerializeToString,
            response_deserializer=deserializer,
        )
        return rpc(request)


def _normalize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalize_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_payload(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def pack_data_to_struct(payload: dict[str, Any]) -> struct_pb2.Struct:
    message = struct_pb2.Struct()
    if payload:
        ParseDict(_normalize_payload(payload), message)
    return message


def parse_struct_to_data(message: struct_pb2.Struct) -> dict[str, Any]:
    if message is None:
        return {}
    return MessageToDict(message, preserving_proto_field_name=True)
