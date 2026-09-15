from __future__ import annotations

import copy
import json
import os
import re
import time
from datetime import datetime
from typing import Any

from google.protobuf import struct_pb2
from loguru import logger

from arenaagent.agent_base import AgentBase, AgentCfg, parse_struct_to_data
from arenaagent.builder import Register
from arenaagent.semantic_mapper import SemanticMapper
from arenaagent.tongsim_grpc_client import TongSimGrpcClient
from arenaagent.tongsim_interface import Rotation, TongSimInterface
from arenaagent.utils.configclass import configclass
from arenaagent.vlm_agent.client import ClientFactory
from arenaagent.vlm_agent.json_parsor import extract_last_json_from_text
from arenaagent.vlm_agent.prompt import PromptGenerator
from arenaagent.vlm_agent.raven_skill import (
    cleanup_raven_temp_images,
    materialize_task_data_images,
)
from arenaagent.vlm_agent.raven_skill import (
    handle as handle_raven_skill,
)
from arenaagent.vlm_agent.vlm_config import VLMConfig


@configclass
class VLMAgentCfg(AgentCfg):
    name: str = "vlm_agent"
    vlm_config: VLMConfig = VLMConfig()
    tongsim_server_endpoint: str = "127.0.0.1:50060"
    max_history_messages: int = 15


@Register("vlm_agent")
class VLMAgent(AgentBase):
    _MAX_ACTION_HISTORIES = 10
    _NPC_PINYIN_ALIASES: dict[str, str] = {
        "jiangshuyan": "江淑艳",
        "liuweidong": "刘伟东",
        "zhaoyeye": "赵爷爷",
        "zhangnainai": "张奶奶",
    }

    def __init__(
        self,
        stub,
        channel,
        cfg: VLMAgentCfg | None = None,
        sleep_between_steps: float = 2.0,
    ) -> None:
        super().__init__(
            stub=stub,
            channel=channel,
            cfg=cfg or VLMAgentCfg(),
            sleep_between_steps=sleep_between_steps,
        )
        self._initialized = False

        self.vlm_client = None
        self.tongsim: TongSimInterface = None
        self.character_id: str | None = None
        self.prompt_generator: PromptGenerator | None = None
        self.semantic_mapper: SemanticMapper | None = None
        self.history_messages: list[dict[str, Any]] = []
        self.last_json_parse_message = {}
        self._prompt_dump_index = 0
        self._last_visible_objects_info: list[dict[str, Any]] = []
        self._last_npc_reply: str = ""
        self._last_npc_subject: dict[str, Any] | None = None
        self._last_action_res: Any = {}
        self._last_apply_resp: dict[str, Any] = {}
        self._handled_piece_transfers: set[str] = set()
        self._npc_name_to_asset_name: dict[str, str] = {}
        self._task_spec_prompt_cache: dict[str, str] | None = None
        self._movable_objects: list[Any] = []
        self._action_histories: list[dict[str, Any]] = []
        self._raven_candidates_cache: dict[str, list[list[int]]] = {}
        self._raven_next_index: dict[str, int] = {}
        self._raven_image_temp_path: str = ""

    def init(self, opt: dict[str, Any]) -> None:
        if self._initialized:
            return

        self.cfg.vlm_config.apply_env_overrides()
        client_cfg = self.cfg.vlm_config.client_cfg
        logger.debug(
            "client config type={} name={} api_base={}",
            self.cfg.vlm_config.client_type,
            client_cfg.name,
            client_cfg.api_base,
        )
        self.vlm_client = ClientFactory().build(self.cfg.vlm_config.client_type, self.cfg.vlm_config.client_cfg)

        tongsim_server_endpoint = opt.get("tongsim_server_endpoint") or self.cfg.tongsim_server_endpoint
        self.tongsim = TongSimGrpcClient(endpoint=tongsim_server_endpoint)

        spawn_loc = json.loads(opt["spawn_loc"])
        spawn_rot = json.loads(opt["spawn_rot"])
        camera_fov = float(opt.get("camera_fov", 120.0))
        camera_width = int(opt.get("camera_width", 1280))
        camera_height = int(opt.get("camera_height", 720))
        self.character_id = self.tongsim.spawn_character(
            self.cfg.agent_body_asset_name, spawn_loc, spawn_rot, opt["name"], camera_fov, camera_width, camera_height
        )
        self.prompt_generator = PromptGenerator(self.cfg.vlm_config.prompt_config)
        self.semantic_mapper = SemanticMapper(self.tongsim, self.character_id, log_dir=self.cfg.log_dir)
        self._initialized = True

    def deinit(self):
        self._cleanup_raven_temp_images()
        if not self._initialized:
            return
        if self.tongsim:
            try:
                self.tongsim.close()
                logger.info("Released TongSim resources for character {}", self.character_id)
            except Exception as exc:
                logger.warning("Failed to release TongSim resources for {}: {}", self.character_id, exc)
        self.tongsim = None
        self.character_id = None
        self._initialized = False

    def run_step(self, subject, task_response: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR0915
        """
        1. 获取第一视角图片和可见物体映射；
        2. 生成大模型 prompt；
        3. 调用大模型；
        4. 解析回复结果；
        5. 执行动作；
        6. 将本轮结果写入历史消息。
        """
        if not self._initialized:
            self.init()

        if self._should_handle_piece_transfer():
            piece_action = self._maybe_handle_piece_transfer(subject)
            if piece_action is not None:
                logger.debug("Handled piece transfer for subject {}, returning action {}", subject, piece_action)
                return piece_action

        # 本地结构化策略无需每一步生成 RGB/分割拼图；需要调用 VLM 时再延迟截图。
        lightweight_perception = self._should_use_lightweight_perception(subject)
        refresh_lightweight_scene = self._should_refresh_lightweight_scene(subject)
        force_full_perception_vlm = False
        if self.semantic_mapper and lightweight_perception and refresh_lightweight_scene:
            initial_delay = self._lightweight_initial_perception_delay(subject)
            if initial_delay > 0:
                self._before_lightweight_initial_perception(subject)
                # 首帧先给 UE 相机与可见物体缓冲一个固定的稳定窗口，避免把
                # 场景加载中的瞬时空列表直接当成真实观察。
                logger.debug(
                    "Wait {:.2f}s before initial lightweight perception",
                    initial_delay,
                )
                time.sleep(initial_delay)
            b64_image, visible_objects, visible_objects_info = self._acquire_camera_perception(
                subject,
                is_save=False,
                include_images=False,
                include_object_details=True,
                use_cached_details=True,
            )
            force_full_perception_vlm = bool(
                not visible_objects
                and self._should_force_vlm_after_empty_lightweight_perception(subject)
            )
            empty_retry_delay = (
                0.0
                if force_full_perception_vlm
                else self._lightweight_empty_scene_retry_delay(subject)
            )
            if not visible_objects and empty_retry_delay > 0:
                # UE 场景刚启动或刚转向时，可见物体缓冲有时会先返回空列表。
                # 原地短等后重读比立即继续旋转更快，也不会污染扫描角度。
                logger.debug(
                    "Lightweight scene is empty; retry perception after {:.2f}s",
                    empty_retry_delay,
                )
                time.sleep(empty_retry_delay)
                b64_image, visible_objects, visible_objects_info = (
                    self._acquire_camera_perception(
                        subject,
                        is_save=False,
                        include_images=False,
                        include_object_details=True,
                        use_cached_details=True,
                    )
                )
        elif self.semantic_mapper and lightweight_perception:
            # 已经完成结构化建图的本地任务不必反复枚举整幅视野。原始 ID
            # 映射和世界模型均会保留，当前动作对象由任务策略按 ID 精确刷新。
            b64_image, visible_objects, visible_objects_info = None, [], []
        else:
            b64_image, visible_objects, visible_objects_info = (
                self._acquire_camera_perception(subject, is_save=True)
                if self.semantic_mapper
                else (None, [], [])
            )
        self._last_visible_objects_info = visible_objects_info or []
        image_data = self._to_data_url(b64_image)

        # 当前手中物体
        object_in_hand = None
        if self.tongsim and self.character_id:
            try:
                object_in_hand = self.tongsim.get_object_in_hand(self.character_id)
            except Exception as exc:
                logger.warning(f"获取手中物体失败: {exc}")

        # 4: 生成大模型 prompt
        api_info = self._load_api_info()
        if not self._npc_name_to_asset_name:
            self._npc_name_to_asset_name = (
                subject["npc_asset_name"] if isinstance(subject, dict) and "npc_asset_name" in subject else {}
            )

        self._before_prompt_hook(subject)

        if force_full_perception_vlm and self.semantic_mapper:
            # 首帧结构化结果为空时，直接执行一次完整采集。这里必须同时请求
            # RGB、分割图、可见 ID 和物体详情，随后用本次结果覆盖轻量结果，
            # 保证发给 VLM 的图像和结构化上下文来自同一次采集。
            self._on_forced_vlm(subject, "initial_scene_empty")
            b64_image, visible_objects, visible_objects_info = (
                self._acquire_camera_perception(
                    subject,
                    is_save=True,
                    include_images=True,
                    include_object_details=True,
                    use_cached_details=False,
                )
            )
            self._last_visible_objects_info = visible_objects_info or []
            image_data = self._to_data_url(b64_image)
            logger.info(
                "Initial scene was empty; acquired one full same-frame perception "
                "for VLM (visible_objects={})",
                len(visible_objects or []),
            )

        logger.info("current subject {}", subject["subject"])
        logger.info("current task response {}", task_response)
        prompt_variables = self._build_prompt_variables(
            subject=subject,
            task_response=task_response,
            api_info=api_info,
            visible_objects_info=visible_objects_info,
            object_in_hand=object_in_hand,
        )

        # 单题策略可以直接给出确定性动作，从而省去没有必要的远程模型调用。
        # 轻量模式到这里仅采集了结构化信息，尚未请求或保存相机图像。
        local_action = (
            None
            if force_full_perception_vlm
            else self._get_local_action(subject, task_response, prompt_variables)
        )
        if local_action is not None:
            logger.info("Using local deterministic action without VLM call: {}", local_action)
            self._after_prompt_hook(subject)
            return self._execute_action_and_record(local_action)

        if lightweight_perception and self.semantic_mapper and not force_full_perception_vlm:
            # 本地策略明确升级到 VLM 后才付出截图、语义分割和 JPEG 保存成本。
            # 升级时结构化信息也必须随 RGB/分割图一起重采，不能继续使用上一次
            # 轻量观察；否则模型看到的 ID、结构化属性与图片可能属于不同帧。
            b64_image, visible_objects, visible_objects_info = self._acquire_camera_perception(
                subject,
                is_save=True,
                include_images=True,
                include_object_details=True,
                use_cached_details=False,
            )
            self._last_visible_objects_info = visible_objects_info or []
            image_data = self._to_data_url(b64_image)
            logger.info(
                "Local strategy requested VLM; acquired one full same-frame perception "
                "(visible_objects={})",
                len(visible_objects or []),
            )
            prompt_variables = self._build_prompt_variables(
                subject=subject,
                task_response=task_response,
                api_info=api_info,
                visible_objects_info=visible_objects_info,
                object_in_hand=object_in_hand,
            )

        messages = (
            self.prompt_generator.Generate(  # type: ignore[union-attr]
                variables=prompt_variables,
                image=image_data,
                context_messages=self._trim_history_messages(),
                last_json_parse_message=self.last_json_parse_message
                if isinstance(self.last_json_parse_message, str)
                else "",
            )
            if self.prompt_generator
            else []
        )

        self._last_npc_reply = ""
        self._last_npc_subject = {}
        self._save_prompt_messages(messages)

        self._after_prompt_hook(subject)

        # 5: 调用大模型
        response = self.vlm_client.invoke(messages) if self.vlm_client else None
        response_text = getattr(response, "text", None) or ""
        json_parsed_message = extract_last_json_from_text(response_text)
        self.last_json_parse_message = json_parsed_message

        if response.token_usage:
            logger.info("Token Usage: \n")
            logger.info(
                "Prompt tokens: {}, Completion tokens: {}, Total tokens: {}",
                response.token_usage["prompt_tokens"],
                response.token_usage["completion_tokens"],
                response.token_usage["total_tokens"],
            )

        logger.info("vlm client response {} and parsed json message {}", str(response.text), str(json_parsed_message))
        # 6: 解析回复
        parsed_action = self._parse_action_from_response(json_parsed_message)

        # 7: 执行动作
        action_res = self._execute_action_and_record(parsed_action)

        # 9: 存入历史
        if messages:
            self._append_history_messages(
                [
                    messages[-1],  # user message
                    {"role": "assistant", "content": response_text},
                ]
            )

        return action_res if isinstance(action_res, dict) else {}

    def _get_local_action(
        self,
        subject: Any,
        task_response: dict[str, Any],
        prompt_variables: dict[str, Any],
    ) -> dict[str, Any] | None:
        """扩展点：默认仍调用 VLM，具体任务可返回一个本地确定性动作。"""
        del subject, task_response, prompt_variables
        return None

    def _should_use_lightweight_perception(self, subject: Any) -> bool:
        """扩展点：返回 True 时先只读取结构化物体数据，按需再获取图像。"""
        del subject
        return False

    def _acquire_camera_perception(
        self,
        subject: Any,
        **kwargs: Any,
    ) -> tuple[str | None, list[dict[str, Any]], list[dict[str, Any]]]:
        """扩展点：任务可在公共相机采集前后增加一致性检查。"""
        del subject
        if self.semantic_mapper is None:
            return None, [], []
        return self.semantic_mapper.get_perception_from_camera(**kwargs)

    def _should_refresh_lightweight_scene(self, subject: Any) -> bool:
        """扩展点：轻量模式下是否还需要枚举整幅当前视野。"""
        del subject
        return True

    def _lightweight_empty_scene_retry_delay(self, subject: Any) -> float:
        """扩展点：轻量感知意外为空时，原地等待多久后重读一次。"""
        del subject
        return 0.0

    def _lightweight_initial_perception_delay(self, subject: Any) -> float:
        """扩展点：首次轻量感知前等待场景稳定的时间。"""
        del subject
        return 0.0

    def _before_lightweight_initial_perception(self, subject: Any) -> None:
        """扩展点：首次轻量感知等待前执行相机或场景预热。"""
        del subject

    def _should_force_vlm_after_empty_lightweight_perception(self, subject: Any) -> bool:
        """扩展点：轻量结果为空时是否直接完整采集并调用 VLM。"""
        del subject
        return False

    def _on_forced_vlm(self, subject: Any, reason: str) -> None:
        """扩展点：记录由公共感知流程强制触发的 VLM 请求。"""
        del subject, reason

    def _execute_action_and_record(self, action: dict[str, Any]) -> dict[str, Any]:
        """统一执行动作并写入短历史，供本地策略与 VLM 路径共同使用。"""
        action_res = self._do_action(action)
        self._last_action_res = action_res if action_res is not None else {}
        if action:
            self._action_histories.append({"action": action, "result": self._last_action_res})
            self._trim_action_histories()
        return action_res if isinstance(action_res, dict) else {}

    def _append_history_messages(self, messages: list[dict[str, Any]]) -> None:
        if not messages:
            return
        self.history_messages.extend(messages)
        self._trim_history_messages()

    def _trim_history_messages(self) -> list[dict[str, Any]]:
        max_history_messages = max(int(getattr(self.cfg, "max_history_messages", 15) or 0), 0)
        if max_history_messages == 0:
            if self.history_messages:
                logger.debug(
                    "History messages disabled by config, clearing {} cached messages",
                    len(self.history_messages),
                )
                self.history_messages = []
            return self.history_messages

        if len(self.history_messages) <= max_history_messages:
            return self.history_messages

        original_length = len(self.history_messages)
        self.history_messages = self.history_messages[-max_history_messages:]

        # Keep the retained context starting from a user message when possible.
        if self.history_messages and self.history_messages[0].get("role") == "assistant":
            self.history_messages = self.history_messages[1:]

        logger.debug(
            "Trimmed history messages from {} to {} (limit={})",
            original_length,
            len(self.history_messages),
            max_history_messages,
        )
        return self.history_messages

    def _trim_action_histories(self) -> list[dict[str, Any]]:
        if len(self._action_histories) <= self._MAX_ACTION_HISTORIES:
            return self._action_histories

        original_length = len(self._action_histories)
        self._action_histories = self._action_histories[-self._MAX_ACTION_HISTORIES :]
        logger.debug(
            "Trimmed action histories from {} to {} (limit={})",
            original_length,
            len(self._action_histories),
            self._MAX_ACTION_HISTORIES,
        )
        return self._action_histories

    def _before_prompt_hook(self, subject):
        if isinstance(subject, dict):
            for data_key in ("task_data", "stage_data"):
                if data_key in subject and subject[data_key]:
                    self._materialize_task_data_images(subject[data_key])
                    break

        if isinstance(subject, dict) and "movable_object_id" in subject and subject["movable_object_id"]:
            self._enqueue_movable_objects(subject["movable_object_id"])

    def _after_prompt_hook(self, subject):
        pass

    def _materialize_task_data_images(self, task_data: Any) -> None:
        materialize_task_data_images(self, task_data)

    def _cleanup_raven_temp_images(self) -> None:
        cleanup_raven_temp_images(self)

    def _should_handle_piece_transfer(self) -> bool:
        return True

    def _build_prompt_variables(
        self,
        subject: Any,
        task_response: dict[str, Any],
        api_info: Any,
        visible_objects_info: list[dict[str, Any]],
        object_in_hand: Any,
    ) -> dict[str, Any]:
        return {
            "api_info": api_info,
            "task_goal": subject["goal"] if isinstance(subject, dict) else subject,
            "task_text": subject["task_prompt"] if isinstance(subject, dict) and "task_prompt" in subject else "",
            "visiable_objects_info": visible_objects_info,
            "example_objects_info": visible_objects_info[0] if len(visible_objects_info) > 0 else {},
            "npc_reply": self._last_npc_reply,
            "object_in_hand": object_in_hand,
            "npc_subject": self._last_npc_subject or {},
            "action_res": self._serialize_prompt_status(self._last_action_res),
            "apply_resp": self._serialize_prompt_status(self._last_apply_resp),
            "action_histories": self._trim_action_histories(),
        }

    def _save_prompt_messages(self, messages: list[dict[str, Any]]) -> None:
        if not messages:
            return
        prompt_dir = os.path.join(getattr(self.cfg, "log_dir", "") or "logs", "prompts")
        try:
            os.makedirs(prompt_dir, exist_ok=True)
            self._prompt_dump_index += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(
                prompt_dir,
                f"prompt_{self.agent_id}_{timestamp}_{self._prompt_dump_index:04d}.txt",
            )
            sanitized = self._strip_image_urls(messages)
            with open(file_path, "w", encoding="utf-8") as handle:
                json.dump(sanitized, handle, ensure_ascii=False, indent=2)
            logger.info("Saved prompt messages to {}", file_path)
        except Exception as exc:  # pragma: no cover - logging guard
            logger.warning("Failed to save prompt messages: %s", exc)

    @staticmethod
    def _strip_image_urls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove image_url fields and image_url message blocks before persisting prompts."""

        def cleanse(obj: Any) -> Any:
            if isinstance(obj, dict):
                obj = dict(obj)
                obj.pop("image_url", None)
                for k, v in list(obj.items()):
                    obj[k] = cleanse(v)
                if obj.get("type") == "image_url":
                    return None
                return obj
            if isinstance(obj, list):
                cleaned_list = []
                for item in obj:
                    cleaned_item = cleanse(item)
                    if cleaned_item is not None:
                        cleaned_list.append(cleaned_item)
                return cleaned_list
            return obj

        cleaned_messages: list[dict[str, Any]] = []
        for msg in copy.deepcopy(messages):
            cleaned = cleanse(msg)
            if cleaned is not None:
                cleaned_messages.append(cleaned)
        return cleaned_messages

    @staticmethod
    def _to_data_url(image_b64: str | None) -> str | None:
        if not image_b64:
            return None
        prefix = "data:image/jpeg;base64,"
        if image_b64.startswith("data:image"):
            return image_b64
        return f"{prefix}{image_b64}"

    def _parse_action_from_response(self, resp):
        """
        按 react.txt 约定的格式解析模型回复：
        [
          {
            "think": "...",
            "action": "move_and_take_object",
            "parameters": {...},
            "output": ...
          }
        ]
        """
        try:
            data = resp
            if isinstance(resp, str):
                text = resp.strip()
                if text.startswith("```"):
                    text = text.strip("` \n")
                    # 去掉可能的语言标记
                    prefix, separator, body = text.partition("\n")
                    if separator and not prefix.startswith(("{", "[")):
                        text = body
                data = json.loads(text)

            if isinstance(data, list) and data:
                first = data[0]
            elif isinstance(data, dict):
                first = data
            else:
                logger.error("响应格式异常，无法解析动作。")
                return {}

            action = first.get("action")
            params = first.get("parameters", {})
            output = first.get("output")
            think = first.get("think")
            return {"action": action, "parameters": params, "output": output, "think": think}
        except Exception as e:
            logger.error(f"解析动作失败: {e}")
            return {}

    def _enqueue_movable_objects(self, object_raw_id: list[Any]):
        logger.debug("object movable id {}", object_raw_id)
        self._movable_objects = object_raw_id

    def _maybe_handle_piece_transfer(self, subject: Any) -> dict[str, Any] | None:
        if not isinstance(subject, dict):
            return None

        stage_name = subject.get("stage", "")
        piece_object_id = subject.get("piece_object_id")
        piece_loc = subject.get("piece_loc")
        target_loc = subject.get("place_piece_target_loc")
        target_rot = subject.get("place_piece_target_rot")
        if piece_object_id is None or piece_loc is None or target_loc is None or target_rot is None:
            logger.warning(
                "Piece transfer parameters missing: piece_object_id={}, piece_loc={}, target_loc={}, target_rot={}",
                piece_object_id,
                piece_loc,
                target_loc,
                target_rot,
            )
            return None

        cache_key = f"{piece_object_id}:{piece_loc}:{target_loc}"
        if cache_key in self._handled_piece_transfers:
            return None

        if self.tongsim is None or self.character_id is None:
            logger.warning("TongSim not ready for piece transfer.")
            return {}

        raw_obj_id = self._to_raw_object_id(piece_object_id)
        target_loc = self._coerce_location(target_loc)

        try:
            object_in_hand = self.tongsim.get_object_in_hand(self.character_id)
            if object_in_hand:
                _, which_hand = object_in_hand
                which_hand = int(which_hand)
                drop_loc = self._get_current_forward_drop_location(forward_offset_cm=10.0)
                if not drop_loc:
                    drop_loc = target_loc
                if drop_loc:
                    self.tongsim.put_down_to_location(
                        self.character_id,
                        target_location=drop_loc,
                        which_hand=which_hand,
                    )
            self.tongsim.move_and_take_object(self.character_id, raw_obj_id, which_hand=0)
            logger.debug(
                "Picked up piece {} with raw id {}, now moving to target location {} and rotation {} for placement",
                piece_object_id,
                raw_obj_id,
                target_loc,
                target_rot,
            )
            rotation_values = [0.0, 0.0, 90.0]
            supplied_rotation = list(target_rot)[:3]
            rotation_values[: len(supplied_rotation)] = supplied_rotation
            roll, pitch, yaw = rotation_values
            put_rotation = Rotation(roll=roll, pitch=pitch, yaw=yaw)
            self.tongsim.move_to_location(self.character_id, target_loc, stop_distance=30.0)
            self.tongsim.put_down_to_location(
                self.character_id, target_location=target_loc, rotation=put_rotation, which_hand=0
            )
            self.tongsim.set_object_pose(raw_obj_id, target_loc, put_rotation)

            if stage_name == "jigsaw":
                view_loc = [target_loc[0] - 150, target_loc[1] - 120.0, 3]
                self.tongsim.move_to_location(self.character_id, view_loc, stop_distance=5.0)
                self.tongsim.turn_in_degree(self.character_id, 180)
        except Exception as exc:
            logger.warning("Piece transfer failed: {}", exc)
            return {}

        self._handled_piece_transfers.add(cache_key)
        return self._build_piece_transfer_action(piece_object_id)

    @staticmethod
    def _coerce_location(value: Any) -> Any:
        if value is None or isinstance(value, (dict, list, tuple)):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    pass
            numbers = re.findall(r"[-+]?\d*\.?\d+", stripped)
            try:
                x_value, y_value, z_value = numbers[:3]
                return [float(x_value), float(y_value), float(z_value)]
            except ValueError:
                pass
        return value

    def _get_current_forward_drop_location(self, forward_offset_cm: float = 10.0) -> list[float] | None:
        if self.tongsim is None or self.character_id is None:
            return None

        get_agent = getattr(self.tongsim, "_get_agent", None)
        if not callable(get_agent):
            return None

        try:
            agent = get_agent(str(self.character_id))
        except Exception:
            return None

        loc = None
        get_pose = getattr(agent, "get_pose", None)
        if callable(get_pose):
            pose = get_pose()
            loc = getattr(pose, "location", None)

        if loc is None:
            get_location = getattr(agent, "get_location", None)
            if callable(get_location):
                loc = get_location()

        if loc is None:
            loc = getattr(agent, "location", None)
        if loc is None:
            return None

        if isinstance(loc, dict):
            x = loc.get("x", loc.get("X"))
            y = loc.get("y", loc.get("Y"))
            z = loc.get("z", loc.get("Z"))
        else:
            x = getattr(loc, "x", getattr(loc, "X", None))
            y = getattr(loc, "y", getattr(loc, "Y", None))
            z = getattr(loc, "z", getattr(loc, "Z", None))

        if x is None or y is None or z is None:
            return None

        return [float(x), float(y) + float(forward_offset_cm), float(z)]

    def _build_piece_transfer_action(self, piece_object_id: Any) -> dict[str, Any]:
        key = self.action_space.get("key") or "action"
        action = {
            key: "piece_transfer_done",
            "piece_transfer_done": True,
            "piece_object_id": piece_object_id,
        }
        if key != "action":
            action["action"] = "piece_transfer_done"
        return action

    def _do_action(self, action: dict[str, Any]) -> Any:
        """
        根据解析结果调用 TongSim 接口。
        """
        if not action:
            logger.error("动作为空，无法执行。")
            return self._fail_result(error="empty action")
        if self.tongsim is None or self.character_id is None:
            logger.error("TongSim 未初始化。")
            return self._fail_result(error="TongSim not initialized")

        name = (action.get("action") or "").lower()
        params = action.get("parameters") or {}

        handler = {
            "finish_task": self._handle_finish,
            "submit_answer": self._handle_submit_answer,
            "submit_puzzle_answer": self._handle_submit_answer,
            "solve_raven": self._handle_solve_raven,
            "look_at_location": self._handle_look_at_location,
            "look_at_object": self._handle_look_at_object,
            "point_at_object": self._handle_point_at_object,
            "move_and_take_object": self._handle_move_and_take,
            "move_forward": self._handle_move_forward,
            "move_backward": self._handle_move_backward,
            "put_down_to_location": self._handle_put_down_to_location,
            "turn_in_degree": self._handle_turn_degree,
            "move_to_object": self._handle_move_to_object,
            "move_to_npc": self._handle_move_to_npc,
            "move_to_location": self._handle_move_to_location,
            "pour_water": self._handle_pour_water,
            "sit_down_to_object": self._handle_sit_down_to_object,
            "slice_food": self._handle_slice_food,
            "wash_hands": self._handle_wash_hands,
            "wash_object_in_hand": self._handle_wash_object_in_hand,
            "mop_floor": self._handle_mop_floor,
            "rest": self._handle_rest,
            "speak_to_npc": self._handle_speak_to_npc,
            "move_and_put_down": self._handle_move_and_put_down,
            # 兼容旧动作
            "move_and_put_down_object_in_container": self._handle_put_in_container,
            "put_down_sth_to_location": self._handle_put_down_to_location,
            "turn_around_to_degree": self._handle_turn_degree,
        }.get(name)

        if handler is None:
            logger.error(f"未支持的动作类型: {name}")
            return self._fail_result(error=f"unsupported action type: {name}")

        try:
            result = handler(params, action)
            logger.debug("action run result {}", result)
        except Exception as exc:
            logger.exception("动作执行失败 {}: {}", name, exc)
            return self._fail_result(error=str(exc))

        return result

    @staticmethod
    def _serialize_prompt_status(value: Any) -> str:
        if value is None:
            return "{}"
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)

    def _apply_action(self, action: dict[str, Any]) -> dict[str, Any]:
        resp = super()._apply_action(action)
        self._last_apply_resp = resp if isinstance(resp, dict) else {}
        return resp

    @staticmethod
    def _fail_result(error: str = "error message", **kwargs: Any) -> dict[str, Any]:
        result = {"result": "failed", "error": str(error)}
        result.update(kwargs)
        return result

    def _handle_submit_answer(self, params: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
        key = self.action_space.get("key") or "action"
        return {key: str(action["output"])}

    def _handle_solve_raven(self, params: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
        return handle_raven_skill(self, params, action)

    def _get_param(self, params: dict[str, Any], *keys: str, default=None):
        for k in keys:
            if k in params:
                return params[k]
            lk = k.lower()
            for key, value in params.items():
                if key.lower() == lk:
                    return value
        return default

    def _to_raw_object_id(self, obj_id: Any) -> Any:
        if obj_id is None or self.semantic_mapper is None:
            return obj_id
        if isinstance(obj_id, bool):
            return obj_id
        mapped_id = None
        if isinstance(obj_id, int):
            mapped_id = obj_id
        elif isinstance(obj_id, float) and obj_id.is_integer():
            mapped_id = int(obj_id)
        elif isinstance(obj_id, str):
            stripped = obj_id.strip()
            if stripped.isdigit():
                mapped_id = int(stripped)
        if mapped_id is None:
            return obj_id
        raw_id = self.semantic_mapper.get_raw_id(mapped_id)
        return raw_id

    def _handle_look_at_location(self, params: dict[str, Any], action: dict[str, Any]) -> Any:
        target_location = self._get_param(params, "target_location", "location")
        is_cancel = bool(self._get_param(params, "is_cancel", default=False))
        execute_immediately = bool(self._get_param(params, "execute_immediately", default=False))
        return self.tongsim.look_at_location(
            self.character_id,
            target_location,
            is_cancel=is_cancel,
            execute_immediately=execute_immediately,
        )

    def _handle_look_at_object(self, params: dict[str, Any], action: dict[str, Any]) -> Any:
        obj_id = self._get_param(params, "object_id", "object")
        obj_id = self._to_raw_object_id(obj_id)
        is_cancel = bool(self._get_param(params, "is_cancel", default=False))
        if not obj_id:
            logger.error("缺少 object_id，无法执行 look_at_object。")
            return self._fail_result(error="missing required parameter: object_id")
        return self.tongsim.look_at_object(self.character_id, obj_id, is_cancel=is_cancel)

    def _handle_point_at_object(self, params: dict[str, Any], action: dict[str, Any]) -> Any:
        obj_id = self._get_param(params, "object_id", "object")
        obj_id = self._to_raw_object_id(obj_id)
        is_cancel = bool(self._get_param(params, "is_cancel", default=False))
        which_hand = self._get_param(params, "which_hand", default=0)
        if not obj_id:
            logger.error("缺少 object_id，无法执行 point_at_object。")
            return self._fail_result(error="missing required parameter: object_id")
        return self.tongsim.point_at_object(
            self.character_id,
            obj_id,
            is_cancel=is_cancel,
            which_hand=which_hand,
        )

    def _handle_move_and_take(self, params: dict[str, Any], action: dict[str, Any]) -> Any:
        obj_id = self._get_param(params, "object_id", "object")
        raw_obj_id = self._to_raw_object_id(obj_id)
        if raw_obj_id not in self._movable_objects:
            return {"result": "failed", "error": "can not take this object"}

        which_hand = self._get_param(params, "which_hand", default=0)
        if not raw_obj_id:
            logger.error("缺少 object_id，无法执行 move_and_take_object。")
            return self._fail_result(error="missing required parameter: object_id")
        return self.tongsim.move_and_take_object(self.character_id, raw_obj_id, which_hand=which_hand)

    def _handle_put_in_container(self, params: dict[str, Any], action: dict[str, Any]) -> Any:
        which_hand = self._get_param(params, "which_hand", default=0)
        return self.tongsim.move_and_put_down_object_in_container(
            self.character_id,
            which_hand=which_hand,
        )

    def _handle_move_and_put_down(self, params: dict[str, Any], action: dict[str, Any]) -> Any:
        move_target_location = self._get_param(params, "move_target_location", "move_location")
        put_target_location = self._get_param(params, "put_target_location", "put_location", "target_location")
        which_hand = self._get_param(params, "which_hand", default=0)
        rot_val = self._get_param(params, "put_rotation", "rotation")
        put_rotation = None
        if isinstance(rot_val, dict):
            put_rotation = Rotation(
                roll=rot_val.get("roll", 0.0),
                yaw=rot_val.get("yaw", 0.0),
                pitch=rot_val.get("pitch", 0.0),
            )
        if move_target_location is None or put_target_location is None:
            logger.error("缺少 move_target_location 或 put_target_location，无法执行 move_and_put_down。")
            return self._fail_result(error="missing required parameter: move_target_location or put_target_location")
        return self.tongsim.move_and_put_down(
            self.character_id,
            move_target_location=move_target_location,
            put_target_location=put_target_location,
            which_hand=which_hand,
            put_rotation=put_rotation,
        )

    def _handle_put_down_to_location(self, params: dict[str, Any], action: dict[str, Any]) -> Any:
        loc = self._get_param(params, "target_location", "location")
        rot_val = self._get_param(params, "rotation")
        rotation = None
        if isinstance(rot_val, dict):
            rotation = Rotation(
                roll=rot_val.get("roll", 0.0),
                yaw=rot_val.get("yaw", 0.0),
                pitch=rot_val.get("pitch", 0.0),
            )
        which_hand = self._get_param(params, "which_hand", default=0)
        auto_rotate = bool(self._get_param(params, "auto_rotate", default=False))
        force_release = bool(self._get_param(params, "force_release", default=True))
        disable_physics = bool(self._get_param(params, "disable_physics", default=False))
        hold_if_unreachable = bool(self._get_param(params, "hold_if_unreachable", default=True))
        force_locate = bool(self._get_param(params, "force_locate", default=False))
        return self.tongsim.put_down_to_location(
            self.character_id,
            target_location=loc,
            which_hand=which_hand,
            rotation=rotation,
            auto_rotate=auto_rotate,
            force_release=force_release,
            disable_physics=disable_physics,
            hold_if_unreachable=hold_if_unreachable,
            force_locate=force_locate,
        )

    def _handle_move_forward(self, params: dict[str, Any], action: dict[str, Any]) -> Any:
        distance = self._get_param(params, "distance", "step", default=0.0)
        return self.tongsim.move_forward(self.character_id, float(distance))

    def _handle_move_backward(self, params: dict[str, Any], action: dict[str, Any]) -> Any:
        turn_result = self._handle_turn_degree({"degree": 180.0}, action)
        if isinstance(turn_result, dict) and turn_result.get("result") == "failed":
            return turn_result

        move_result = None
        turn_back_result = None
        try:
            move_result = self._handle_move_forward(params, action)
            return move_result
        finally:
            turn_back_result = self._handle_turn_degree({"degree": 180.0}, action)
            if isinstance(turn_back_result, dict) and turn_back_result.get("result") == "failed":
                if isinstance(move_result, dict) and move_result.get("result") == "failed":
                    move_result["turn_back_error"] = turn_back_result.get("error")
                elif isinstance(move_result, dict):
                    move_result["result"] = "failed"
                    move_result["error"] = turn_back_result.get("error", "failed to turn back after move_backward")
                else:
                    raise RuntimeError(turn_back_result.get("error", "failed to turn back after move_backward"))

    def _handle_turn_degree(self, params: dict[str, Any], action: dict[str, Any]) -> Any:
        degree = self._get_param(params, "degree", default=0.0)
        return self.tongsim.turn_in_degree(self.character_id, float(degree))

    def _handle_move_to_location(self, params: dict[str, Any], action: dict[str, Any]) -> Any:
        target_location = self._get_param(params, "target_location", "location")
        stop_distance = self._get_param(params, "stop_distance", default=0.5)
        if target_location is None:
            logger.error("缺少 target_location，无法执行 move_to_location。")
            return self._fail_result(error="missing required parameter: target_location")

        return self.tongsim.move_to_location(self.character_id, target_location, stop_distance=float(stop_distance))

    def _handle_move_to_object(self, params: dict[str, Any], action: dict[str, Any]) -> Any:
        obj_id = self._get_param(params, "object_id", "object")
        raw_obj_id = self._to_raw_object_id(obj_id)
        if not raw_obj_id:
            loc = self._find_object_location(raw_obj_id)
            if loc:
                return self.tongsim.move_to_location(self.character_id, loc)

            logger.error("缺少 object_id 或 object_id 无效，无法执行 move_to_object。")
            return self._fail_result(error="missing or invalid parameter: object_id")
        else:
            return self.tongsim.move_to_object(self.character_id, raw_obj_id)

    def _handle_move_to_npc(self, params: dict[str, Any], action: dict[str, Any]) -> Any:
        raw_npc_name = self._get_param(params, "npc_name", "npc", "target", "name")
        npc_name = self._resolve_npc_name(raw_npc_name)
        if not npc_name:
            logger.error("缺少 npc_name，无法执行 move_to_npc。")
            return self._fail_result(error="missing required parameter: npc_name")

        npc_asset_name = self._npc_name_to_asset_name.get(npc_name)
        npc_object_id = self.tongsim.get_object_id_by_name(npc_asset_name) if npc_asset_name else None
        logger.info(
            "move_to_npc got npc_name {} mapped to asset_name {} and object_id {}",
            npc_name,
            npc_asset_name,
            npc_object_id,
        )
        if npc_object_id:
            return self.tongsim.move_to_object(self.character_id, npc_object_id)
        return self._fail_result(error=f"npc not found: {npc_name}")

    def _find_object_location(self, obj_id: Any) -> dict[str, Any] | None:
        if obj_id is None:
            return None
        target = str(obj_id)
        for obj in self._last_visible_objects_info:
            if str(obj.get("object_id")) == target:
                return obj.get("place_location") or None
        return None

    def _handle_pour_water(self, params: dict[str, Any], action: dict[str, Any]) -> Any:
        obj_id = self._get_param(params, "object_id", "object")
        location = self._get_param(params, "location", "target_location")
        which_hand = self._get_param(params, "which_hand", default=0)
        obj_id = self._to_raw_object_id(obj_id)
        if not obj_id or location is None:
            logger.error("缺少 object_id 或 location，无法执行 pour_water。")
            return self._fail_result(error="missing required parameter: object_id or location")
        return self.tongsim.pour_water(self.character_id, obj_id, location, which_hand=which_hand)

    def _handle_sit_down_to_object(self, params: dict[str, Any], action: dict[str, Any]) -> Any:
        obj_id = self._get_param(params, "object_id", "object")
        obj_id = self._to_raw_object_id(obj_id)
        if not obj_id:
            logger.error("缺少 object_id，无法执行 sit_down_to_object。")
            return self._fail_result(error="missing required parameter: object_id")
        return self.tongsim.sit_down_to_object(self.character_id, obj_id)

    def _handle_slice_food(self, params: dict[str, Any], action: dict[str, Any]) -> Any:
        obj_id = self._get_param(params, "object_id", "object")
        location = self._get_param(params, "location", "target_location")
        obj_id = self._to_raw_object_id(obj_id)
        if not obj_id or location is None:
            logger.error("缺少 object_id 或 location，无法执行 slice_food。")
            return self._fail_result(error="missing required parameter: object_id or location")
        return self.tongsim.slice_food(self.character_id, obj_id, location)

    def _handle_wash_hands(self, params: dict[str, Any], action: dict[str, Any]) -> Any:
        faucet_id = self._get_param(params, "faucet_object_id", "object_id", "object")
        faucet_id = self._to_raw_object_id(faucet_id)
        if not faucet_id:
            logger.error("缺少 faucet_object_id，无法执行 wash_hands。")
            return self._fail_result(error="missing required parameter: faucet_object_id")
        return self.tongsim.wash_hands(self.character_id, faucet_id)

    def _handle_wash_object_in_hand(self, params: dict[str, Any], action: dict[str, Any]) -> Any:
        faucet_id = self._get_param(params, "faucet_object_id", "object_id", "object")
        faucet_id = self._to_raw_object_id(faucet_id)
        if not faucet_id:
            logger.error("缺少 faucet_object_id，无法执行 wash_object_in_hand。")
            return self._fail_result(error="missing required parameter: faucet_object_id")
        return self.tongsim.wash_object_in_hand(self.character_id, faucet_id)

    def _handle_mop_floor(self, params: dict[str, Any], action: dict[str, Any]) -> Any:
        dirt_id = self._get_param(params, "dirt_id", "object_id", "object")
        dirt_id = self._to_raw_object_id(dirt_id)
        if not dirt_id:
            logger.error("缺少 dirt_id，无法执行 mop_floor。")
            return self._fail_result(error="missing required parameter: dirt_id")
        return self.tongsim.mop_floor(self.character_id, dirt_id)

    def _handle_rest(self, params: dict[str, Any], action: dict[str, Any]) -> Any:
        return self.tongsim.rest(self.character_id)

    def _handle_speak_to_npc(self, params: dict[str, Any], action: dict[str, Any]) -> Any:
        raw_target = self._get_param(params, "npc_name", "npc", "npc_id", "target", "name")
        target = self._resolve_npc_name(raw_target)
        if not target:
            logger.error("缺少 npc_name，无法执行 speak_to_npc。")
            return self._fail_result(error="missing required parameter: npc_name")

        move_result = self._handle_move_to_npc({"npc_name": target}, action)
        if isinstance(move_result, dict) and move_result.get("result") == "failed":
            return move_result

        message_text = self._extract_speak_to_npc_message(params, action)
        submitted_message = message_text
        payload = {
            "agent_id": self.agent_id,
            "npc_name": str(target),
            "target": str(target),
            "message": submitted_message,
            "content": submitted_message,
        }
        try:
            resp = self._call_struct("speak_to", payload, struct_pb2.Struct.FromString)
            data = parse_struct_to_data(resp)
        except Exception as exc:
            logger.warning("speak_to gRPC failed: {}", exc)
            return self._fail_result(error=str(exc))

        reply = data.get("npc_reply") or data.get("reply") or data.get("content") or ""
        self._last_npc_reply = str(reply)
        hints = data.get("hints")
        logger.debug("speak_to_npc got reply {} with hints {}", self._last_npc_reply, hints)
        if isinstance(hints, dict):
            self._last_npc_subject = hints
        elif hints:
            self._last_npc_subject = {"hints": hints}
        else:
            self._last_npc_subject = None
        if isinstance(data, dict):
            data = dict(data)
            data["move_result"] = move_result
            data["npc_name"] = str(target)
            data["submitted_message"] = submitted_message
            return data
        return {
            "move_result": move_result,
            "result": data,
            "npc_name": str(target),
            "submitted_message": submitted_message,
        }

    @staticmethod
    def _normalize_npc_lookup_key(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if not text:
            return ""
        return re.sub(r"[\s_.-]+", "", text).lower()

    def _resolve_npc_name(self, value: Any) -> str:
        if value is None or isinstance(value, bool):
            return ""

        text = str(value).strip()
        if not text:
            return ""
        if text in self._npc_name_to_asset_name:
            return text

        lookup_key = self._normalize_npc_lookup_key(text)
        alias_map = {
            self._normalize_npc_lookup_key(name): str(name)
            for name in self._npc_name_to_asset_name
            if str(name).strip()
        }
        for alias_key, canonical_name in self._NPC_PINYIN_ALIASES.items():
            if canonical_name in self._npc_name_to_asset_name:
                alias_map.setdefault(alias_key, canonical_name)

        resolved_name = alias_map.get(lookup_key)
        if resolved_name:
            return resolved_name

        # Allow free-form outputs like "npc_zhaoyeye" or "去找zhaoyeye对话".
        for alias_key, canonical_name in self._NPC_PINYIN_ALIASES.items():
            if canonical_name in self._npc_name_to_asset_name and alias_key in lookup_key:
                return canonical_name

        return text

    def _extract_speak_to_npc_message(self, params: dict[str, Any], action: dict[str, Any]) -> str:
        message = self._get_param(params, "message", "content", "text", "reply", "answer")
        if message in (None, ""):
            message = "你好！告诉我点什么吧"

        return str(message).strip()

    def _load_api_info(self) -> Any:
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            api_info_path = os.path.join(here, "prompts", "api_info.json")
            with open(api_info_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning(f"加载 api_info.json 失败: {exc}")
            return {}

    @staticmethod
    def _to_vector3(values: list[float]) -> list[float]:
        padded = list(values) + [0.0, 0.0, 0.0]
        return [float(padded[0]), float(padded[1]), float(padded[2])]

    @staticmethod
    def _zeros(shape: list[int]) -> list:
        if not shape:
            return 0
        size = int(shape[0])
        if len(shape) == 1:
            return [0.0] * size
        return [VLMAgent._zeros(shape[1:]) for _ in range(size)]

    @staticmethod
    def _agent_location(agent) -> str:
        for attr in ("get_location", "get_pose", "pose"):
            value = getattr(agent, attr, None)
            if callable(value):
                return str(value())
            if value is not None:
                return str(value)
        logger.debug("Agent location not available on TongSim entity.")
        return "<unknown>"
