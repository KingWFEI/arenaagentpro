from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Location:
    """位置（XYZ 坐标）。"""

    X: float
    Y: float
    Z: float


@dataclass
class Rotation:
    """旋转（roll / yaw / pitch）。"""

    roll: float
    yaw: float
    pitch: float


class TongSimInterface(ABC):
    """TongSim 数据与动作接口（信息 / 动作）。"""

    # 功能接口
    @abstractmethod
    def acquire_first_person_image(self, character_id, encode_base64: bool = True) -> np.ndarray | str | None:
        """
        信息：获取角色的第一人称画面，可选返回 base64。
        参数：
            character_id: 仿真中的角色唯一标识。
            encode_base64: 是否以 base64 字符串返回。
        """
        raise NotImplementedError()

    @abstractmethod
    def set_object_pose(self, object_id: str, location, rotation: Rotation) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def acquire_first_person_segmantic_image(self, character_id, encode_base64: bool = True) -> np.ndarray | str | None:
        """
        信息：获取角色的第一人称的语义分割画面，可选返回 base64。
        参数：
            character_id: 仿真中的角色唯一标识。
            encode_base64: 是否以 base64 字符串返回。
        """
        raise NotImplementedError()

    @abstractmethod
    def fetch_first_person_visible_objects(self, character_id) -> list:
        """
        信息：列出第一人称视角可见的物体。
        参数：
            character_id: 仿真中的角色唯一标识。
        """
        raise NotImplementedError()

    @abstractmethod
    def get_object_basic_info(self, object_id: str) -> dict[str, Any]:
        """
        信息：根据 object_id 获取物体的基础属性（颜色、形状、放置位置）。
        返回值中的字段缺失时可返回 None。
        """
        raise NotImplementedError()

    @abstractmethod
    def get_object_world_aabb(self, object_id: str) -> dict[str, Any]:
        """
        信息：获取物体在世界坐标系下的 3D 轴对齐包围盒 (AABB)。
        返回值包含 min 和 max 两个字典，每个字典包含 x, y, z 坐标。
        """
        raise NotImplementedError()

    @abstractmethod
    def spawn_character(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def destory_character(self):
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get_object_id_by_name(self, name: str) -> str | None:
        """
        信息：根据物体名称获取物体信息（至少包含 object_id 字段）。
        """
        raise NotImplementedError()

    # 动作接口（按 api_info.json 定义，保留兼容旧接口）
    @abstractmethod
    def look_at_location(
        self, character_id, target_location, is_cancel: bool = False, execute_immediately: bool = False
    ):
        raise NotImplementedError()

    @abstractmethod
    def look_at_object(self, character_id, object_id: str, is_cancel: bool = False):
        raise NotImplementedError()

    @abstractmethod
    def point_at_object(self, character_id, object_id: str, is_cancel: bool = False, which_hand: int = 0):
        raise NotImplementedError()

    @abstractmethod
    def move_and_take_object(self, character_id, object_id: str, which_hand: int = 0):
        raise NotImplementedError()

    @abstractmethod
    def move_to_location(self, character_id, target_location, stop_distance: float = 0.5):
        raise NotImplementedError()

    @abstractmethod
    def move_forward(self, character_id, distance: float):
        raise NotImplementedError()

    @abstractmethod
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
        raise NotImplementedError()

    @abstractmethod
    def turn_in_degree(self, character_id, degree):
        raise NotImplementedError()

    @abstractmethod
    def move_to_object(self, character_id, object_id: str):
        raise NotImplementedError()

    @abstractmethod
    def pour_water(self, character_id, object_id: str, location, which_hand: int = 0):
        raise NotImplementedError()

    @abstractmethod
    def sit_down_to_object(self, character_id, object_id: str):
        raise NotImplementedError()

    @abstractmethod
    def slice_food(self, character_id, object_id: str, location):
        raise NotImplementedError()

    @abstractmethod
    def wash_hands(self, character_id, faucet_object_id: str):
        raise NotImplementedError()

    @abstractmethod
    def wash_object_in_hand(self, character_id, faucet_object_id: str):
        raise NotImplementedError()
    
    @abstractmethod
    def mop_floor(self, character_id, dirt_id: str):
        raise NotImplementedError()

    @abstractmethod
    def rest(self, character_id):
        raise NotImplementedError()

    @abstractmethod
    def speak_to_npc(self, character_id, target: str, content: str):
        raise NotImplementedError()

    # 兼容旧接口
    @abstractmethod
    def move_and_put_down_object_in_container(
        self,
        character_id,
        which_hand: int = 0,
    ):
        raise NotImplementedError()

    @abstractmethod
    def move_and_put_down(
        self,
        character_id,
        move_target_location,
        put_target_location,
        which_hand: int = 0,
        put_rotation: Rotation | None = None,
    ):
        raise NotImplementedError()

    @abstractmethod
    def get_object_in_hand(self, character_id) -> tuple[str, int] | None:
        raise NotImplementedError()
