from __future__ import annotations

import base64
import io
import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from arenaagent.tongsim_interface import TongSimInterface


def _rgb_to_int(rgb: np.ndarray) -> int:
    """将 RGB 颜色映射成单个整数 ID。"""
    return int(rgb[2]) * 256 * 256 + int(rgb[1]) * 256 + int(rgb[0])


def _build_segmentation_map(seg_image: Image.Image) -> np.ndarray:
    """把分割图转换为整数 ID 的二维数组。"""
    seg_array = np.array(seg_image)
    return np.apply_along_axis(_rgb_to_int, 2, seg_array)


def _color_from_id(idx: int) -> np.ndarray:
    """为 ID 生成可视化颜色（可重复）。"""
    rng = np.random.default_rng(idx)
    return rng.integers(0, 255, (3,), dtype=np.uint8)


@dataclass
class MappedObject:
    mapped_id: int
    segmentation_id: int | None
    raw_object_id: str
    center: tuple[int, int] | None = None  # (y, x)

    def __str__(self):
        return f"(mapped_id={self.mapped_id}),(seg_id={self.segmentation_id}),(raw_id={self.raw_object_id})"

class SemanticMapper:
    def __init__(
        self,
        tongsim: TongSimInterface | None = None,
        character_id: str | None = None,
        log_dir: str | None = None,
    ) -> None:
        # object_id_map: tongsim 原始 object_id -> 连续的 1..N 映射
        self.object_id_map: Dict[str, int] = {}
        # rev_object_id_map: 映射后的 ID -> tongsim 原始 object_id
        self.rev_object_id_map: Dict[int, str] = {}
        self.tongsim = tongsim
        self.character_id = character_id
        self.log_dir = log_dir
        # 同一局中颜色、形状、家具 AABB 等绝大部分信息不变；本地策略复用缓存，
        # 只有当前操作目标会通过任务策略单独刷新 AABB。
        self._object_details_cache: dict[str, dict[str, Any]] = {}
        self.last_perception_diagnostics: dict[str, Any] = {}

    def get_id_mapping(self, object_list: List[Any]) -> List[Dict[str, Any]]:
        """
        使用语义分割信息，把可见物体的 object_id 映射为 1..N 的整数。
        返回值中同时带上 segmentation_id 便于后续对齐。
        """
        mapped = []
        for obj in object_list:
            raw_id = getattr(obj, "object_id", None) if not isinstance(obj, dict) else obj.get("object_id")
            seg_id = getattr(obj, "segmentation_id", None) if not isinstance(obj, dict) else obj.get("segmentation_id")
            if raw_id is None:
                continue
            if raw_id not in self.object_id_map:
                new_id = len(self.object_id_map) + 1
                self.object_id_map[raw_id] = new_id
                self.rev_object_id_map[new_id] = raw_id

            mapped.append({"object_id": self.object_id_map[raw_id], "segmentation_id": seg_id})
        return mapped

    def get_raw_id(self, mapped_id: int) -> str | None:
        """通过映射后的 ID 找回 tongsim sdk 中的原始 visible object id。"""
        return self.rev_object_id_map.get(mapped_id)

    def register_identity_ids(self, object_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """把服务端原始 ID 直接登记为映射 ID（映射退化为恒等）。

        较新的服务端自己绘制分割图，图上的数字就是原始 ID，而提示词要求模型
        "只能使用分割图中的数字 ID"。若这里压缩成 1..N，模型从图上读到的数字
        会被再翻译一次，指向另一个物体。
        """
        mapped = []
        for obj in object_list:
            raw_id = str(obj["object_id"])
            try:
                identity = int(raw_id)
            except ValueError:
                # 非数字 ID（旧服务端的实例名）与分割图数字无法对应，顺延编号。
                identity = len(self.rev_object_id_map) + 1
            self.object_id_map[raw_id] = identity
            self.rev_object_id_map[identity] = raw_id
            mapped.append({"object_id": identity, "segmentation_id": None})
        return mapped

    def get_mapped_id(self, raw_id: str) -> int | None:
        """通过 tongsim sdk 中的原始 visible object id 找回映射后的 ID。"""
        return self.object_id_map.get(raw_id)

    def align_objects_with_segmentation(
        self, rgb_image: Image.Image, seg_image: Image.Image, object_list: List[Dict[str, Any]]
    ) -> Tuple[Dict[int, MappedObject], Image.Image]:
        """
        将 segmentation 图与 object_list 对齐，并生成 ID 可视化图。
        返回：映射后的物体信息、ID 伪彩色图。
        """
        rgb_array = np.array(rgb_image)
        seg_id_map = _build_segmentation_map(seg_image)

        aligned: Dict[int, MappedObject] = {}
        id_visual = rgb_array.copy()

        for obj in object_list:
            seg_id = obj.get("segmentation_id")
            mapped_id = obj.get("object_id")
            if seg_id is None or mapped_id is None:
                continue

            mask = seg_id_map == int(seg_id)
            if not mask.any():
                continue

            coords = np.argwhere(mask)
            center = tuple(coords.mean(axis=0).astype(int))  # (y, x)
            color = _color_from_id(mapped_id)

            id_visual[mask] = (0.7 * id_visual[mask] + 0.3 * color).astype(np.uint8)
            aligned[mapped_id] = MappedObject(
                mapped_id=mapped_id,
                segmentation_id=seg_id,
                raw_object_id=self.get_raw_id(mapped_id) or "",
                center=center,
            )

        # draw mapped_id text at centers for debugging/visualization
        for m_id, m_obj in aligned.items():
            if m_obj.center is None:
                continue
            y, x = int(m_obj.center[0]), int(m_obj.center[1])
            cv2.putText(
                id_visual,
                str(m_id),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        return aligned, Image.fromarray(id_visual.astype(np.uint8))

    @staticmethod
    def _decode_image(data: bytes | bytearray | str | np.ndarray | memoryview | None) -> Image.Image | None:
        """解码 rgb/seg 图数据，支持 bytes/base64/ndarray。"""
        if data is None:
            return None
        if isinstance(data, np.ndarray):
            return Image.fromarray(data)
        if isinstance(data, str):
            try:
                data = base64.b64decode(data)
            except Exception:
                return None
        if isinstance(data, (bytes, bytearray, memoryview)):
            try:
                return Image.open(io.BytesIO(data)).convert("RGB")
            except Exception:
                return None
        return None

    def get_perception_from_camera(
        self,
        is_save: bool = False,
        log_dir: str | None = None,
        *,
        include_images: bool = True,
        include_object_details: bool = True,
        use_cached_details: bool = False,
        save_label: str | None = None,
    ) -> tuple[str | None, list[dict[str, Any]], list[dict[str, Any]]]:
        """
        通过 TongSimInterface 获取感知结果：左侧 RGB，右侧 ID 伪彩色图，组合后以 base64(JPEG) 返回。
        同时返回当前可见物体的映射列表（含 segmentation_id）以及可见物体基础信息（颜色/形状/放置位置）。
        若 is_save=True，会将拼接后的图片保存到 log_dir（默认使用 AgentCfg.log_dir 或 logs）。

        较新的服务端提供一次性原子感知，已经拼好带标签的画面并附带全部物体详情，
        此时直接采用；旧服务端返回 None，退回下面的分步采集。
        """
        if self.tongsim is None or self.character_id is None:
            return None, [], []

        unified = self._acquire_unified_perception()
        if unified is not None:
            return self._perception_from_unified(
                unified,
                is_save=is_save,
                log_dir=log_dir,
                include_images=include_images,
                include_object_details=include_object_details,
                save_label=save_label,
            )
        return self._perception_from_split(
            is_save=is_save,
            log_dir=log_dir,
            include_images=include_images,
            include_object_details=include_object_details,
            use_cached_details=use_cached_details,
            save_label=save_label,
        )

    def _acquire_unified_perception(self) -> dict[str, Any] | None:
        """调用较新的统一感知接口；服务端不支持时返回 None。"""
        acquire = getattr(self.tongsim, "acquire_first_person_perception", None)
        if acquire is None:
            return None
        try:
            return acquire(self.character_id)
        except NotImplementedError:
            return None

    def _perception_from_unified(  # noqa: PLR0913
        self,
        perception: dict[str, Any],
        *,
        is_save: bool,
        log_dir: str | None,
        include_images: bool,
        include_object_details: bool,
        save_label: str | None,
    ) -> tuple[str | None, list[dict[str, Any]], list[dict[str, Any]]]:
        """把统一感知的响应整理成与分步采集一致的返回值。

        服务端负责同帧拼图，但场景刚加载或刚转向时可能返回“有 RGB、对象列表为空、
        右半分割图全黑”的过渡帧。因此仍记录统一感知的最低限度就绪指标，交给任务
        运行时决定是否重试。
        """
        raw_objects = [
            item
            for item in (perception.get("objects") or [])
            if isinstance(item, dict) and item.get("object_id") is not None
        ]
        visible_objects = self.register_identity_ids(raw_objects)
        image_b64 = perception.get("image")
        decoded = self._decode_image(image_b64)
        right_nonblack_ratio = 0.0
        left_right_difference_ratio = 0.0
        if decoded is not None and decoded.width >= 2:
            pixels = np.asarray(decoded.convert("RGB"))
            half_width = decoded.width // 2
            left_half = pixels[:, :half_width]
            right_half = pixels[:, half_width : half_width * 2]
            if right_half.size:
                right_nonblack_ratio = float(
                    np.count_nonzero(np.max(right_half, axis=2) > 8)
                    / (right_half.shape[0] * right_half.shape[1])
                )
                # 正常右半图是伪彩色分割图，绝大多数像素都应与同位置 RGB
                # 明显不同。服务端未就绪时右半图会直接复制 RGB，只给 NPC
                # 上色；仅看“非黑像素”无法识别这种坏帧。
                pixel_delta = np.max(
                    np.abs(left_half.astype(np.int16) - right_half.astype(np.int16)),
                    axis=2,
                )
                left_right_difference_ratio = float(
                    np.count_nonzero(pixel_delta > 24) / pixel_delta.size
                )
        self.last_perception_diagnostics = {
            "source": "unified",
            "visible_object_count": len(visible_objects),
            "visible_object_ids": tuple(
                sorted(str(item["object_id"]) for item in visible_objects)
            ),
            "image_present": decoded is not None,
            "right_nonblack_ratio": right_nonblack_ratio,
            "left_right_difference_ratio": left_right_difference_ratio,
        }

        visible_objects_info: list[dict[str, Any]] = []
        if include_object_details:
            for obj, mapped in zip(raw_objects, visible_objects, strict=True):
                info = {
                    "object_id": str(mapped["object_id"]),
                    "color": obj.get("color") or "Unknown",
                    "shape": obj.get("shape") or "Unknown",
                    "place_location": obj.get("place_location") or {},
                    "world_aabb": obj.get("world_aabb"),
                }
                visible_objects_info.append(info)
                self._object_details_cache[str(obj["object_id"])] = deepcopy(info)

        if not include_images:
            return None, visible_objects, visible_objects_info

        if not image_b64:
            logger.warning("unified perception returned no image: {}", perception.get("error"))
            return None, visible_objects, visible_objects_info

        if is_save:
            if decoded is not None:
                self._save_image(decoded, log_dir=log_dir, save_label=save_label)

        return image_b64, visible_objects, visible_objects_info

    def _perception_from_split(  # noqa: PLR0912, PLR0913, PLR0915
        self,
        *,
        is_save: bool,
        log_dir: str | None,
        include_images: bool,
        include_object_details: bool,
        use_cached_details: bool,
        save_label: str | None,
    ) -> tuple[str | None, list[dict[str, Any]], list[dict[str, Any]]]:
        """分步采集：分别取 RGB、分割图和可见物体，再在本地对齐并拼接。"""
        buffer_rgb = None
        buffer_seg = None
        if include_images:
            buffer_rgb = self.tongsim.acquire_first_person_image(self.character_id, encode_base64=False)
            buffer_seg = self.tongsim.acquire_first_person_segmantic_image(self.character_id, encode_base64=False)
        buffer_visible = self.tongsim.fetch_first_person_visible_objects(self.character_id)

        visible_objects = self.get_id_mapping(buffer_visible or []) if buffer_visible else []
        self.last_perception_diagnostics = {
            "visible_object_count": len(visible_objects),
            "aligned_object_count": 0,
            "segmentation_region_count": 0,
            "total_pixel_count": 0,
            "mapped_pixel_count": 0,
            "mapped_pixel_coverage": 0.0,
        }
        visible_objects_info: list[dict[str, Any]] = []
        if include_object_details and buffer_visible and self.tongsim:
            for obj in buffer_visible:
                raw_id = getattr(obj, "object_id", None) if not isinstance(obj, dict) else obj.get("object_id")
                if raw_id is None:
                    continue
                cache_key = str(raw_id)
                mapped_id = self.get_mapped_id(raw_id)
                mapped_id_str = str(mapped_id) if mapped_id is not None else str(raw_id)

                if use_cached_details and cache_key in self._object_details_cache:
                    cached = deepcopy(self._object_details_cache[cache_key])
                    cached["object_id"] = mapped_id_str
                    visible_objects_info.append(cached)
                    continue

                color = "Unknown"
                shape = "Unknown"
                place_location = None
                world_aabb = None

                try:
                    info = self.tongsim.get_object_basic_info(raw_id)
                    color = info.get("color") or color
                    shape = info.get("shape") or shape
                    place_location = info.get("place_location") if info else place_location
                except Exception as exc:  # pragma: no cover - runtime guard
                    logger.warning("failed to fetch object info for {}: {}", raw_id, exc)

                try:
                    aabb_info = self.tongsim.get_object_world_aabb(raw_id)
                    if aabb_info and aabb_info.get("min") and aabb_info.get("max"):
                        world_aabb = {
                            "min": aabb_info.get("min"),
                            "max": aabb_info.get("max"),
                        }
                except Exception as exc:  # pragma: no cover - runtime guard
                    logger.debug("failed to fetch world aabb for {}: {}", raw_id, exc)

                details = {
                    "object_id": mapped_id_str,
                    "color": color,
                    "shape": shape,
                    "place_location": place_location or {},
                    "world_aabb": world_aabb,
                }
                visible_objects_info.append(details)
                self._object_details_cache[cache_key] = deepcopy(details)

        # logger.info("Perception: {} visible objects", visible_objects_info)

        if not include_images:
            return None, visible_objects, visible_objects_info

        rgb_image = self._decode_image(buffer_rgb)
        if rgb_image is None:
            logger.warning("rgb image is None")
            return None, visible_objects, visible_objects_info

        id_vis = None
        if buffer_seg:
            seg_image = self._decode_image(buffer_seg)
            if seg_image is not None:
                segmentation_map = _build_segmentation_map(seg_image)
                self.last_perception_diagnostics["segmentation_region_count"] = int(
                    len(np.unique(segmentation_map))
                )
                total_pixel_count = int(segmentation_map.size)
                self.last_perception_diagnostics["total_pixel_count"] = total_pixel_count
                segmentation_ids: list[int] = []
                for item in visible_objects:
                    raw_segmentation_id = item.get("segmentation_id")
                    if raw_segmentation_id is None:
                        continue
                    try:
                        segmentation_ids.append(int(raw_segmentation_id))
                    except (TypeError, ValueError):
                        continue
                if segmentation_ids and total_pixel_count > 0:
                    mapped_pixel_count = int(
                        np.count_nonzero(np.isin(segmentation_map, segmentation_ids))
                    )
                    self.last_perception_diagnostics["mapped_pixel_count"] = mapped_pixel_count
                    self.last_perception_diagnostics["mapped_pixel_coverage"] = (
                        mapped_pixel_count / total_pixel_count
                    )
                if visible_objects:
                    aligned, id_vis = self.align_objects_with_segmentation(
                        rgb_image,
                        seg_image,
                        visible_objects,
                    )
                    self.last_perception_diagnostics["aligned_object_count"] = len(aligned)

        if id_vis is None:
            id_vis = Image.new("RGB", rgb_image.size, color=(0, 0, 0))

        left = rgb_image
        right = id_vis.resize(left.size)
        combined = Image.new("RGB", (left.width * 2, left.height))
        combined.paste(left, (0, 0))
        combined.paste(right, (left.width, 0))

        img_arr = np.array(combined)
        b64_image = base64.b64encode(cv2.imencode(".jpg", img_arr)[1]).decode("utf-8")

        if is_save:
            self._save_image(combined, log_dir=log_dir, save_label=save_label)

        return b64_image, visible_objects, visible_objects_info

    def _save_image(
        self,
        image: Image.Image,
        log_dir: str | None = None,
        save_label: str | None = None,
    ) -> None:
        target_dir = os.path.join(log_dir or self.log_dir or "logs", "prompts")
        try:
            os.makedirs(target_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            name_prefix = self.character_id or "agent"
            safe_label = "".join(
                char if char.isalnum() or char in {"-", "_"} else "_"
                for char in str(save_label or "")
            ).strip("_")
            label_part = f"_{safe_label}" if safe_label else ""
            file_path = os.path.join(
                target_dir,
                f"perception_{name_prefix}_{timestamp}{label_part}.jpg",
            )
            image.save(file_path, format="JPEG")
            logger.info("Saved perception image to {}", file_path)
        except Exception as exc:  # pragma: no cover - best-effort logging
            logger.warning("Failed to save perception image: %s", exc)

    # 兼容旧命令
    def get_perception(
        self, is_save: bool = False, log_dir: str | None = None
    ) -> tuple[str | None, list[dict[str, Any]], list[dict[str, Any]]]:
        return self.get_perception_from_camera(is_save=is_save, log_dir=log_dir)
