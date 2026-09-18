"""扫完整圈后，用一次多图请求把整个房间的语义标出来。

逐帧问模型会让同一件家具在不同帧里得到互相矛盾的标签（实测同一件鞋架一帧
标 19、下一帧标 20），规划器据此算出的落点也是矛盾的。攒齐一圈一起发，模型
才能给出全局一致的那一份；而且往返次数从"每件物品一次"降到"每个房间一次"。
"""

from __future__ import annotations

import base64
import io
from typing import Any

from loguru import logger
from PIL import Image

from arenaagent.vlm_agent.json_parsor import extract_last_json_from_text

SURVEY_MAX_IMAGE_SIDE = 960
SURVEY_INVENTORY_LIMIT = 60

# 任务描述已经列全了类别，目的地映射也是赛题固定的，因此这里把词汇表封死，
# 不让模型自由发挥——它只需要回答"哪个编号属于哪一类"。
SURVEY_PROMPT = """这是整理房间任务的勘测阶段。下面若干张画面是同一个房间不同朝向的截图，
**右侧分割图上的数字就是物体编号**。

房间任务：将客厅内杂乱摆放的抱枕、鞋子、垃圾、食物、杯子等物品整理好。

这些物品只能放到四类家具上：
- 抱枕 → sofa
- 鞋子 → shoe_storage
- 垃圾 → trash_bin
- 食物、杯子、饮料容器 → dining_table

请**只输出一个 JSON 对象**，不要输出动作、不要输出 think：

{
  "items": [
    {"object_id": "37", "semantic_label": "shoe", "destination_type": "shoe_storage"}
  ],
  "furniture": [
    {"object_id": "14", "destination_type": "sofa"}
  ]
}

规则：
- items 是**需要整理的物品**；furniture 是上面四类目的地家具，其他家具（电视柜、椅子、灯等）不要列
- **只列你在这几张画面里真正看见的**。某一类这局房间里没有，就不要编一个出来；
  furniture 为空数组是完全正常的
- 地板、墙面、天花板虽然会出现在分割图上，但它们**不是家具**，绝不能当作目的地
- object_id 必须是画面上真实出现过的数字，没有把握的整条不要写
- destination_type 只能是 sofa / dining_table / trash_bin / shoe_storage 之一
- 同一件物体只出现一次
"""


def _shrink(image_b64: str, max_side: int = SURVEY_MAX_IMAGE_SIDE) -> str:
    """多图请求里每张图都要计费，按最长边等比缩小到够看清编号即可。"""
    payload = image_b64.split(",", 1)[-1]
    try:
        image = Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
        if max(image.size) <= max_side:
            return payload
        scale = max_side / max(image.size)
        resized = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
        output = io.BytesIO()
        resized.save(output, format="JPEG", quality=88, optimize=True)
        return base64.b64encode(output.getvalue()).decode("ascii")
    except Exception as exc:  # noqa: BLE001 - 缩图失败就发原图
        logger.warning("Survey image shrink failed; sending original: {}", exc)
        return payload


def _inventory_text(scene_objects: dict[str, dict[str, Any]]) -> str:
    """累积清单让模型能对照没出现在当前帧里的物体。"""
    lines = []
    for object_id, info in sorted(scene_objects.items(), key=lambda kv: _sort_key(kv[0])):
        shape = info.get("shape") or "Unknown"
        color = info.get("color") or "Unknown"
        if shape == "Unknown" and color == "Unknown":
            continue
        lines.append(f"  {object_id}: shape={shape} color={color}")
    if not lines:
        return "（暂无结构化清单）"
    return "\n".join(lines[:SURVEY_INVENTORY_LIMIT])


def _sort_key(object_id: str) -> tuple[int, str]:
    return (0, f"{int(object_id):08d}") if object_id.isdigit() else (1, object_id)


def build_survey_messages(frames: list[str], scene_objects: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": SURVEY_PROMPT},
        {"type": "text", "text": "本房间已见过的物体（编号: 形状 颜色）：\n" + _inventory_text(scene_objects)},
    ]
    for index, frame in enumerate(frames):
        content.append({"type": "text", "text": f"画面 {index + 1}/{len(frames)}："})
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_shrink(frame)}"}}
        )
    return [{"role": "user", "content": content}]


def run_scene_survey(strategy: Any, agent: Any) -> int:
    """执行一次勘测标注，返回成功登记的条数。失败时返回 0，不抛异常。"""
    frames = list(getattr(agent, "_tidyroom_survey_frames", []) or [])
    if not frames:
        logger.warning("Tidy-room survey skipped: no scan frames were retained")
        return 0
    client = agent._vlm_client_for_current_task()
    if client is None:
        return 0

    messages = build_survey_messages(frames, strategy.world.scene_objects)
    try:
        response = client.invoke(messages)
    except Exception as exc:  # noqa: BLE001 - 勘测失败退回逐帧标注
        logger.warning("Tidy-room survey request failed: {}", exc)
        return 0

    parsed = extract_last_json_from_text(getattr(response, "text", "") or "")
    if isinstance(parsed, dict):
        annotations = [
            {"object_id": entry.get("object_id"), **(entry or {})}
            for entry in (parsed.get("items") or [])
            if isinstance(entry, dict)
        ]
        annotations += [
            {"object_id": entry.get("object_id"), "destination_type": entry.get("destination_type")}
            for entry in (parsed.get("furniture") or [])
            if isinstance(entry, dict)
        ]
    elif isinstance(parsed, list):
        # 模型有时直接回一个数组；每条按自己的字段分类（有 semantic_label 是
        # 物品，只有 destination_type 是家具），不必强行套进上面的结构。
        annotations = [entry for entry in parsed if isinstance(entry, dict)]
    else:
        logger.warning("Tidy-room survey returned unparsable payload: {}", str(parsed)[:200])
        return 0
    applied = strategy.world.apply_scene_annotations(annotations, getattr(agent, "_task_context", None))
    logger.info(
        "Tidy-room survey over {} frames: submitted={} applied={} targets={} anchors={}",
        len(frames),
        len(annotations),
        applied,
        len(strategy.world.targets),
        len(strategy.world.scene_anchors),
    )
    return applied
