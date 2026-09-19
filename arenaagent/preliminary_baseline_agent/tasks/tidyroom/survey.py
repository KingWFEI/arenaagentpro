"""初始正向分割稳定后，用一次单图请求标注可见物品和目的地。"""

from __future__ import annotations

import base64
import io
from typing import Any

from loguru import logger
from PIL import Image

from arenaagent.vlm_agent.json_parsor import extract_last_json_from_text

# 分割编号是关键信息；过度缩到 960 会让右半幅的小物体编号难以辨认。
SURVEY_MAX_IMAGE_SIDE = 2048
SURVEY_INVENTORY_LIMIT = 60

# 任务描述已经列全了类别，目的地映射也是赛题固定的，因此这里把词汇表封死，
# 不让模型自由发挥——它只需要回答"哪个编号属于哪一类"。
SURVEY_PROMPT = """这是整理房间任务的勘测阶段。下面只有一张初始正向截图，
**右侧分割图上的数字就是物体编号**。

房间任务：将客厅内杂乱摆放的抱枕、鞋子、垃圾、食物、杯子等物品整理好。

这些物品只能放到四类家具上：
- 抱枕 → sofa
- 鞋子 → shoe_storage
- 垃圾 → trash_bin
- 食物、杯子、饮料容器 → dining_table

请**只输出一个 JSON 对象**，不要输出动作、不要输出 think。严格使用下面的空结构，
再把你确实识别出的条目填入数组；结构中的空数组不是示例答案：

{
  "items": [],
  "furniture": []
}

items 每条必须包含 object_id、semantic_label、destination_type；只有目的地家具也被
你明确识别并列入 furniture 时，才能额外填写 destination_object_id。furniture 每条
必须包含 object_id、destination_type。

规则：
- items 是**需要整理的物品**；furniture 是上面四类目的地家具，其他家具（电视柜、椅子、灯等）不要列
- 先逐一检查所有显眼的地面/茶几散落小物体，不要因为已经找到一两件就提前停止
- furniture 要单独遍历整张图，检查 sofa / dining_table / trash_bin / shoe_storage 四类；看得见就必须列出
- 若某个 item 对应的目的地在图中可见，必须在该 item 中填写 destination_object_id，且该 ID 也必须出现在 furniture 中
- 已经放在/靠在正确家具上的物品不是杂物，绝不能列入 items。例如鞋架格子里或紧靠鞋架整齐陈列的鞋、沙发上正常摆放的靠垫、垃圾桶里的垃圾、餐桌上正常摆放的餐具都不要搬
- 只有明显散落在地板、茶几等错误位置，确实需要从当前位置搬走的物品才列入 items
- **只列你在这张画面里真正看见的**。某一类这局房间里没有，就不要编一个出来；
  furniture 为空数组是完全正常的
- 地板、墙面、天花板虽然会出现在分割图上，但它们**不是家具**，绝不能当作目的地
- object_id 必须是画面上真实出现过的数字，没有把握的整条不要写
- 不要根据字段说明或历史常见编号猜测 ID；每局编号都会变化，只能读取本图右侧数字
- 鞋必须具有明确的鞋/靴外形；细长落地灯绝不是 shoe_storage，红色直立罐体也不是鞋
- shoe_storage 应当是具有搁板、格口或柜体的鞋架/鞋柜；若本图看不到就不要编造
- destination_type 只能是 sofa / dining_table / trash_bin / shoe_storage 之一
- 结构化清单中带 FIXED_FURNITURE=... 的编号已由固定世界坐标确认；
  它们都是不可拾取的环境家具，绝对不得列入 items。目的地已由本地
  先验建立，你只需识别待整理小物体；furniture 可以为空
- 必须结合下面结构化清单里的 AABB 尺寸判断：几厘米大的小球/核桃绝不是抱枕；
  约 30cm 以上长、截面十几厘米、横放在地面的软质圆柱/长方体优先判断为抱枕，
  不能仅因轮廓像圆柱就写成 cup；椅子即使与餐桌相邻也绝不是 dining_table
- dining_table 必须是宽大的连续桌面；不要把餐桌周围任何 chair 编号列进 furniture
- sofa 只允许主长沙发，不要把脚凳、单椅、靠垫或沙发分割碎片列成 sofa
- 同一件物体只出现一次
"""


def _shrink(image_b64: str, max_side: int = SURVEY_MAX_IMAGE_SIDE) -> str:
    """按最长边等比缩小，同时保留右侧分割编号的可读性。"""
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


def _inventory_text(
    scene_objects: dict[str, dict[str, Any]],
    fixed_furniture: dict[str, str] | None = None,
) -> str:
    """列出形状、颜色和 AABB 尺寸，约束模型不要混淆小食物与抱枕。"""
    lines = []
    for object_id, info in sorted(scene_objects.items(), key=lambda kv: _sort_key(kv[0])):
        shape = info.get("shape") or "Unknown"
        color = info.get("color") or "Unknown"
        if shape == "Unknown" and color == "Unknown":
            continue
        size_text = ""
        aabb = info.get("world_aabb") or {}
        low, high = aabb.get("min") or {}, aabb.get("max") or {}
        try:
            spans = (
                abs(float(high["x"]) - float(low["x"])),
                abs(float(high["y"]) - float(low["y"])),
                abs(float(high["z"]) - float(low["z"])),
            )
            if all(span > 0.0 for span in spans):
                size_text = " size_xyz=" + "x".join(f"{span:.1f}" for span in spans)
        except (KeyError, TypeError, ValueError):
            pass
        prior_text = ""
        if fixed_furniture and object_id in fixed_furniture:
            prior_text = f" FIXED_FURNITURE={fixed_furniture[object_id]} (never an item)"
        lines.append(f"  {object_id}: shape={shape} color={color}{size_text}{prior_text}")
    if not lines:
        return "（暂无结构化清单）"
    return "\n".join(lines[:SURVEY_INVENTORY_LIMIT])


def _sort_key(object_id: str) -> tuple[int, str]:
    return (0, f"{int(object_id):08d}") if object_id.isdigit() else (1, object_id)


def build_survey_messages(
    frames: list[str],
    scene_objects: dict[str, dict[str, Any]],
    fixed_furniture: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": SURVEY_PROMPT},
        {
            "type": "text",
            "text": "本房间已见过的物体（编号: 形状 颜色 AABB尺寸cm）：\n"
            + _inventory_text(scene_objects, fixed_furniture),
        },
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
    # 即使外部调用者误传了历史帧，也只提交最新的初始朝向稳定帧。
    frames = frames[-1:]
    client = getattr(agent, "tidyroom_vlm_client", None) or agent._vlm_client_for_current_task()
    if client is None:
        return 0

    messages = build_survey_messages(
        frames,
        strategy.world.scene_objects,
        strategy.world.fixed_furniture_for_prompt(),
    )
    try:
        # 比赛只有 400 秒；单次 45 秒超时后最多再试一次，不能沿用通用客户端的
        # 三轮重试把两分钟以上耗在启动盘点上。
        response = client.invoke(messages, max_retries=2)
    except Exception as exc:  # noqa: BLE001 - 勘测失败退回逐帧标注
        logger.warning("Tidy-room survey request failed: {}", exc)
        return 0

    response_text = getattr(response, "text", "") or ""
    logger.info("Tidy-room survey raw vision response: {}", response_text[:8000])
    parsed = extract_last_json_from_text(response_text)
    if isinstance(parsed, dict):
        furniture = [entry for entry in (parsed.get("furniture") or []) if isinstance(entry, dict)]
        confirmed_destinations = {
            str(entry.get("object_id")): str(entry.get("destination_type") or "")
            for entry in furniture
            if entry.get("object_id") is not None
        }
        annotations = []
        for raw_entry in parsed.get("items") or []:
            if not isinstance(raw_entry, dict):
                continue
            entry = {"object_id": raw_entry.get("object_id"), **raw_entry}
            destination_object_id = str(entry.get("destination_object_id") or "")
            destination_type = str(entry.get("destination_type") or "")
            if destination_object_id and confirmed_destinations.get(destination_object_id) != destination_type:
                logger.warning(
                    "Ignored unconfirmed survey destination object_id={} type={} for item={}; "
                    "the furniture list did not confirm the same pair",
                    destination_object_id,
                    destination_type,
                    entry.get("object_id"),
                )
                entry.pop("destination_object_id", None)
            annotations.append(entry)
        annotations += [
            {"anchor_object_id": entry.get("object_id"), "destination_type": entry.get("destination_type")}
            for entry in furniture
        ]
    elif isinstance(parsed, list):
        # 模型有时直接回一个数组；每条按自己的字段分类（有 semantic_label 是
        # 物品，只有 destination_type 是家具），不必强行套进上面的结构。
        annotations = [entry for entry in parsed if isinstance(entry, dict)]
    else:
        logger.warning("Tidy-room survey returned unparsable payload: {}", str(parsed)[:200])
        return 0
    context = getattr(agent, "_task_context", None)
    applied = strategy.world.apply_scene_annotations(annotations, context)
    # K2.6 偶尔会漏掉画面中非常明显的地面长抱枕（真实日志中 object 33
    # 约 45×16×15cm），但同时把 75×59×27cm 的脚凳写成 pillow。用窄范围
    # AABB 规则补回前者，服务端不可拾取检查继续兜底其他误标。
    applied += strategy.world.discover_geometric_pillow_targets(context)
    logger.info(
        "Tidy-room survey over {} frames: submitted={} applied={} targets={} anchors={}",
        len(frames),
        len(annotations),
        applied,
        len(strategy.world.targets),
        len(strategy.world.scene_anchors),
    )
    return applied
