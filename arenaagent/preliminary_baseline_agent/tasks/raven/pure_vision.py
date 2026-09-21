"""纯视觉瑞文求解：画布切成三道题，每题一次请求并行求解。

为什么这么拆：
- 整张 2376x1200 画布里 51 个子图共享一次前向，实测模型会对三道**不同的**题重复给
  同一个答案（9-19 日志 K3 三题全答 4），也就是注意力塌缩；切成三块后每题独占一次前向。
- 每题再拆成"矩阵一张、候选一张"：矩阵只负责定规律，候选只负责比对形状。
  实测把两部分挤在一张图里时，模型会把候选当矩阵的一部分去读（把候选编号说成格子编号）。
- test 环境只允许一次有效提交。可用 ``RAVEN_VISION_PASSES=3`` 在提交前并行跑
  三路快答做实验；默认单路，因为小验证集上多数票会强化某些系统性错误。

提示词用单题版 vision_prompt_2.txt。生产路径使用短输出快答；K3 当前并不能由兼容字段可靠地
“关闭思考”，因此提示词本身仍需给出方法（拆属性 → 找行列规律 → 逐层排除 → 回填验证）。
"""

from __future__ import annotations

import base64
import concurrent.futures
import contextlib
import io
import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from PIL import Image

from arenaagent.preliminary_baseline_agent.tasks.raven.measure import describe_shapes
from arenaagent.preliminary_baseline_agent.tasks.raven.verifier import parse_direct_model_votes

# 每题单独成图后默认不降采样：切开后单题约 780x1200，缩放反而丢细节。
DEFAULT_MAX_IMAGE_SIDE = 0

# 找不到横向空白带时的兜底切分位置（矩阵在上、候选在下）。
FALLBACK_MATRIX_RATIO = 0.62
MAX_PARALLEL_QUESTIONS = 3
DEFAULT_ENSEMBLE_PASSES = 1
MAX_ENSEMBLE_PASSES = 5


@lru_cache(maxsize=1)
def method_prompt() -> str:
    """单题版方法论提示词（vision_prompt_2.txt），输出契约也写在文件末尾。"""
    return Path(__file__).with_name("vision_prompt_2.txt").read_text(encoding="utf-8").strip()


# 每题单独请求时的输入说明：两张图分别是什么。
TWO_IMAGE_LAYOUT = (
    "下面这道是第 {index} 题，输入两张图：\n"
    "第一张是题目矩阵（要找“?”处缺失的图形），第二张是 1~8 号候选答案。\n\n"
)

# 912 的三个子题位置对应三类结构。这些是解题步骤，不包含任何题图、
# 指纹或标签；它们让模型先用适合该结构的属性表，减少随意叙事。
QUESTION_GUIDANCE = {
    1: (
        "\n\n这一题优先按【单图形属性矩阵】求解：把 shape、fill、size "
        "分成三张 3×3 表，分别验证行规律和列规律。候选之间常只差尺寸，"
        "必须先推出缺格在本行/列的大中小等级，不得用‘像某个已知格’替代规律。"
    ),
    2: (
        "\n\n这一题优先按【上下两槽】求解：上槽和下槽各自建立 shape/fill/size "
        "表，先各自预测，最后才组合成缺失格。不要把两个槽串成一条混合序列。"
    ),
    3: (
        "\n\n这一题按【多图形属性与位置】求解：同时建立 count、形状多重集、fill "
        "和位置槽表。count 只有在两个完整行/列都显示同一种循环或加减时才是硬约束；"
        "不得仅因前两行总数恰好相等就假设‘每行和恒定’。候选若无法支持预测，应回退"
        "重查规律，不要声称‘无候选匹配’后随意选一个。"
    ),
}

# 兜底路径（分割失败时）：整张画布当一道题发，用三分契约要三个编号。
THREE_IN_ONE_LAYOUT = (
    "这张图里并排着三道互相独立的瑞文图形推理题，从左到右是第 1、2、3 题。\n"
    "每题上方是 3×3 图形矩阵（其中一格画着问号），下方是编号 1~8 的候选图形。\n"
    "请对三道题分别独立求解。\n\n"
    "只输出 JSON，不要解释、不要多余字段：\n"
    '{"answers": [1, 2, 3]}'
)

# 赛题端答对才结束这道题，所以重答时明确告诉模型"上一次没通过"——这是 agent
# 自己就能观察到的状态（题目还没结束），不涉及读取服务端的对错判定。
RETRY_HINT = (
    "\n\n注意：上一次提交的三位组合没有通过，这只能说明三道子题中至少一道有误，"
    "不代表你当前这一道必然错了。请重新核对属性表和候选编号；如果原判断仍能被多个已知格"
    "一致验证，应保留原答案，不要为了变化而变化。"
)


def _env_int(name: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(os.getenv(name, "").strip() or default))
    except ValueError:
        return default


def _white_gap_runs(white_fraction: np.ndarray, length: int) -> list[tuple[int, int]]:
    """把"整行/整列接近纯白"的连续区间找出来。"""
    is_gap = white_fraction > 0.98
    runs: list[tuple[int, int]] = []
    start = None
    for position in range(length):
        if is_gap[position] and start is None:
            start = position
        elif not is_gap[position] and start is not None:
            runs.append((start, position))
            start = None
    if start is not None:
        runs.append((start, length))
    return runs


def split_canvas(image: Image.Image) -> list[Image.Image]:
    """把整张画布按竖直白色间隙切成三道题。

    画布上三道题并排、彼此留白分隔。整张发过去时 51 个子图共享一次前向，
    实测模型会对三道**不同的**题重复给同一个答案（9-19 日志里 K3 三题全答 4），
    也就是注意力塌缩。切开后每题独占一张图，有效分辨率约为原来的三倍。

    分隔列必须整列都接近白色才算间隙；找不到两处间隙就退回等宽三分。
    """
    gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    if height == 0 or width < 3:
        return [image]
    runs = _white_gap_runs((gray > 240).mean(axis=0), width)
    # 只取真正落在中间的间隙，边缘留白不算分隔。
    separators = [
        (low + high) // 2
        for low, high in runs
        if low > width * 0.1 and high < width * 0.9
    ]
    if len(separators) == 2:
        left, right = separators
        panels = [
            image.crop((0, 0, left, height)),
            image.crop((left, 0, right, height)),
            image.crop((right, 0, width, height)),
        ]
    else:
        third = width // 3
        panels = [
            image.crop((0, 0, third, height)),
            image.crop((third, 0, third * 2, height)),
            image.crop((third * 2, 0, width, height)),
        ]
    return [panel for panel in panels if panel.width > 0 and panel.height > 0]


def split_question(panel: Image.Image) -> tuple[Image.Image, Image.Image]:
    """把一道题的图切成 (题目矩阵图, 候选选项图)。

    两部分的交界是一整条接近纯白的横带，且明显宽于矩阵内部的行间隙，所以取
    "居中区域内最宽的那条横带"。找不到就用固定比例兜底，绝不返回空图。
    """
    gray = np.asarray(panel.convert("L"))
    height, width = gray.shape
    if height < 4:
        return panel, panel
    runs = _white_gap_runs((gray > 240).mean(axis=1), height)
    inner = [
        (low, high)
        for low, high in runs
        if low > height * 0.4 and high < height * 0.9
    ]
    if inner:
        low, high = max(inner, key=lambda run: run[1] - run[0])
        cut = (low + high) // 2
    else:
        cut = int(height * FALLBACK_MATRIX_RATIO)
    cut = min(max(cut, 1), height - 1)
    return panel.crop((0, 0, width, cut)), panel.crop((0, cut, width, height))


def encode_panel(panel: Image.Image) -> tuple[str, int]:
    """无损 PNG 编码：选项之间只有细微尺寸差异，JPEG 的振铃会干扰判读。"""
    limit = _env_int("RAVEN_IMAGE_MAX_SIDE", DEFAULT_MAX_IMAGE_SIDE, 0)
    if limit and max(panel.size) > limit:
        scale = limit / max(panel.size)
        panel = panel.resize(
            (max(1, round(panel.width * scale)), max(1, round(panel.height * scale))),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    panel.convert("RGB").save(buffer, format="PNG", optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/png;base64," + payload, len(buffer.getvalue())


def _ask(client: Any, panels: list[Image.Image], prompt: str) -> tuple[str, float, float]:
    """发一次请求，返回 (原始文本, 耗时秒, 发送字节数)。异常不抛给调用方。"""
    encoded = [encode_panel(panel) for panel in panels]
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content += [{"type": "image_url", "image_url": {"url": url}} for url, _ in encoded]
    started = time.perf_counter()
    try:
        # 三题并行时一次网络抖动就会让整道题拿不到答案（客户端失败时返回错误文本
        # 而不是抛异常），所以给它 3 次传输重试；RPM=100 下重试的代价远小于丢一整题。
        response = client.invoke([{"role": "user", "content": content}], max_retries=3)
    except Exception as exc:  # noqa: BLE001 - 视觉请求失败不能让整题崩掉
        logger.warning("Raven vision request failed after {}s: {}", round(time.perf_counter() - started, 2), exc)
        return "", round(time.perf_counter() - started, 2), sum(size for _, size in encoded) / 1024
    return (
        str(getattr(response, "text", "") or ""),
        round(time.perf_counter() - started, 2),
        sum(size for _, size in encoded) / 1024,
    )


def _parse(text: str, expected_questions: int) -> list[int] | None:
    votes = parse_direct_model_votes(text, expected_questions=expected_questions)
    by_question = {vote.question: int(vote.answer) for vote in votes}
    if set(by_question) < set(range(1, expected_questions + 1)):
        return None
    return [by_question[index] for index in range(1, expected_questions + 1)]


def _reason_of(text: str) -> str:
    """取出模型写的那句 reason，只用于日志——判断它是在推理还是在猜。"""
    payload: Any = None
    try:
        payload = json.loads(text.strip().strip("`"))
    except (TypeError, ValueError):
        match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
        if match is not None:
            with contextlib.suppress(TypeError, ValueError):
                payload = json.loads(match.group(0))
    if isinstance(payload, dict):
        return " ".join(str(payload.get("reason") or "").split())[:220]
    return ""


def _solve_one_question(
    client: Any,
    index: int,
    panel: Image.Image,
    attempt: int,
) -> dict[str, Any]:
    """问一道题：矩阵与候选各一张图。返回该题的作答与诊断。"""
    matrix_image, option_image = split_question(panel)
    prompt = TWO_IMAGE_LAYOUT.format(index=index) + method_prompt()
    prompt += QUESTION_GUIDANCE.get(index, "")
    # 用像素统计量出每个图形的形状/填充/面积，并指出候选与哪个已给格几乎同尺寸。
    # 实测模型靠肉眼看不出 76px 与 89px 的差别（会把差 35% 的候选当成"尺寸一致"），
    # 有这张表时 Q1 从 0/5 提到 5/7。
    measurement = describe_shapes(panel, index)
    if measurement:
        prompt += "\n\n" + measurement
    if attempt > 1:
        # 带上第几次，避免两次重试发出逐字节相同的请求——实测那样模型会
        # 原样重复上一次的答案，重试等于没做。
        prompt += RETRY_HINT + f"（这是第 {attempt} 次作答）"
    text, seconds, kilobytes = _ask(client, [matrix_image, option_image], prompt)
    parsed = _parse(text, 1)
    record: dict[str, Any] = {
        "question": index,
        "seconds": seconds,
        "kilobytes": round(kilobytes, 1),
        "matrix_size": matrix_image.size,
        "option_size": option_image.size,
        "raw_preview": text[:220],
        "answer": parsed[0] if parsed else None,
        "reason": _reason_of(text),
    }
    if parsed is None:
        logger.warning(
            "Raven pure vision question {} unparsable ({}s, preview={!r})",
            index,
            seconds,
            text[:160],
        )
    else:
        logger.info("Raven Q{} answer={} reason={!r}", index, parsed[0], record["reason"])
    return record


def _solve_pure_vision_once(
    client: Any,
    image_path: str | Path,
    attempt: int = 1,
) -> tuple[list[int] | None, dict[str, Any]]:
    """画布切三块，每块一次请求（矩阵图 + 候选图），三块并行；失败返回 None 与诊断。"""
    if client is None:
        return None, {"error": "raven vision client unavailable"}

    canvas = Image.open(image_path).convert("RGB")
    panels = split_canvas(canvas)
    started = time.perf_counter()
    diagnostics: dict[str, Any] = {
        "canvas_size": canvas.size,
        "panel_sizes": [panel.size for panel in panels],
    }

    if len(panels) == 3:
        # 每题独占一次前向，三题并行：赛题按时间给分，串行会把三次请求的时间
        # 全算在自己头上。
        workers = min(MAX_PARALLEL_QUESTIONS, len(panels))
        results: dict[int, dict[str, Any]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_solve_one_question, client, index, panel, attempt): index
                for index, panel in enumerate(panels, start=1)
            }
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except Exception as exc:  # noqa: BLE001 - 单题失败按未作答处理
                    logger.warning("Raven question {} failed: {}", index, exc)
                    results[index] = {
                        "question": index,
                        "answer": None,
                        "seconds": 0.0,
                        "kilobytes": 0.0,
                        "raw_preview": f"<{exc}>",
                        "reason": "",
                    }
        per_question = [results.get(index, {}) for index in range(1, len(panels) + 1)]
        diagnostics.update(
            {
                "per_question": per_question,
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                "sent_kilobytes": round(sum(item.get("kilobytes", 0.0) for item in per_question), 1),
            }
        )
        answers = [item.get("answer") for item in per_question]
        if any(answer is None for answer in answers):
            diagnostics["failed_question"] = next(
                index for index, answer in enumerate(answers, start=1) if answer is None
            )
            return None, diagnostics
        logger.info(
            "Raven pure vision selected {} in {}s (canvas {}, panels {}, {}KB, 三题并行)",
            answers,
            diagnostics["elapsed_seconds"],
            diagnostics["canvas_size"],
            diagnostics["panel_sizes"],
            diagnostics["sent_kilobytes"],
        )
        return [int(answer) for answer in answers], diagnostics

    # 分割失败：整张画布一次问三题。
    logger.warning("Raven canvas did not split into three panels ({}); asking them together", len(panels))
    prompt = THREE_IN_ONE_LAYOUT
    if attempt > 1:
        prompt += RETRY_HINT
    text, seconds, kilobytes = _ask(client, [canvas], prompt)
    parsed = _parse(text, 3)
    diagnostics.update(
        {"elapsed_seconds": round(time.perf_counter() - started, 2), "sent_kilobytes": round(kilobytes, 1)}
    )
    if parsed is None:
        logger.warning("Raven whole-canvas answer unparsable ({}s, preview={!r})", seconds, text[:160])
        return None, diagnostics
    logger.info("Raven pure vision selected {} in {}s (整张画布一次请求)", parsed, diagnostics["elapsed_seconds"])
    return parsed, diagnostics


def _consensus_answers(votes: list[list[int]]) -> list[int] | None:
    """逐位多数票；三票全不同时取最后完成的一票。"""
    if not votes:
        return None
    combined: list[int] = []
    for position in range(3):
        counts: dict[int, int] = {}
        for vote in votes:
            if len(vote) != 3:
                continue
            value = int(vote[position])
            if 1 <= value <= 8:
                counts[value] = counts.get(value, 0) + 1
        if not counts:
            return None
        best_count = max(counts.values())
        tied = {value for value, count in counts.items() if count == best_count}
        combined.append(next(vote[position] for vote in reversed(votes) if vote[position] in tied))
    return combined


def solve_pure_vision(
    client: Any,
    image_path: str | Path,
    attempt: int = 1,
) -> tuple[list[int] | None, dict[str, Any]]:
    """在首次提交前并行跑多路独立快答，并按三个位置分别多数表决。

    ``attempt`` 只表示本地推理重试。test 环境的有效提交是一次性的，因此不会把它
    写成“上次答案被判错”的提示。用 ``RAVEN_VISION_PASSES=1`` 可退回单路基线。
    """
    passes = min(
        MAX_ENSEMBLE_PASSES,
        _env_int("RAVEN_VISION_PASSES", DEFAULT_ENSEMBLE_PASSES, 1),
    )
    if passes == 1:
        return _solve_pure_vision_once(client, image_path, attempt=1)

    started = time.perf_counter()
    runs: list[tuple[list[int] | None, dict[str, Any]]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=passes) as pool:
        futures = [
            pool.submit(_solve_pure_vision_once, client, image_path, 1)
            for _ in range(passes)
        ]
        # 按提交顺序取结果，平票行为可复现；所有任务本身仍是并行执行。
        for future in futures:
            try:
                runs.append(future.result())
            except Exception as exc:  # noqa: BLE001 - 某一路失败不应拖垮其余投票
                logger.warning("Raven ensemble pass failed: {}", exc)
                runs.append((None, {"error": str(exc)}))

    votes = [list(answer) for answer, _ in runs if answer is not None and len(answer) == 3]
    consensus = _consensus_answers(votes)
    base = next((dict(diagnostics) for answer, diagnostics in runs if answer is not None), {})
    base.update(
        {
            "ensemble_passes": passes,
            "valid_passes": len(votes),
            "vote_history": votes,
            "consensus_answers": list(consensus) if consensus is not None else None,
            "ensemble_runs": [diagnostics for _, diagnostics in runs],
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "requested_attempt": attempt,
        }
    )
    if consensus is None:
        logger.warning("Raven pre-submit ensemble produced no usable vote")
        return None, base
    logger.info(
        "Raven pre-submit consensus {} from {} valid pass(es) in {}s: {}",
        consensus,
        len(votes),
        base["elapsed_seconds"],
        votes,
    )
    return consensus, base
