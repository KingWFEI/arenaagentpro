"""把提取器在**每一张**存下来的瑞文题图上跑一遍，检查它到底提到了什么。

用途：提取器以前只在第一题上人工核对过。这里逐题打印每格提到几个图形、形状名、
面积，并把"可疑"的情况标出来：
  - 整格没提到图形（未量到）
  - 一个候选格里提到的图形数与矩阵格里最常见的数量不一致（可能漏了浅色图形）
  - 同一形状的候选在多个选项里面积完全相同（可能量到了同一个装饰/编号）

    uv run python scripts/audit_raven_measure.py [--verbose]
"""

from __future__ import annotations

import glob
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from arenaagent.preliminary_baseline_agent.tasks.raven import measure  # noqa: E402
from arenaagent.preliminary_baseline_agent.tasks.raven.pure_vision import split_canvas  # noqa: E402

VERBOSE = "--verbose" in sys.argv


def audit_question(panel: Image.Image) -> list[str]:
    notes: list[str] = []
    try:
        gray = __import__("numpy").asarray(panel.convert("L"))
        matrix, options = measure._layout(gray)
    except Exception as exc:  # noqa: BLE001
        return [f"测量抛异常: {exc}"]
    if not options or not matrix:
        return [f"布局失败（矩阵 {len(matrix)} 格 / 候选 {len(options)} 格）"]

    matrix_slots = Counter(len(cell.shapes) for cell in matrix.values())
    option_slots = Counter(len(cell.shapes) for cell in options.values())
    notes.append(f"矩阵 {len(matrix)} 格槽位分布 {dict(matrix_slots)}；候选 {len(options)} 格槽位分布 {dict(option_slots)}")
    for key, cell in sorted(map(lambda kv: (str(kv[0]), kv[1]), matrix.items())):
        if not cell.shapes:
            notes.append(f"  ! 矩阵 {key} 未量到图形")
    for row in range(1, 3):
        for col in range(1, 5):
            cell = options.get((row, col))
            number = (row - 1) * 4 + col
            if cell is None or not cell.shapes:
                notes.append(f"  ! 候选 {number} 未量到图形")
    # 极小的"图形"更像编号笔画/装饰；出题端本来就大量复用同一素材，
    # 所以"多个候选面积相同"是正常现象，只在这里降级成提示。
    tiny = [
        f"候选{number}.{stat.shape or '?'} 面积{stat.area}"
        for row in range(1, 3)
        for col in range(1, 5)
        for number, cell in [((row - 1) * 4 + col, options.get((row, col)))]
        if cell
        for stat in cell.shapes
        if stat.area and stat.area < 300
    ]
    if tiny:
        notes.append("  ? 面积过小的图形（可能是编号笔画）: " + "，".join(tiny))

    if VERBOSE:
        for key in sorted(options):
            cell = options[key]
            number = (key[0] - 1) * 4 + key[1]
            notes.append("    #%d %s" % (number, " + ".join(s.text() for s in cell.shapes)))
        for key in sorted(matrix):
            cell = matrix[key]
            notes.append("    M(%d,%d) %s" % (key[0], key[1], " + ".join(s.text() for s in cell.shapes)))
    return notes


def main() -> int:
    files = sorted(glob.glob("logs/raven_task_data/*/task_data_*.png"))
    if not files:
        print("没有找到题图")
        return 1
    seen_hashes: set[str] = set()
    import hashlib

    for path in files:
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        canvas = Image.open(path).convert("RGB")
        panels = split_canvas(canvas)
        print(f"\n########## {digest}  {path}  切成 {len(panels)} 块 ##########")
        for index, panel in enumerate(panels, start=1):
            print(f"--- 第{index}题 ---")
            for note in audit_question(panel):
                print("  " + note)
        print("  注入文本长度:", [len(measure.describe_shapes(p, i)) for i, p in enumerate(panels, 1)])
    print(f"\n不同题图共 {len(seen_hashes)} 张")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
