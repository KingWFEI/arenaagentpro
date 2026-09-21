"""把 measure.py 在某一张瑞文题图上的全部中间结果导出来人工核对。

打印：组件检测（哪些被当格子、哪些被过滤掉及原因）、每格量测的轮廓、
行列表格文本；并把检测框与轮廓画到图上存盘。

    uv run python scripts/dump_raven_measure.py [题图路径] [题号]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from arenaagent.preliminary_baseline_agent.tasks.raven import measure  # noqa: E402
from arenaagent.preliminary_baseline_agent.tasks.raven.pure_vision import (  # noqa: E402
    split_canvas,
)

DEFAULT_IMAGE = "logs/raven_task_data/62f357be81_82lyvig5/task_data_001.png"
OUT_DIR = Path("logs/raven_debug")


def dump_components(gray: np.ndarray, label: str) -> None:
    height, width = gray.shape
    ink = (gray < 200).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    print(f"\n--- {label}：连通组件 {count - 1} 个（阈值 {200}）---")
    rows = []
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index][:5])
        solidity = area / float(w * h) if w * h else 0.0
        reasons = []
        if w * h < measure.CELL_MIN_AREA:
            reasons.append(f"bbox面积{w * h}<{measure.CELL_MIN_AREA}")
        if w > width * measure.PANEL_EDGE_RATIO and h > height * measure.PANEL_EDGE_RATIO:
            reasons.append("整图外框")
        if solidity > measure.CELL_MAX_SOLIDITY:
            reasons.append(f"实心 solidity={solidity:.2f}>{measure.CELL_MAX_SOLIDITY}")
        verdict = "→ 当格子" if not reasons else " 丢弃(" + ",".join(reasons) + ")"
        rows.append((y, x, w, h, area, solidity, verdict))
    for y, x, w, h, area, solidity, verdict in sorted(rows):
        print(
            f"  x={x:4d} y={y:4d} w={w:4d} h={h:4d} 像素数={area:6d} solidity={solidity:.3f} {verdict}"
        )


def dump_cells(gray: np.ndarray, boxes: list[tuple[int, int, int, int]], label: str) -> None:
    print(f"\n--- {label}：逐格量测（inset={measure.CELL_INSET}）---")
    for x, y, w, h in sorted(boxes, key=lambda b: (b[1], b[0])):
        inner = gray[y + measure.CELL_INSET : y + h - measure.CELL_INSET,
                     x + measure.CELL_INSET : x + w - measure.CELL_INSET]
        mask = (inner < 200).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"  格子 x={x:4d} y={y:4d} w={w:4d} h={h:4d} 框面积={w * h:6d}  轮廓{len(contours)}个:", end=" ")
        if not contours:
            print("无")
            continue
        parts = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:4]:
            area = int(cv2.contourArea(contour))
            bx, by, bw, bh = cv2.boundingRect(contour)
            cx, cy = bx + bw / 2, by + bh / 2
            parts.append(
                f"外接{area}(bbox {bw}x{bh}, 中心偏({cx - inner.shape[1] / 2:+.0f},{cy - inner.shape[0] / 2:+.0f}))"
            )
        cell = measure._measure_cell(gray, (x, y, w, h))
        chosen = " + ".join(s.text() for s in cell.shapes) if cell else "未采纳"
        print(" ".join(parts) + f"  => 采纳: {chosen}")


def annotate(panel: Image.Image, boxes, stat_map, option_map, out: Path) -> None:
    canvas = panel.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for x, y, w, h in boxes:
        draw.rectangle([x, y, x + w, y + h], outline=(0, 160, 255), width=3)
    for (row, col), cell in stat_map.items():
        text = " + ".join(stat.text() for stat in cell.shapes)
        draw.text((4, 4 + 18 * (row - 1) * 3), f"M({row},{col}) {text}", fill=(200, 0, 0))
    for (row, col), cell in option_map.items():
        number = (row - 1) * 4 + col
        text = " + ".join(stat.text() for stat in cell.shapes)
        draw.text((6, canvas.height - 60 + 16 * (row - 1)), f"#{number} {text}", fill=(0, 120, 0))
    canvas.save(out)


def cell_montage(panel: Image.Image, boxes, out: Path) -> None:
    """把每个检测到的格子裁出来、放大 1.6×，按位置拼成一张图，便于肉眼核对框对不对。"""
    if not boxes:
        return
    scale = 1.6
    tiles = []
    for x, y, w, h in sorted(boxes, key=lambda b: (b[1], b[0])):
        crop = panel.crop((x, y, x + w, y + h))
        tiles.append(crop.resize((int(w * scale), int(h * scale)), Image.Resampling.NEAREST))
    width = max(t.width for t in tiles)
    rows = []
    current, current_w = [], 0
    for tile in tiles:
        if current and current_w + tile.width > width * 1.05:
            rows.append(current)
            current, current_w = [], 0
        current.append(tile)
        current_w += tile.width + 6
    if current:
        rows.append(current)
    height = sum(max(t.height for t in row) + 6 for row in rows)
    sheet = Image.new("RGB", (width + 6, height), (255, 255, 255))
    y = 0
    for row in rows:
        x = 0
        for tile in row:
            sheet.paste(tile, (x, y))
            x += tile.width + 6
        y += max(t.height for t in row) + 6
    sheet.save(out)


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    index = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    canvas = Image.open(path).convert("RGB")
    panels = split_canvas(canvas)
    print(f"题图 {path}  {canvas.size}  切成 {len(panels)} 块：{[p.size for p in panels]}")
    panel = panels[index - 1]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.save(OUT_DIR / f"q{index}_panel.png")

    gray = np.asarray(panel.convert("L"))
    dump_components(gray, f"第{index}题")
    boxes = measure._cell_boxes(gray)
    print(f"\n被当作格子的一共 {len(boxes)} 个")
    dump_cells(gray, boxes, f"第{index}题")
    matrix, options = measure._layout(gray)
    print(f"\n矩阵格 {len(matrix)} 个，候选格 {len(options)} 个")
    print("\n=== 注入提示词的表格文本 ===")
    print(measure.describe_shapes(panel, index))

    all_boxes = []
    for key in list(options) + list(matrix):
        pass
    all_boxes = boxes
    annotate(panel, all_boxes, matrix, options, OUT_DIR / f"q{index}_annotated.png")
    cell_montage(panel, boxes, OUT_DIR / f"q{index}_cells.png")
    print(f"\n已保存：{OUT_DIR}/q{index}_annotated.png, {OUT_DIR}/q{index}_cells.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
