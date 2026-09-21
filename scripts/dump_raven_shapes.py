"""在 Q1 上做一次"完整属性提取"原型：形状、灰度、尺寸（按格子归一化）、旋转角。

当前 measure.py 只注入 面积/宽高，而这道题的规律是"第三行=第一行同列图形旋转"，
所以关键属性是**旋转角**与**形状**，两者现在都没进提示词。这里把它们量出来，
并给出每个候选相对"第一行同列图形"的旋转差——用来验证"旋转对齐"能不能直接定出答案。

    uv run python scripts/dump_raven_shapes.py [题图路径] [题号]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from arenaagent.preliminary_baseline_agent.tasks.raven import measure  # noqa: E402
from arenaagent.preliminary_baseline_agent.tasks.raven.pure_vision import split_canvas  # noqa: E402

DEFAULT_IMAGE = "logs/raven_task_data/62f357be81_82lyvig5/task_data_001.png"
NORMALIZED_AREA = 12000.0  # 旋转对齐前把图形缩放到同一面积，消除尺寸差异
CANVAS = 220
SWEEP = range(-90, 91, 2)


def shape_mask(gray: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray | None:
    """返回格子内最大图形的**填充**掩膜（轮廓外边界内部全填满），全图坐标。"""
    x, y, w, h = box
    inner = gray[y + measure.CELL_INSET : y + h - measure.CELL_INSET,
                 x + measure.CELL_INSET : x + w - measure.CELL_INSET]
    if inner.size == 0:
        return None
    mask = (inner.astype(np.int16) < int(inner.max()) - measure.BACKGROUND_MARGIN).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None
    # 用"离格心最近"的大块：单图形题里图形居中，而角标数字/装饰贴边
    cx, cy = inner.shape[1] / 2, inner.shape[0] / 2
    best, best_score = None, -1e9
    for index in range(1, count):
        lx, ly, lw, lh, larea = (int(v) for v in stats[index][:5])
        if larea < 80:
            continue
        score = larea - 3.0 * (abs(lx + lw / 2 - cx) + abs(ly + lh / 2 - cy))
        if score > best_score:
            best, best_score = index, score
    if best is None:
        return None
    blob = (labels == best).astype(np.uint8)
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    filled = np.zeros_like(blob)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 1, thickness=-1)
    full = np.zeros_like(gray, dtype=np.uint8)
    full[y + measure.CELL_INSET : y + measure.CELL_INSET + filled.shape[0],
         x + measure.CELL_INSET : x + measure.CELL_INSET + filled.shape[1]] = filled
    return full


def classify(mask: np.ndarray) -> tuple[str, int, float]:
    """返回 (形状名, 顶点数, 圆度)。顶点数用 approxPolyDP 估。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter else 0.0
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    vertices = len(approx)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area else 0.0
    if circularity > 0.87 and vertices > 6:
        name = "圆"
    else:
        name = {3: "三角形", 4: "四边形", 5: "五边形", 6: "六边形"}.get(vertices, f"{vertices}边形")
        if name == "四边形":
            rect = cv2.minAreaRect(contour)
            (_, _), (rw, rh), angle = rect
            # 长宽比接近 1 且相对外接框明显转动 → 菱形
            name = "四边形"
    if solidity < 0.9:
        name += "(凹)"
    return name, vertices, circularity


def normalize(mask: np.ndarray) -> np.ndarray | None:
    """把图形缩放到统一面积、按质心居中，放到固定画布上，消除尺寸差异。"""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    scale = np.sqrt(NORMALIZED_AREA / len(xs))
    small = cv2.resize(
        mask, (max(1, int(mask.shape[1] * scale)), max(1, int(mask.shape[0] * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    ys, xs = np.nonzero(small)
    if len(xs) == 0:
        return None
    cx, cy = xs.mean(), ys.mean()
    out = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    x0 = int(round(CANVAS / 2 - cx))
    y0 = int(round(CANVAS / 2 - cy))
    for sy, sx in zip(ys, xs):
        ty, tx = sy + y0, sx + x0
        if 0 <= ty < CANVAS and 0 <= tx < CANVAS:
            out[ty, tx] = 1
    return out


def relative_rotation(ref: np.ndarray, cand: np.ndarray) -> tuple[float, float]:
    """扫角度求 IoU 最大处，返回 (相对旋转角, 峰值IoU)。"""
    best_angle, best_iou = 0.0, 0.0
    for angle in SWEEP:
        matrix = cv2.getRotationMatrix2D((CANVAS / 2, CANVAS / 2), angle, 1.0)
        rotated = cv2.warpAffine(cand, matrix, (CANVAS, CANVAS), flags=cv2.INTER_NEAREST)
        union = np.logical_or(ref, rotated).sum()
        if union == 0:
            continue
        iou = np.logical_and(ref, rotated).sum() / union
        if iou > best_iou:
            best_angle, best_iou = float(angle), float(iou)
    return best_angle, best_iou


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    index = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    panel = split_canvas(Image.open(path).convert("RGB"))[index - 1]
    gray = np.asarray(panel.convert("L"))

    boxes = measure._cell_boxes(gray)
    matrix, options = measure._layout(gray)

    def describe(tag: str, box, stat) -> dict:
        mask = shape_mask(gray, box)
        row: dict = {"tag": tag, "box": box, "stat": stat}
        if mask is None:
            row["shape"] = "未测到"
            return row
        name, vertices, circularity = classify(mask)
        values = gray[mask.astype(bool)]
        row.update(
            shape=name,
            vertices=vertices,
            circularity=round(circularity, 2),
            gray_median=int(np.median(values)) if values.size else -1,
            mask=mask,
        )
        return row

    def find_box(maps: dict, key: tuple[int, int]):
        """从 layout 的格子映射反查原始 box：用中心点匹配。"""
        stat = maps.get(key)
        if stat is None:
            return None
        return None  # 由下面的重建替代

    # layout() 丢掉了原始 box，这里按同样的分桶规则重建 (行,列)->box
    height = gray.shape[0]
    boxes_sorted = sorted(boxes, key=lambda b: (b[1] + b[3] / 2))
    split = height * 0.655
    matrix_boxes = [b for b in boxes_sorted if b[1] + b[3] / 2 < split]
    option_boxes = [b for b in boxes_sorted if b[1] + b[3] / 2 >= split]

    def assign(box_list, rows, cols):
        ys = [b[1] + b[3] / 2 for b in box_list]
        xs = [b[0] + b[2] / 2 for b in box_list]
        y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
        out = {}
        for box, cy, cx in zip(box_list, ys, xs):
            r = 1 + int(round((cy - y0) / max((y1 - y0) / max(rows - 1, 1), 1)))
            c = 1 + int(round((cx - x0) / max((x1 - x0) / max(cols - 1, 1), 1)))
            out.setdefault((min(max(r, 1), rows), min(max(c, 1), cols)), box)
        return out

    matrix_assign = assign(matrix_boxes, 3, 3)
    option_assign = assign(option_boxes, 2, 4)

    print(f"矩阵格 {sorted(matrix_assign)}")
    print(f"候选格 {sorted(option_assign)}\n")

    matrix_rows = {}
    for key in sorted(matrix_assign):
        row = describe(f"矩阵({key[0]},{key[1]})", matrix_assign[key], matrix.get(key))
        matrix_rows[key] = row
    option_rows = {}
    for key in sorted(option_assign):
        number = (key[0] - 1) * 4 + key[1]
        row = describe(f"候选{number}", option_assign[key], options.get(key))
        option_rows[number] = row

    print("=== 提取到的属性 ===")
    print(f"{'位置':<10}{'形状':<12}{'灰度中位':>8}{'面积':>8}{'宽高':>12}{'占格比':>8}")
    for key in sorted(matrix_rows):
        row = matrix_rows[key]
        stat = row.get("stat")
        box = row["box"]
        inner = box[2] - 2 * measure.CELL_INSET
        ratio = f"{stat.width / inner:.2f}" if stat else "-"
        print(
            f"{row['tag']:<10}{row.get('shape', '-'):<12}"
            f"{row.get('gray_median', -1):>8}"
            f"{(f'{stat.area}' if stat else '-'):>8}"
            f"{(f'{stat.width}x{stat.height}' if stat else '-'):>12}{ratio:>8}"
        )
    for number in sorted(option_rows):
        row = option_rows[number]
        stat = row.get("stat")
        box = row["box"]
        inner = box[2] - 2 * measure.CELL_INSET
        ratio = f"{stat.width / inner:.2f}" if stat else "-"
        print(
            f"{row['tag']:<10}{row.get('shape', '-'):<12}"
            f"{row.get('gray_median', -1):>8}"
            f"{(f'{stat.area}' if stat else '-'):>8}"
            f"{(f'{stat.width}x{stat.height}' if stat else '-'):>12}{ratio:>8}"
        )

    print("\n=== 相对第一行同列图形的旋转（负=逆时针，正=顺时针）===")
    refs = {}
    for key, row in matrix_rows.items():
        if row.get("mask") is not None and key[0] == 1:
            refs[key] = normalize(row["mask"])
    for key in sorted(matrix_rows):
        row = matrix_rows[key]
        if key[0] == 1 or row.get("mask") is None:
            continue
        ref = refs.get((1, key[1]))
        if ref is None:
            continue
        angle, iou = relative_rotation(ref, normalize(row["mask"]))
        print(f"  {row['tag']} 相对 矩阵(1,{key[1]})：旋转 {angle:+.0f}°  对齐度 {iou:.2f}")
    # 候选五边形的旋转：分别与矩阵(1,3) 和矩阵(3,1)/(3,2) 的旋转量比较
    ref_13 = refs.get((1, 3))
    if ref_13 is not None:
        for number in sorted(option_rows):
            row = option_rows[number]
            if row.get("mask") is None or "五边形" not in str(row.get("shape", "")):
                continue
            angle, iou = relative_rotation(ref_13, normalize(row["mask"]))
            print(f"  {row['tag']} 相对 矩阵(1,3) 五边形：旋转 {angle:+.0f}°  对齐度 {iou:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
