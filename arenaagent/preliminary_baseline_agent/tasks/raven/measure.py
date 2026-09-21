"""量出题图里各格图形的几何参数，作为文本提示交给模型。

为什么需要：实测模型能推出规律（"缺格应与第一行第三列的五边形一致"），却分不清
候选里哪个尺寸对得上——那道题的正确答案 5 的填充面积是 3825，参照格 (1,3) 是 3744
（差 2%，同一个素材），而它先后选了 4（2398）和 8（5364）。这种比较在两百像素的
格子里靠肉眼不可靠，用像素统计几行就能量准。

三处实测踩过的坑，都在这里处理：
1. **浅灰图形会漏掉**：Q2 的浅灰图形灰度约 224，阈值取 215 时整格只剩深色图形，
   量出来的"最大轮廓"是角落里的小图形甚至格线残片。所以"非背景"的判据改成
   "比本格背景明显暗"，白底、浅灰、深灰、黑一律能取到。
2. **一格里可能有 1~4 个图形**。全部保留并按上/下、左/右或四象限
   列出槽位，避免把 Q3 的 count 规律截断成“最多两个”。
3. **形状没提取**：只给面积和宽高，模型没法把候选和"某个已给格"对应起来。
   这里用多边形逼近的顶点数与圆度给出形状名，并直接指出"与哪个已给格几乎同尺寸"。

这里**只提供测量值**，选哪个仍由模型判断；量不出可信结果时返回空串，不注入。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

CELL_MIN_AREA = 4000  # 格线矩形的 bbox 面积下限
CELL_MAX_SOLIDITY = 0.35  # 空心矩形（格线）的墨水/bbox 比例远低于实心图形
PANEL_EDGE_RATIO = 0.9  # 比这更大的组件是整图外框
CELL_INSET = 8
MATRIX_GRID = (3, 3)
OPTION_GRID = (2, 4)
MIN_CELLS_MATRIX = 7
MIN_CELLS_OPTION = 7

BACKGROUND_MARGIN = 14  # 比本格背景暗这么多就算图形
MIN_SHAPE_AREA = 120  # 小于这个的按噪点/数字笔画丢掉
SECOND_SLOT_RATIO = 0.08  # 真实多图形可以大小差异很大；格内数字已由边框切除
MAX_SHAPES_PER_CELL = 4
SAME_SIZE_TOLERANCE = 0.06  # 面积差在 6% 以内视为"同一个素材尺寸"

_NGON_NAMES = {3: "三角形", 4: "四边形", 5: "五边形", 6: "六边形", 7: "七边形"}

# 填充色的分档：出题端用"白/浅灰/中灰/深灰/黑"这几档，候选之间常常只差这一项
# （实测某张图的候选 2/4/5/7 是同一尺寸的菱形，只有填充不同），所以必须量出来。
_FILL_BANDS = ((240, "白"), (195, "浅灰"), (120, "中灰"), (45, "深灰"), (0, "黑"))


@dataclass
class ShapeStat:
    area: int
    width: int
    height: int
    shape: str = ""
    fill: str = ""
    fill_gray: int = -1
    slot: str = ""
    match: str = ""
    center_x: float = 0.0
    center_y: float = 0.0

    def text(self) -> str:
        # 只给可直接比较的量：宽高比与主轴对这类近似等轴的图形是噪声，会误导判断。
        head = f"{self.shape} " if self.shape else ""
        fill = f"{self.fill}(灰度{self.fill_gray}) " if self.fill and self.fill_gray >= 0 else ""
        slot = f"[{self.slot}]" if self.slot else ""
        body = f"面积{self.area} 宽高{self.width}x{self.height}"
        tail = f"（{self.match}）" if self.match else ""
        return f"{head}{fill}{body}{slot}{tail}"


@dataclass
class CellStat:
    box: tuple[int, int, int, int]
    shapes: list[ShapeStat] = field(default_factory=list)


def _cell_boxes(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """用"空心矩形"组件定位每个格子，返回 (x, y, w, h)。"""
    height, width = gray.shape
    ink = (gray < 200).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index][:5])
        if w * h < CELL_MIN_AREA:
            continue
        if w > width * PANEL_EDGE_RATIO and h > height * PANEL_EDGE_RATIO:
            continue  # 整图外框
        if area / float(w * h) > CELL_MAX_SOLIDITY:
            continue  # 实心图形，不是格线
        boxes.append((x, y, w, h))
    return boxes


def _classify(mask: np.ndarray) -> str:
    """按顶点数与圆度给出形状名。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ""
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area <= 0:
        return ""
    # 圆用"轮廓面积 / 外接圆面积"判：周长对锯齿很敏感，实测圆会被判成 8 边形
    (_, _), radius = cv2.minEnclosingCircle(contour)
    if radius > 0 and area / (np.pi * radius * radius) > 0.86:
        return "圆"
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    if perimeter <= 0:
        return ""
    vertices = len(cv2.approxPolyDP(hull, 0.035 * perimeter, True))
    return _NGON_NAMES.get(vertices, f"{vertices}边形" if vertices else "")


def _fill_of(values: np.ndarray) -> str:
    """按图形内部像素的中位灰度给填充分档。

    用中位数而不是均值：空心图形的边框只占内部一小圈，均值会被拉黑，中位数
    仍然落在填充色上（实测白底黑边的空心五边形要被判成"白"）。
    """
    if values.size == 0:
        return ""
    median = float(np.median(values))
    for threshold, name in _FILL_BANDS:
        if median >= threshold:
            return name
    return "黑"


def _measure_cell(gray: np.ndarray, box: tuple[int, int, int, int]) -> CellStat | None:
    x, y, w, h = box
    inner = gray[y + CELL_INSET : y + h - CELL_INSET, x + CELL_INSET : x + w - CELL_INSET]
    if inner.size == 0:
        return None
    # 背景取最亮值而不是中位数：图形占格过半时中位数会变成图形本身的颜色，
    # 掩膜直接为空（实测候选 1 那个占格 61% 的大五边形就因此量不到）。
    background = int(inner.max())
    mask = (inner.astype(np.int16) < background - BACKGROUND_MARGIN).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None
    blobs = []
    for index in range(1, count):
        area = int(stats[index][4])
        if area < MIN_SHAPE_AREA:
            continue
        blob = (labels == index).astype(np.uint8)
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        filled = np.zeros_like(blob)
        cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 1, thickness=-1)
        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(contour)
        fill_values = inner[filled.astype(bool)]
        blobs.append(
            ShapeStat(
                area=int(cv2.contourArea(contour)),
                width=int(bw),
                height=int(bh),
                shape=_classify(filled),
                fill=_fill_of(fill_values),
                fill_gray=int(round(float(np.median(fill_values)))) if fill_values.size else -1,
                center_x=float(centroids[index][0]),
                center_y=float(centroids[index][1]),
            )
        )
    if not blobs:
        return None
    blobs.sort(key=lambda stat: stat.area, reverse=True)
    kept = [blobs[0]]
    for stat in blobs[1:]:
        if stat.area >= blobs[0].area * SECOND_SLOT_RATIO:
            kept.append(stat)
    kept = kept[:MAX_SHAPES_PER_CELL]
    if len(kept) == 2:
        first, second = kept
        if abs(first.center_y - second.center_y) >= abs(first.center_x - second.center_x):
            order = sorted(kept, key=lambda stat: stat.center_y)
            order[0].slot, order[1].slot = "上", "下"
        else:
            order = sorted(kept, key=lambda stat: stat.center_x)
            order[0].slot, order[1].slot = "左", "右"
    elif len(kept) >= 3:
        middle_x = inner.shape[1] / 2.0
        middle_y = inner.shape[0] / 2.0
        for stat in kept:
            vertical = "上" if stat.center_y < middle_y else "下"
            horizontal = "左" if stat.center_x < middle_x else "右"
            stat.slot = horizontal + vertical
    else:
        kept[0].slot = ""
    return CellStat(box=box, shapes=kept)


def _layout(gray: np.ndarray) -> tuple[dict[tuple[int, int], CellStat], dict[tuple[int, int], CellStat]]:
    """返回 (矩阵格, 候选格)，键为 (行, 列)，都从 1 开始。"""
    boxes = _cell_boxes(gray)
    if len(boxes) < MIN_CELLS_MATRIX + MIN_CELLS_OPTION:
        return {}, {}
    height = gray.shape[0]
    boxes.sort(key=lambda b: (b[1] + b[3] / 2))
    centers = [b[1] + b[3] / 2 for b in boxes]
    split = height * 0.655
    option_boxes = [b for b, cy in zip(boxes, centers) if cy >= split]
    matrix_boxes = [b for b, cy in zip(boxes, centers) if cy < split]
    if len(option_boxes) < MIN_CELLS_OPTION or len(matrix_boxes) < MIN_CELLS_MATRIX:
        return {}, {}

    def bucket(box_list, rows: int, cols: int) -> dict[tuple[int, int], CellStat]:
        ys = [b[1] + b[3] / 2 for b in box_list]
        xs = [b[0] + b[2] / 2 for b in box_list]
        y0, y1 = min(ys), max(ys)
        x0, x1 = min(xs), max(xs)
        out: dict[tuple[int, int], CellStat] = {}
        for box, cy, cx in zip(box_list, ys, xs):
            row = 1 + int(round((cy - y0) / max((y1 - y0) / max(rows - 1, 1), 1)))
            col = 1 + int(round((cx - x0) / max((x1 - x0) / max(cols - 1, 1), 1)))
            row = min(max(row, 1), rows)
            col = min(max(col, 1), cols)
            cell = _measure_cell(gray, box)
            if cell is not None:
                out.setdefault((row, col), cell)
        return out

    return bucket(matrix_boxes, *MATRIX_GRID), bucket(option_boxes, *OPTION_GRID)


def _same_shape_note(
    stat: ShapeStat,
    matrix: dict[tuple[int, int], CellStat],
) -> str:
    """指出候选与哪个已给格（同形状）几乎同尺寸——这是"同一素材复制"的直接依据。

    只做同形状比较：实测把所有配对（含跨形状）列出来会把模型带偏，它去抓不相干的
    配对，三题全错。
    """
    if not stat.shape:
        return ""
    best_gap = None
    best_cell = ""
    for (row, col), cell in sorted(matrix.items()):
        for other in cell.shapes:
            if other.shape != stat.shape or not other.area:
                continue
            gap = abs(other.area - stat.area) / float(other.area)
            if best_gap is None or gap < best_gap:
                best_gap, best_cell = gap, f"矩阵({row},{col})"
    if best_gap is None:
        return ""
    if best_gap <= SAME_SIZE_TOLERANCE:
        return f"与{best_cell}的同形状格面积相差 {best_gap * 100:.0f}%，几乎同尺寸"
    return f"与最接近的同形状格{best_cell}差 {best_gap * 100:.0f}%"


def describe_shapes(panel: Image.Image, question_index: int) -> str:
    """生成给模型看的测量表；量不出来就返回空串。"""
    try:
        gray = np.asarray(panel.convert("L"))
        matrix, options = _layout(gray)
    except Exception:  # noqa: BLE001 - 测量只是辅助，失败就当没有
        return ""
    if len(options) < MIN_CELLS_OPTION or len(matrix) < MIN_CELLS_MATRIX:
        return ""

    lines = ["=== 系统已用像素统计量好这张图的每个图形（数值仅供比较，判断仍由你来做）==="]
    lines.append("候选（编号照图，【共N个】是实测 count，方括号是位置槽）：")
    for row in range(1, OPTION_GRID[0] + 1):
        for col in range(1, OPTION_GRID[1] + 1):
            cell = options.get((row, col))
            number = (row - 1) * OPTION_GRID[1] + col
            if cell is None or not cell.shapes:
                lines.append(f"  {number}. （未量到）")
                continue
            parts = []
            for stat in cell.shapes:
                stat.match = _same_shape_note(stat, matrix)
                parts.append(stat.text())
            lines.append(f"  {number}. 【共{len(cell.shapes)}个】" + " + ".join(parts))
    lines.append("矩阵（行,列）：")
    for row in range(1, MATRIX_GRID[0] + 1):
        cells = []
        for col in range(1, MATRIX_GRID[1] + 1):
            cell = matrix.get((row, col))
            if cell is None or not cell.shapes:
                continue
            parts = " + ".join(stat.text() for stat in cell.shapes)
            cells.append(f"({row},{col})【共{len(cell.shapes)}个】 {parts}")
        if cells:
            lines.append("  " + "；".join(cells))
    lines.append(
        "用法：必须先由完整行列规律确定缺格的 count、shape、fill 和 size 等级，"
        "然后再用灰度/面积/宽高在候选中精确比对。‘与某已知格最像’仅是测量信息，"
        "不能代替全局规律，也不能推翻已经确定的数量或大中小等级。"
    )
    return "\n".join(lines) + f"\n（以上是第 {question_index} 题的测量。）\n"
