# 瑞文求解器（纯视觉）

一条路径：**画布按留白切成三道题 → 每题单独一次 K3 请求 → 直接提交三位答案**。

## 为什么拆图、为什么每题一次请求（关键）

整张画布 2376×1200 上有 3 道题 × (9 格矩阵 + 8 个选项) = 51 个子图，每格只有约
180×120 像素。实测模型在这种输入上会**注意力塌缩**：

- 2026-09-19 那次：K3 对三道**不同的**题给出的视觉答案是 `[4, 4, 4]`；
- 2026-09-20 六次作答：第 1 题五次答 "4"、第 3 题五次答 "1"——在重复读同几个位置。

所以切成三张 792×1200 的题图（每格约 215×215 像素），并且**每题独占一次前向**——
三张图放同一次请求里仍会让三题互相抢注意力。账户当前是组织级 RPM=100，
每题一次请求（一轮 3 次）不需要额外间隔；若退回 RPM=3 档位，需给
`_RAVEN_MIN_CALL_INTERVAL_SECONDS` 加回间隔。

## 提示词：单题版 `vision_prompt_2.txt`

`thinking` 是关的，模型没有内部推理空间——所以提示词本身就是方法，必须把
"拆属性 → 找行列规律 → 逐层排除 → 回填验证"写清楚。三点关键约束：

- **大小是硬约束**：候选里常有 3~4 个形状/填充完全相同、只差尺寸的选项（实测那道题
  的正确答案是 582，模型答 482，选项 1/4/5/8 全是黑五边形）。提示词要求先预测
  缺格的 size，再按**相对占格比例**（图形/单元格）比较，而不是凭肉眼印象。
- **多图形要拆槽位**：本题第 2、3 题的每个格子就是上下两个图形，不拆开推不出规律。
- **编号照图上数字读**：两行四列排布时第二行第一个是 5 不是 1。
- 输出为一个 JSON：`{"reason": "一句话说明用了哪条规律", "answers": [5], "confidence": 0.9}`。

（旧的整张画布版 `vision_prompt.txt` 保留未用；混合求解器走 `RAVEN_PURE_VISION=0` 时会用到。）

## 实现

- `pure_vision.py`
  - `split_canvas()`：按整列接近纯白的竖直间隙切三块；找不到两处间隙就等宽三分。
  - `encode_panel()`：无损 PNG（选项之间只有细微尺寸差异，JPEG 振铃会干扰判读）。
  - `solve_pure_vision()`：**每题一次请求**，每次只带一块题图、只问一个编号；
    任一道解析失败就整体报失败，绝不猜。诊断记录 `canvas_size` / `panel_sizes` /
    `per_question`（每题耗时与预览）/ 总耗时。
- `arenaagent/vlm_agent/raven_skill.py` 的 `handle()`：默认走上面这条；设
  `RAVEN_PURE_VISION=0` 可切回旧的混合求解器（只为对比，它在 train 上调过，test 下不可靠）。
- `preliminary_baseline_agent._run_raven_subject_safely()`：赛题端**答对才结束**这道题，
  所以答错就整题重做（最多 3 轮，每轮 3 次请求），提交前先确认解析出了三个编号。

## 开关

- `RAVEN_PURE_VISION`：默认 `1`（纯视觉）；设 `0` 用混合求解器。
- `RAVEN_IMAGE_MAX_SIDE`：题图最长边上限，默认 0（不缩放）。

## 关键日志

- `Raven pure vision selected [a, b, c] in X.XXs (canvas WxH, panels [...], NKB)`：
  调优主要看这一行（三个数字是三题各自的答案）。
- `Raven pure vision answer unparsable`：模型没给出可解析的三位答案，本题不提交。
- `Raven subject index N settled after K attempt(s)` / `still open after attempt K`：
  这题过了 / 在重解。

## 本地验证

```powershell
uv run python -m unittest tests.test_raven_hybrid -v
uv run python -m unittest discover -s tests -v
```
