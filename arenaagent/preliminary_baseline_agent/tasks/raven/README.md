# 瑞文混合求解器

瑞文任务已从“通用 VLM 决定是否调用旧模型”改为独立的混合流水线。整理房间及其他任务不会导入或调用这里的求解逻辑。

## 流水线

1. `vision.py`：OpenCV 提取前景、连通域、洞、多边形边数、填充密度、深浅、质心、包围盒、对称性、方向和象限分布。
2. `scene_graph.py`：将图元组织为节点，以及 `left_of`、`above`、`near` 关系。
3. `rules.py`：同时从完整的行和列归纳标量规律、集合运算、复制、旋转和镜像规律，并为 8 个候选打分。
4. `reasoners.py`：把每题重排成带有严格 1~8 标签的清晰图板，交给强 VLM；视觉响应缺失时才触发独立文本复核。
5. `ensemble.py`：融合规则分、旧视觉模型先验、VLM 与独立文本复核结果，逐题选择最高融合置信度答案。
6. `verifier.py`：只接受范围为 1~8 的一基候选编号，容错恢复被截断的单题 JSON，并归一化八项候选置信度。
7. `experience.py`：只把服务端明确判定正确的题目归纳为可迁移技巧；新题按规则相似度检索，不向模型提供历史答案编号或图片。

`strategy.py` 直接发出 `solve_raven` 本地动作，因此不会先消耗一次通用 Agent 的视觉调用，也不会采集无关的 3D 房间画面。
三题的 48 张子图默认只在内存中裁剪。提交后以服务端的题目状态为准：首选错误时保存证据并只复盘最可疑题；不会遍历答案组合，也不会用本地标记伪造题目已结束。

运行时采用分阶段升级：先在约亚秒级时间内提交专用 Raven 视觉网络的首选；只有服务端判定该选项未通过，才启动 CV + Rule + VLM + LLM 混合升级。这样保留未知题型的长尾能力，同时避免在常见题上预付几十秒模型时延。

## 开关

- `RAVEN_ENABLE_VLM=0`：只使用 CV、规则引擎和旧视觉模型。
- `RAVEN_ENABLE_TEXT_VERIFIER=0`：关闭独立文本复核。
- `RAVEN_SAVE_CROPS=1`：调试时把 48 张裁图写入磁盘；默认不写入。
- `RAVEN_TEXT_API_KEY` 或 `DEEPSEEK_API_KEY`：DeepSeek 独立验证器密钥。
- `RAVEN_TEXT_MODEL`：默认 `deepseek-v4-pro`。
- `RAVEN_TEXT_API_BASE`：默认 `https://api.deepseek.com`。
- `RAVEN_TEXT_TIMEOUT_SECONDS`：单次文本复核的最长等待时间，默认 35 秒。
- `RAVEN_ENABLE_EXPERIENCE=0`：关闭已验证经验的记录与检索；默认开启。
- `RAVEN_EXPERIENCE_PATH`：经验库文件位置；默认 `logs/raven_experience.json`。

文本复核不是每题必调：只有视觉结果缺失、格式不可恢复，或服务端未接受首答后复盘最可疑题时才调用。每次最多一次受时限保护的请求；没有独立 DeepSeek 密钥时保持关闭，绝不复用视觉模型自我验证。

## 本地验证

```powershell
uv run python -m unittest tests.test_raven_hybrid -v
uv run python -m unittest discover -s tests -v
```
