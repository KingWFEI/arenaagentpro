# 瑞文（PreliminaryRavenTask）交接文档

> 记录 2026-09-19 ~ 09-20 这轮在 `competition-preliminary-raven-task` 上做的事：
> 实测出来的赛题机制、现在的答题流程、改过的文件、踩过的坑（都带日志证据），以及还没做完的部分。
> **所有改动仍未提交**（清单见文末）。

---

## 0. 一句话结论

准确率是当前唯一的瓶颈：一轮 10 个 subject（26 次提交）**首次尝试 0/10 全错**，只有 2 个 subject 靠第 2、3 次重试答对。
流程侧的 bug 已经修完（预取取错题图、答对不记分、网络抖动丢题），所以下一轮同水平的回答应该能真的拿到分；
但要提分必须解决"每题约 60~100%、三道全对约 20~40%"的准确率问题——现在**已经有 18 道题的验证集**支撑这件事。

---

## 1. 赛题机制（都是实测出来的，不是看文档猜的）

### 1.1 一次运行 = 10 个 subject，一张图 = 三道题

- `arena_offline/config.toml`：`use_static_subjects = true`、`num_subjects = 5`（但 flow 实际跑 **10** 个）、`task_duration_limit = 1800`。
- `[flow] wait_after_first_agent_secs = 15.0`：agent 连上后，赛题端要等 15 秒才"通知开始答题"。
- 图是 2376×1200 的 PNG：横向并排三道题，每题上 3×3 矩阵（一格是"?"）+ 下 2×4 候选（编号 1~8）。
- 题面要求"按题目顺序组成三位数"，即答案为 `[Q1, Q2, Q3]`。

### 1.2 答案键是固定的，而且 10 个 slot 只用了 6 道题

多轮（09-19 13:30、09-19 15:54、09-20 15:10、09-20 17:24）逐字一致：

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 键 | 582 | 528 | 582 | 687 | 254 | 254 | 272 | 861 | 582 | 272 |
| 题图 sha256(前12) | 94f58bf30711 | 23093182d997 | 94f58bf30711 | 6a1edf4cb3f3 | e03d58ff74c6 | e03d58ff74c6 | e7043310d3e8 | 2acda712fd0d | 94f58bf30711 | e7043310d3e8 |

**同键 ⇒ 同图 ⇒ 同答案**（6 道题：582 占 3 个 slot，254 和 272 各占 2 个）。题库自洽，没有"同图不同键"。

### 1.3 计分

- **三道全对**才算这道题完成；答对立刻结算，分数随作答速度变（同一个键 582 出现过 **97.15 / 95.65 / 91.15**）。
- 答错不结束，可以继续作答；一直不结束则约 **400 秒**后超时结算（`score = 0`）。
- 答错有惩罚项（`wrong_answer_penalty_score`，可从二进制字符串里看到）。

### 1.4 赛题端的评测时机（关键，踩过坑）

赛题端**要等所有 agent 都停手**才跑评测，且评测时要求 agent 仍在 `EVALUATING` 状态。
如果 agent 先断开，它会报错并**跳过这个 subject 的评分**（连 `correct_answer` 都不写）。详见 §4.2。

### 1.5 读答案键的正确姿势

- ✅ `arena_offline/logs/arena_*.log`：每个 subject 结算时打印 `current_evaluate_result`（含 `correct_answer` / `answer` / `score`），按 "第几轮 + 第几个 subject" 对齐最可靠。
- ❌ `eval_res.<时间戳>.json`：里面的内容是**被替换掉的旧记录**，时间戳是新记录落盘的时刻，按它对齐会错开一条。

---

## 2. 现在的答题流程

```
raven_skill.handle()
  └─ RAVEN_PURE_VISION 默认 1 → 纯视觉路径（=0 回旧的混合求解器）
       └─ pure_vision.solve_pure_vision(client, 题图路径, attempt)

solve_pure_vision：
  1. split_canvas      按"整列接近纯白"的竖直间隙切成 3 块（792×1200），找不到就等宽三分
  2. 每题并行（ThreadPoolExecutor(3)，每题一次请求，总耗时 = 最慢那一次）
       ├─ split_question  按"整行接近纯白"的横向带切出 矩阵图(792×746) + 候选图(792×454)
       ├─ measure.describe_shapes(panel)  像素测量表（形状/填充/面积/槽位/同尺寸匹配）
       ├─ 提示词 = 第几题说明 + vision_prompt_2.txt + 测量表（+ 重试时追加"这是第 N 次作答"）
       └─ K3（kimi-k3，thinking 关闭，30s 超时，传输重试 3 次）
  3. 解析 {"reason","answers":[n],"confidence"} —— 只认长度为 1 的 answers 数组
     任何一题解析不出 → 整题放弃、不提交（避免把错误文本当答案）
  4. 返回 [a,b,c] → agent 提交 → response.answer_right
       答对：subject 立即结算；等 session 进终态（约 16s，兜底 25s）再重连下一题
       答错：等 12s 未结算 → 重新求解（最多 3 次）→ 都不对就留给服务端 400s 超时
```

关键常量（`preliminary_baseline_agent.py`）：
`_RAVEN_MAX_ATTEMPTS = 3`、`_RAVEN_SUBJECT_SETTLE_WINDOW_SECONDS = 12.0`、`_RAVEN_MIN_CALL_INTERVAL_SECONDS = 0.0`

### 2.1 像素测量表（`measure.py`，三题共约 53ms）

| 步骤 | 做法 |
|---|---|
| 定位格子 | 连通组件里找"空心矩形"（bbox 面积 ≥4000 且 solidity ≤0.35）→ 矩阵 8 格 + 候选 8 格（"?"格没边框，天然不出现） |
| 分区 | 按中心 y 与图高 ×0.655 分矩阵区/候选区，再按行列中心分桶；**候选编号 = (行-1)×4 + 列** |
| 量图形 | 格内以"比本格最亮值暗 14"为掩膜 → 连通块 → 取最大两块（第二块 ≥25% 才算，标 `[上]/[下]`）→ 填洞取外边界 → 轮廓面积 / 宽高 / 形状（顶点数+圆度）/ **填充**（内部灰度中位数分五档：白·浅灰·中灰·深灰·黑） |
| 同尺寸匹配 | 每个候选找"形状相同、面积最接近"的已给格，给差值百分比；≤6% 标"几乎同尺寸" |

输出是一段中文表格，例如：

```
5. 五边形 黑 面积3877 宽高76x76（与矩阵(1,3)的同形状格面积相差 3%，几乎同尺寸）
8. 五边形 黑 面积5443 宽高89x89（与最接近的同形状格矩阵(2,1)差 35%）
```

### 2.2 提示词

`arenaagent/preliminary_baseline_agent/tasks/raven/vision_prompt_2.txt`（5832 字符 / 499 行）= **当前生效**的那份。
`vision_prompt.txt`（14276 字符）是旧的，只在 `reasoners.py` 的混合求解器里用（默认不走）。
注意：`method_prompt()` 带 `@lru_cache(maxsize=1)`，**改完必须重启 agent 进程才生效**。

---

## 3. 改过 / 新增的文件

| 文件 | 状态 | 说明 |
|---|---|---|
| `tasks/raven/pure_vision.py` | 新增 | 纯视觉求解器：切画布、切两图、三题并行、解析、诊断、测量表注入 |
| `tasks/raven/measure.py` | 新增 | 像素特征提取（形状/填充/面积/槽位/同尺寸匹配），fail-open |
| `tasks/raven/vision_prompt_2.txt` | 新增 | 单题版提示词（用户原稿 + 我补的【两张图】【像素统计】【先看第一行与第三行…】【同形状候选也可能只差朝向】【尺寸要和已有的某个格对齐】【编号必须照图上读】+ §11 checklist 两项） |
| `preliminary_baseline_agent.py` | 改 | raven 快速路径；**预取修复**（raven 不再用 init 预取的 subject）；配对落盘 `_record_raven_pair` / `_record_raven_subject_image` |
| `agent_base.py` | 改 | `session_poll_seconds=0.2`；**`settle_grace_seconds` 3 → 25**（见 §4.2）；去掉连接后固定 `sleep(2)` |
| `vlm_agent/raven_skill.py` | 改 | 纯视觉开关 `RAVEN_PURE_VISION`（默认 1）；`max_retries` 传输重试 |
| `tasks/raven/vision_client.py` | 改 | K3 `max_tokens` 256 → 512 |
| `builder.py` | 改 | 重连退避 5s → `ARENA_RECONNECT_DELAY_SECONDS`（默认 0.5） |
| `tests/test_raven_hybrid.py` | 改 | 纯视觉路径、两图请求、并行、测量表注入的回归测试（`_FakeVisionClient` 按题号作答，线程安全） |

新增脚本（都在 `scripts/`）：

| 脚本 | 用途 |
|---|---|
| `probe_raven_prompt.py` | **离线跑生产路径**（不起仿真）：`[题图] [第几次]` 单图；`--all` 跑配对表里所有图 |
| `map_raven_keys.py` | 把 agent 落盘记录 / 历史日志与 arena 日志里的键连接，产出 `logs/raven_key_pairs.json` |
| `raven_pair_report.py` | 出完整配对表 `logs/raven_pair_table.md`，并把答案同步给探针 |
| `dump_raven_measure.py` | 单题全流程 + 把检测框画到图上（`logs/raven_debug/`） |
| `dump_raven_shapes.py` | 形状/旋转属性原型（旋转对齐用 IoU 扫角度，尚未进生产） |
| `audit_raven_measure.py` | 全库审计：每题每格提到几个图形、有没有"未量到"、有没有把编号当图形 |
| `probe_arena_state.py` | 直接问 arena `is_ready_for_agent / get_num_subjects / get_current_subject_index` |

---

## 4. 踩过的坑（每条都有日志证据）

### 4.1 预取拿到"上一题的图" ⇒ 解的图和判分的题不是同一道

`init()` 在 agent 刚连上时就 `get_subject` 预取，而 `_run_subject` **优先用这份预取**；
赛题端要等 `wait_after_first_agent_secs`(15s) 才派题，这个窗口里取到的可能还是上一题的图。

```
09-20 14:26:41.235  arena: Starting subject 2/10
09-20 14:26:46.479  agent: get agent spawn info
       ~14:26:47  agent 预取 subject            ← 就在这里取图
09-20 14:27:07.590  arena: All agents connected  ← 等满 15 秒才通知开始
```

**症状**：同一张 `94f58bf30711` 被记录成 slot 1（键 582）和 slot 2（键 528）的题图，看起来像"题库坏了"——
其实是我们拿旧图去答新题，那些 subject 无论怎么推理都不可能对。

**修法**：`_run_subject` 的 raven 分支改成不用预取，在 `is_ready_for_agent` 通过之后再 `get_subject` 一次。
（用 09-20 17:24 那轮验证：slot 2 变成了它自己的图 `23093182d997`，slot 1/3/9 同图同键。）

### 4.2 提前 3 秒重连 ⇒ 赛题端跳过评分，答对的题记 0 分

```
18:02:06.362  attempt 2/3 提交 → answer_right: True
18:02:06.366  Subject already settled; giving the arena 3.0s before reconnecting
18:02:09.391  Reconnecting without waiting for the terminal session status
18:02:09.393  Closing Arena gRPC channel              ← 我们断开
18:02:10.264  arena: All agents finished answering. Evaluating subjects.
18:02:10.264  ERROR Agent adc77c8bfe is not in EVALUATING state during evaluation
18:02:10.265  arena: Starting subject 7/10            ← 没有 current_evaluate_result，没记分
```

当时 `settle_grace_seconds=3`（我为省 13 秒加的"结算后提前重连"）。09-19 那些 95+ 的成绩正是**加这个优化之前**跑出来的。

**修法**：默认改成 **25 秒**（正常路径是等 session 进终态，实测约 16 秒；25 秒只是兜底）。
环境变量 `ARENA_SETTLE_GRACE_SECONDS` 可覆盖。
**验证方法**：下一轮看 arena 日志里答对的 subject 有没有写出 `current_evaluate_result` 和 90 多分的 `score`。

### 4.3 客户端失败会返回错误文本而不是抛异常 ⇒ 一次抖动丢一整题

`Client.invoke` 在重试用尽后返回 `_ERROR_RESPONSE`（"error occurred when sending message to VLA..."），
而 `_ask` 原来传 `max_retries=1` = 只试一次。
**修法**：`max_retries=3`；解析失败时不提交（原逻辑已保证）。

### 4.4 整张画布一次问三题 ⇒ 注意力塌缩

9-19 日志里 K3 对三道**不同的**题重复给同一个数字（三题全答 4）。
**修法**：切三块 + 每题单独一次请求；后来又发现"矩阵+候选挤一张图"时模型会把候选编号当格子编号读，
于是再拆成两张图（矩阵一张、候选一张）。

### 4.5 提取器的三个硬伤

1. **阈值 215 漏浅灰图形**：Q2 有一格的浅灰填充灰度约 224 → 整格只剩深色图形，量出来的"最大轮廓"是编号笔画或角块
   （曾出现候选 1/2/4/6/8 全是 `面积1885 宽高53x53`、矩阵 (2,2) 是 `123 宽高24x24` 这种垃圾）。
   改成"比本格最亮值暗 14"。
2. **背景取中位数**：图形占格过半时中位数就是图形本身，掩膜为空（候选 1 那个 134px 大五边形直接"未量到"）。改成取最亮值。
3. **没提取形状/填充**：只给面积宽高，模型无法把候选和"某个已给格"对应起来；
   另一张图（`e7043310d3e8`）的候选 2/4/5/7 是**同一尺寸的菱形**（都 140x139、面积 10031），只有填充不同，
   没有填充这一项根本无法区分。补上形状名与填充五档。

### 4.6 单测把 mock 数据写进了真实 manifest

`test_raven_submits_once_when_the_subject_settles` 和 `test_raven_unsolved_subject_is_re_solved_not_enumerated`
直接调 `_run_raven_subject_safely`，把 `subject_index=4` / 答案 `[6,8,7]` `[7,8,7]` 写进了真实的
`logs/raven_subject_pairs.jsonl`（表现为文件里出现陌生的 agent id）。
**修法**：`_record_raven_pair` 开头加"没连上 arena 就跳过"（`self.connected`）的门；脏记录已删。

### 4.7 （历史遗留）盲枚举

更早的 train 环境实现用服务端 `answer_right` 反馈枚举 512 个三元组，本地分数虚高到 95+，
但这在 test 环境无效也不能算"会做题"。**已废弃**，现在答错只用新的一次视觉请求重做，不枚举。

---

## 5. 数据

### 5.1 准确率（离线探针，同一张图重复跑）

| 配置 | Q1 | Q2 | Q3 | 三道全对 |
|---|---|---|---|---|
| 无提取器（两图并行） | 0/5 | 3/5 | 2/5 | 0/5 |
| 有提取器（测量表） | 6/10 | 10/10 | 7/10 | 5/10 |

同一配置方差极大（7 次里 3/3 出现 4 次、0/3 出现 1 次），所以**单次结果不能作为判断依据**。
固定的失败模式：
- **Q1 错** → 模型推"第三行是第一行同批形状的旋转放大版"，按"大"去选 8 或 4；它**读到了**测量表里"候选 5 差 3%"仍然不采信。
- **Q3 错** → 把第三列读成"每列边数固定，缺格是四边形" → 选 6。

### 5.2 实跑（09-20 17:24 那轮，10 个 subject / 26 次提交）

- 首次尝试全错（0/10）；只有 slot 6（第 2 次）与 slot 8（第 3 次）答对 → **2/10**；
- 这两个答对的还因为 §4.2 被跳过评分，记 0 分。

### 5.3 验证集（本次产出的最大资产）

`logs/raven_subject_images/<sha12>.png`（永久保存，不会被临时目录清理）+ `logs/raven_key_pairs.json`：

| 题图 | 键 | 正确答案 |
|---|---|---|
| `94f58bf30711` | 582 | 5、8、2 |
| `23093182d997` | 528 | 5、2、8 |
| `6a1edf4cb3f3` | 687 | 6、8、7 |
| `e03d58ff74c6` | 254 | 2、5、4 |
| `e7043310d3e8` | 272 | 2、7、2 |
| `2acda712fd0d` | 861 | 8、6、1 |

**共 18 道题**，答案全部已知（其中 254/861 两个键是用"被服务端接受的答案"反推的，其余来自 arena 日志）。

---

## 6. 怎么跑 / 怎么复现

```bash
# 0. 环境（顺序不能颠倒；stop 只能在 start_test 窗口 Ctrl+C，强杀会连带杀掉 UE 客户端）
#    UE 客户端 → start_test.bat train --task competition-preliminary-raven-task

# 1. 起 agent：run_times 必须 ≥ subject 数（10），否则后面的 subject 没人答
cd D:/Contest/Windows/基准Agent/baseline_agent/baseline_agent/arenaagentpro
uv run arenaagent --agent_name preliminary_baseline_agent --config config.toml \
    --vlm_model VLMGPT5Config --run_times 10
# 单个进程就够：多 agent 只是重复答同一道题，还会把 subject 提前结束掉

# 2. 采集配对（跑完一轮后）
uv run python scripts/map_raven_keys.py       # 落盘记录 + arena 日志 → logs/raven_key_pairs.json
uv run python scripts/raven_pair_report.py    # → logs/raven_pair_table.md（含推断出的键）

# 3. 离线验证（不碰仿真环境）
uv run python scripts/probe_raven_prompt.py --all          # 18 道题的逐题命中率
uv run python scripts/probe_raven_prompt.py <题图> 3        # 单图重复测稳定性

# 4. 看提取器到底提了什么
uv run python scripts/audit_raven_measure.py               # 全库审计
uv run python scripts/dump_raven_measure.py <题图> 1        # 单题 + 标框图

# 5. 查 arena 现在能不能接 agent
uv run python scripts/probe_arena_state.py
```

注意：
- 直接调用 agent 必须**先切到仓库目录**（`.env` 按当前目录查找）。
- 改了 `vision_prompt_2.txt` 要**重启 agent 进程**（`lru_cache`）。
- agent 的配对记录只在"真连上 arena"时落盘（单测里不会污染）。

---

## 7. 没做完 + 建议的下一步

1. **跨图基线还没测**（2 分钟、54 次 K3 调用）：`probe_raven_prompt.py --all`。
   现在所有"有效/无效"的判断都建立在 1 张图的 3 道题上，先有基线才谈优化。
2. **"同素材尺寸"还没做成确定性取舍**：模型只判"缺格该是什么形状/几个槽位"（这个它很稳），
   系统再按测量表在**同形状候选中**按"与某个已给格面积差最小"定死编号。
   已知两道题上成立（Q1 差 3%、Q3 差 3%，而模型爱选的差 35%），18 道题正好够验证它是否普遍成立、会不会误伤 Q2。
3. **验证 §4.2 的评分修复**：下一轮看答对的 subject 有没有真的写出 90+ 的 `score`。
4. **未纳入生产**：`dump_raven_shapes.py` 里的旋转角对齐（IoU 扫角度）没进提示词——
   实测那道题的所有五边形候选旋转角相同（都 ≈ -9° mod 72），旋转从来不是判别维度，尺寸才是。
5. **老图**：本地还有 6 张 09-13/09-14 的题图（`215a693b1cea` 等），但那几轮的 arena 日志已不存在，
   拿不到答案键，不能当验证样本。

---

## 8. 未提交的改动

```
 M arenaagent/agent_base.py                                         （settle grace 25s、轮询 0.2s）
 M arenaagent/builder.py                                            （重连退避）
 M arenaagent/preliminary_baseline_agent/preliminary_baseline_agent.py  （预取修复、配对落盘、raven 快速路径）
 M arenaagent/preliminary_baseline_agent/tasks/raven/{README.md,solver.py,strategy.py,verifier.py,vision_client.py,vision_prompt.txt}
 M arenaagent/vlm_agent/raven_skill.py
 M tests/test_raven_hybrid.py
?? arenaagent/preliminary_baseline_agent/tasks/raven/{measure.py,pure_vision.py,vision_prompt_2.txt}
?? scripts/{probe_raven_prompt.py,map_raven_keys.py,raven_pair_report.py,dump_raven_measure.py,
            dump_raven_shapes.py,audit_raven_measure.py,probe_arena_state.py}
```

此前测试：`uv run python -m pytest tests/ -q` → **234 passed**。

---

## 9. 09-20 晚：面向未知 test 题的泛化优化

本轮明确不将训练图指纹或已知答案写入生产代码。六张带标签图只用于离线误差分析。

### 9.1 新基线

| 配置 | Q1 | Q2 | Q3 | 合计 | 三题全对 |
|---|---:|---:|---:|---:|---:|
| 交接时 499 行特例提示词 | 2/6 | 6/6 | 1/6 | 9/18 | 0/6 |
| 通用属性提示词 + 完整多图形测量 | 5/6 | 6/6 | 4/6 | **15/18** | **3/6** |

最终配置单张三题并行约 6~11 秒。

### 9.2 这轮修复

- 删掉原提示词中针对单道旧题的“第三行是第一行旋转/放大版”、
  “复制已知格比趋势外推可靠”等特例，改成属性表、最小可重复规律和全局回填。
- 按结构提示 Q1 的单图形属性、Q2 的上下槽、Q3 的多图形/位置，但不含题图或答案先验。
- `measure.py` 从每格最多 2 个图形扩展到 4 个，增加左/右/四象限槽位、显式 count 和原始灰度。
- 重试提示改为“三位码至少一位错”，不再误导每个子题强制换答案。
- 前两次保留独立推理；第三次对三轮结果做逐位多数票，平票取最新复核。
- 修复 `probe_raven_prompt.py` 的 Windows GBK 打印中断，并让命令行“第几次”真正传入生产求解器。

K3 `reasoning_effort=high/low` 都做了受控探针：推理质量更高，但单题 60~90 秒仍可超时，
不适合当前按时间计分的生产路径，因此没有启用。

测试：`uv run python -m pytest tests/ -q` → **237 passed, 16 subtests passed**。

---

## 10. 09-20 test 日志复核：有效提交是一次性的

用户提供的 test 环境日志共 10 个 subject，统计结果为：

- `selected=10`、`submitted=10`；
- 每题都只有 `attempt 1/3 submitted`，之后立即 settled；
- `answer_right=True` 出现 10 次，但 test 不公开真实答案，因此这里只能解释为“提交被接受”，
  不能当作答对反馈；
- `attempt 2=0`、`attempt 3=0`。

所以 §1.3、§2、§7 中基于 train 行为写的“答错可继续重答”不适用于最终 test。
生产流程已经改为：最多三次只恢复**尚未提交时**的网络/解析失败；一旦得到合法三位答案，
只提交一次，之后保持连接等待 arena 终态，绝不根据 `answer_right` 或短轮询结果二次提交。

### 10.1 提交前集成实验

测试过三路 K3 快答并行（9 个请求）：墙钟约 8.3 秒，说明并发预算足够；但两次完整验证结果
波动明显：一次为 15/18、三位全对 4/6，另一次为 14/18、三位全对 2/6。错误规律会被多数票强化，
三票全不同时的决胜规则也不可靠。因此：

- 默认 `RAVEN_VISION_PASSES=1`，保持原单路生产基线；
- `RAVEN_VISION_PASSES=3` 仅作为显式实验开关，不默认启用；
- K3 低思考在难题上可用，但单题会达到 60~137 秒并可能耗尽输出仍无答案，未上线；
- 尝试降低采样温度时接口明确拒绝：`kimi-k3` 只允许 `temperature=0.6`。

本轮测试：`uv run python -m pytest tests/ -q` → **238 passed, 16 subtests passed**。
