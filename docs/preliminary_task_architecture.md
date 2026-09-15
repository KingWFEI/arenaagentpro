# 初赛五任务独立开发结构

## 总体结构

`PreliminaryBaselineAgent` 现在只做路由和生命周期衔接，公共的相机感知、VLM 调用、动作解析及 TongSim 接口仍由 `VLMAgent` 负责。五类任务的代码和提示词分别位于：

~~~text
arenaagent/preliminary_baseline_agent/
├── preliminary_baseline_agent.py   # 稳定入口，不在日常实验中频繁修改
├── task_registry.py                 # task_type 到策略类的映射
├── task_runtime.py                  # 五类任务共享的上下文数据结构
└── tasks/
    ├── base.py                      # 策略接口
    ├── tidyroom/                    # 整理房间（状态机、规划、几何校验、类别映射）
    ├── jigsaw/                      # 拼图
    ├── counting/                    # 计数
    ├── npc/                         # NPC 问答
    └── raven/                       # 瑞文推理
~~~

每个任务目录都有 `strategy.py` 和 `prompt.txt`。整理房间任务采用本地数据主导的模块化结构：

- `world_model.py`：跨视角保存目标、家具和全局 AABB。
- `scanner.py`：四方向本地扫描，不调用 VLM。
- `scheduler.py`：根据距离、可见性和失败次数选择目标。
- `planner.py`：计算接近点、座面和放置点。
- `geometry.py`：判断地板边界、障碍物和 AABB。
- `verifier.py`：验证手持状态和最终放置位置。
- `recovery.py`：分类失败、轮换重试方案并控制升级阈值。
- `vlm_policy.py`：只有语义不明确或本地恢复耗尽时才请求 VLM。

只优化某一任务时，优先只改该任务目录；这样不同成员提交的改动不会集中冲突在一个 JSON 或入口类中。

## 策略生命周期

每个策略都继承 `TaskStrategy`，可按需覆盖以下方法：

- `reset(subject)`：新题开始时清空该任务自己的状态。
- `before_step(subject, task_response)`：每轮感知前更新计数器或处理任务反馈。
- `observe(context)`：读取本轮可见物体、手中物体、可移动物体和历史动作。
- `enrich_prompt(variables, context)`：向通用 prompt 注入该任务提示和状态。
- `validate_action(action, context)`：动作发给 TongSim 前校验或改写。
- `after_action(action, result, context)`：动作执行后更新任务进度。
- `state_for_prompt()`：只返回短小、可序列化且确实有助于下一步决策的状态。

不要在任务策略中重写 gRPC、客户端初始化或通用动作接口。需要增加所有任务都能使用的新能力时，才修改 `VLMAgent`；需要修改单题决策时，只修改对应任务目录。

## 三人并行建议

- 成员 A：`tasks/tidyroom/`，重点做目标清单、放置区域选择、完成条件与失败恢复。
- 成员 B：`tasks/jigsaw/`，重点做空缺坐标、旋转估计、放置复核。
- 成员 C：`tasks/counting/`、`tasks/npc/`、`tasks/raven/`，三个任务共享代码较少，但单项改动量通常小于前两项。

公共文件 `preliminary_baseline_agent.py`、`task_registry.py`、`task_runtime.py` 与 `tasks/base.py` 由一人维护，其他成员提出接口需求后再统一修改。每人的实验记录至少包含模型名、任务、随机种子或运行批次、总分、成功数、平均步数、失败原因。

## 本地验证

在 `arenaagentpro` 目录执行：

~~~powershell
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q arenaagent/preliminary_baseline_agent
~~~

这两项不需要启动比赛客户端或消耗模型额度。通过后再按原命令进行端到端评分：

~~~powershell
uv run arenaagent --agent_name preliminary_baseline_agent --config config.toml --vlm_model VLMGPT5Config --run_times 1
~~~

旧的 `prompts/task_spec_prompt.json` 保留用于兼容，但新的 Agent 已从五个任务目录各自加载 `prompt.txt`。
