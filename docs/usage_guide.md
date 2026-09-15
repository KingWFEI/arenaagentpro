# ArenaAgentPro 上手指南

这份文档是写给第一次接手这个项目的同学的。不用上来就把每一行代码读懂，也不用一开始就搞明白 gRPC、VLM、仿真接口分别是什么。先把项目跑起来，再知道出问题的时候去哪查日志，差不多就够你开始动手了。

简单说，这个项目在做什么？让一个 Agent 进到仿真环境里，获取题目、感知环境、调用动作接口，然后想办法把任务做完。

## 目录

- [ArenaAgentPro 上手指南](#arenaagentpro-上手指南)
  - [目录](#目录)
  - [1. 先把项目跑起来](#1-先把项目跑起来)
    - [1.1 你需要准备什么](#11-你需要准备什么)
    - [1.2 第一次拿到项目](#12-第一次拿到项目)
    - [1.3 配置文件](#13-配置文件)
    - [1.4 配大模型 key](#14-配大模型-key)
    - [1.5 真正开跑](#15-真正开跑)
  - [2. Agent 是怎么工作的](#2-agent-是怎么工作的)
    - [2.1 从命令行到 Agent](#21-从命令行到-agent)
    - [2.2 AgentBase 在管什么](#22-agentbase-在管什么)
    - [2.3 VLMAgent 干的事](#23-vlmagent-干的事)
  - [3. 感知层：Agent 是怎么看世界的](#3-感知层agent-是怎么看世界的)
  - [4. 动作层：Agent 是怎么真正做事的](#4-动作层agent-是怎么真正做事的)
  - [5. Prompt 才是 Agent 的说明书](#5-prompt-才是-agent-的说明书)
  - [6. 怎么把成绩提上去（重点）](#6-怎么把成绩提上去重点)
    - [6.1 先把失败原因定位清楚](#61-先把失败原因定位清楚)
    - [6.2 优先改 task prompt](#62-优先改-task-prompt)
    - [6.3 通用 prompt 别越写越乱](#63-通用-prompt-别越写越乱)
    - [6.4 正确的调试节奏](#64-正确的调试节奏)
  - [7. 跟任务系统相关的接口说明](#7-跟任务系统相关的接口说明)
    - [7.1 连接和状态](#71-连接和状态)
    - [7.2 拿题目和反馈](#72-拿题目和反馈)
    - [7.3 提交动作和答案](#73-提交动作和答案)
  - [8. 看代码的建议顺序](#8-看代码的建议顺序)

## 1. 先把项目跑起来

### 1.1 你需要准备什么

跑这个项目，下面这些东西缺一不可：

- Python 3.12 及以上。
- `uv`，用来管依赖和跑命令。装一次就行。
- 一个能用的大模型 API key。
- 启动任务系统，默认监听 `127.0.0.1:50051`和 `127.0.0.1:50060`。

或者你直接和你的agent说:
> 帮我在这台机器上安装python环境和uv环境，具体版本参考项目的pyproject.toml文件。uv安装完成后，按照README和docs/usage_guide.md 帮我更新uv环境。

任务系统服务没起来的话，Agent 是怎么也跑不通的。这个项目不是一个能单机小程序，它必须和外部的评测系统、仿真环境一起配合才能动起来。

### 1.2 第一次拿到项目

在项目根目录依次执行：

```bash
uv sync
copy config.toml.example config.toml
```

如果后面运行时报某个 gRPC 模块找不到，重新生成一次 proto 代码就好：

```bash
uv run scripts/generate_pb2.py
```

正常 `uv sync` 一次就把依赖装齐了。之后所有命令都从 `uv run` 走，不要直接 `python xxx`，会用错环境。

### 1.3 配置文件

项目读的是 `config.toml`，常见样子大概长这样：

```toml
[final_baseline_agent]
name = "final_baseline_agent"
sleep_between_steps = 2.0
log_dir = "logs"
tongsim_server_endpoint = "127.0.0.1:50060"

[grpc]
endpoint = "127.0.0.1:50051"
```

几个字段值得拎出来说一下：

- `name`：现在要跑的是哪个 Agent。
- `sleep_between_steps`：每一步动作之后停多久。调太短，仿真还没反应过来就发下一条；调太长，整局任务会拖得很慢。`2.0` 是个比较稳的初始值。
- `log_dir`：日志、prompt 和截图都会落到这里。出问题第一件事就是来这里看。
- `tongsim_server_endpoint`：TongSim Proxy 的地址。**保持默认**
- `[grpc].endpoint`：任务系统的地址。**保持默认**

### 1.4 配大模型 key

模型配置类都在 `arenaagent/vlm_agent/vlm_config.py`。能接的渠道很多：OpenAI 兼容接口、Azure、DashScope、智谱、火山 Ark 等等。

key 一律放环境变量里，不要写死在代码或 toml 里。代码哪天 push 上去就是事故。

PowerShell 里这样设：

```powershell
$env:OPENAI_API_KEY="your_api_key"
$env:OPENAI_API_BASE="https://your_api_base/v1"
```

各家常用的变量名：

- OpenAI 兼容接口：`OPENAI_API_KEY`、`OPENAI_API_BASE`
- Azure OpenAI：`AZURE_OPENAI_API_KEY`、`AZURE_OPENAI_ENDPOINT`
- DashScope：`DASHSCOPE_API_KEY`
- 智谱：`ZHIPU_API_KEY`
- 火山 Ark：`ARK_API_KEY`

不知道项目里有哪些模型可选，直接列出来：

```powershell
uv run arenaagent --agent_name vlm_agent --get_vlm_model
```

### 1.5 真正开跑

确认UE客户端和任务系统服务都开着，然后：

决赛baseline：

```bash
uv run arenaagent --agent_name final_baseline_agent --config config.toml --vlm_model VLMGPT5Config
```

初赛 baseline：

```bash
uv run arenaagent --agent_name preliminary_baseline_agent --config config.toml --vlm_model VLMGPT5Config --run_times 5
```

跑起来以后别急着动代码。先把这两个目录翻一遍：

- `logs/`：完整运行日志。
- `logs/prompts/`：每一轮发给模型的 prompt，以及 Agent 当时看到的那张图。

很多时候 Agent 行为奇怪，根本不是代码问题，是模型看到的图就模糊不清，或者 prompt 把任务说糊涂了。先别甩锅给模型，先看它到底收到了什么，然后再决定怎么修改 :)。

## 2. Agent 是怎么工作的

可以把 Agent 想成一个被丢进仿真世界的机器人。每一轮它都在做四件事：看一眼环境、读一下题目、问大模型该干啥、把模型说的事真正做出来。就这么循环到任务结束。

代码里这条主线是这样串起来的：

```text
builder.py 把 Agent 创出来
AgentBase 负责跟任务系统打交道
VLMAgent 负责获取第一视角图片、写 prompt、问模型、执行动作
TongSimInterface 负责跟仿真环境真正对接
```

### 2.1 从命令行到 Agent

你敲的 `uv run arenaagent ...`，入口写在 `pyproject.toml`：

```toml
[project.scripts]
arenaagent = "arenaagent.builder:main"
```

也就是命令最终落到 `arenaagent/builder.py` 的 `main()`。Agent 类是用 `@Register("名字")` 装饰器注册进来的：

```python
@Register("vlm_agent")
class VLMAgent(AgentBase):
    ...
```

所以 `--agent_name vlm_agent` 这个参数，本质就是去注册表里翻名字叫 `vlm_agent` 的那个类，把它 new 出来。

### 2.2 AgentBase 在管什么

`arenaagent/agent_base.py` 是公共底座。它不关心你用什么模型，也不关心你怎么决策，它只管一件事：跟任务系统对接。

记住三个方法就够了：

```python
def init(self, opt: dict):
    ...

def run_step(self, subject: str, task_response: dict) -> dict:
    ...

def deinit(self):
    ...
```

名字基本就是意思：

- `init()`：任务开始前的准备工作，比如创建角色、初始化模型 client。
- `run_step()`：每一步真正的决策点，Agent 的脑子就在这里。
- `deinit()`：任务结束后扫尾。

`AgentBase.run()` 把标准流程都写好了：

```text
连接任务系统
拿到角色出生信息
初始化 Agent
等任务 ready
循环每一道题：
    取题目
    取反馈
    调 run_step 拿动作
    把动作交给任务系统
结束后发起评分
断开连接
```

新手基本不用动 `run()`。真正会改的就是 `VLMAgent.run_step()`、prompt 文件，以及少量动作处理。

### 2.3 VLMAgent 干的事

`arenaagent/vlm_agent/vlm_agent.py` 里的 `VLMAgent`，想法其实很朴素：

```text
看到啥 -> 告诉模型 -> 模型给动作 -> 执行 -> 记下来 -> 下一轮接着干
```

具体一轮里它会：

1. 从 TongSim 拿第一视角图和语义分割图。
2. 整理当前能看到的物体：颜色、形状、位置、ID。
3. 把当前任务、上一步结果、手里有没有东西、NPC 有没有说话这些信息收集起来。
4. 全部塞进 prompt 模板。
5. 调模型。
6. 从模型回复里把 JSON 解析出来。
7. 按 JSON 里的 `action` 调对应的动作。
8. 把这一步记进历史，下一轮还会带上。

整个 Agent 的"智能"就靠这个循环撑着。它不是预先写死每个任务怎么做，而是靠"观察 + prompt + 动作接口"一步步往前推。

## 3. 感知层：Agent 是怎么看世界的

感知相关的代码主要在：

- `arenaagent/semantic_mapper.py`
- `arenaagent/tongsim_interface.py`

`VLMAgent.run_step()` 里最关键的一句：

```python
self.semantic_mapper.get_perception_from_camera(is_save=True)
```

这一行会同时拿到三样东西：

- `b64_image`：发给模型的图。左边是第一视角 RGB，右边是叠了数字 ID 的语义分割图。
- `visible_objects`：可见物体的 ID 映射表。
- `visible_objects_info`：可见物体的文字描述，比如颜色、形状、位置、包围盒。

这里有一个**特别容易踩的坑**：模型调用动作时用的物体 ID，必须是右边语义分割图上标出来的那个数字。不要让模型自己编 ID，也不要直接拿 TongSim 底层的原始 ID。代码会自动帮你把数字 ID 翻译回真正的 object id。这一层翻译没走对，后面动作怎么调都白搭。

常用的感知接口：

- `acquire_first_person_image()`：第一视角 RGB。
- `acquire_first_person_segmantic_image()`：语义分割图。
- `fetch_first_person_visible_objects()`：当前视野里有哪些物体。
- `get_object_basic_info()`：查物体颜色、形状、放置位置。
- `get_object_world_aabb()`：查物体 3D 包围盒。
- `get_object_in_hand()`：手里有没有拿东西。

模型老是抓错东西，第一件事别改 prompt，先去翻 `logs/prompts/perception_*.jpg`，那张图就是模型当时眼睛看到的世界。很多时候你会发现，问题是目标根本就没进视野，或者被挡住了。

## 4. 动作层：Agent 是怎么真正做事的

模型不会直接控仿真。它只会吐一段 JSON，剩下的事是代码把 JSON 翻译成 TongSim 调用。

动作分发在 `VLMAgent._do_action()`。模型给的 `action` 名字会被路由到对应的 handler，常用的有：

- `move_to_object`：走到某个物体旁边。
- `move_to_location`：走到某个坐标。
- `move_forward` / `move_backward`：前进、后退。
- `turn_in_degree`：转一定角度。
- `move_and_take_object`：走过去并拿起来。
- `put_down_to_location`：把手里的东西放到指定位置。
- `move_and_put_down`：边走边放。
- `look_at_object` / `look_at_location`：把视线移到物体或位置上。
- `speak_to_npc`：跟 NPC 说话。
- `submit_answer`：交问答题答案。
- `solve_raven`：瑞文测试专用解题接口，会自动读取当前题目里的瑞文图片并返回候选答案。
- `finish_task`：告诉任务系统这道行为题做完了。

模型应当输出一个长度为 1 的 JSON 数组，例如：

```json
[
  {
    "think": "我需要先移动到杯子旁边，然后拿起杯子。",
    "action": "move_and_take_object",
    "parameters": {
      "object_id": "3",
      "which_hand": 0
    },
    "output": 0
  }
]
```

问答题大概长这样：

```json
[
  {
    "think": "我数到桌上有 4 个红色物体。",
    "action": "submit_answer",
    "output": "4"
  }
]
```

整理、搬运、开关门这类行为题做完了就是这样：

```json
[
  {
    "think": "目标物体已经放到指定位置，任务完成。",
    "action": "finish_task",
    "output": 0
  }
]
```

动作失败时，代码会把失败信息记下来，下一轮 prompt 会带上 `action_res` 和 `action_histories`。所以 prompt 里最好明明白白告诉模型：上一步失败了别原样再来一次，看着错误信息改策略。否则你会看到它在那里反复撞同一面墙。

## 5. Prompt 才是 Agent 的说明书

这个项目里所谓的"智能"，相当一部分是写在 prompt 里的。换句话说，prompt 写得糙，再强的模型也救不回来。

Prompt 生成代码在：

- `arenaagent/vlm_agent/prompt.py`

常用 prompt 文件分散在三个目录：

- `arenaagent/vlm_agent/prompts/`
- `arenaagent/preliminary_baseline_agent/prompts/`
- `arenaagent/final_baseline_agent/prompts/`

文件大致分工：

- `react.txt`：告诉模型你是谁、能干什么、要按什么格式输出。
- `interaction_info.txt`：把当前任务、能看到的物体、上一步结果摆给模型。
- `instructions*.txt`：策略补充，比如跟 NPC 对话时怎么问、做任务阶段怎么干。
- `task_spec_prompt.json` / `stage_spec_prompts.json`：针对某一类任务的专门提示。
- `api.json` / `api_info.json`：动作接口的说明。

真正发到模型那边的内容，每一轮大概是这么拼起来的：

```text
系统提示：角色设定 + 总规则 + 输出格式
历史消息：最近几轮模型做过什么
当前输入：当前图片 + 当前任务 + 可见物体 + 上一步动作结果
```

模板里会用到一些变量：

- `{api_info}`：可用动作。
- `{task_text}` / `{task_goal}` / `{task_prompt}`：任务描述。
- `{visiable_objects_info}`：可见物体信息。**注意代码里拼的就是 `visiable`，不是 `visible`**，别擅自改名，会导致模板渲染不上。
- `{object_in_hand}`：手里拿的东西。
- `{npc_reply}`：NPC 回话。
- `{action_res}`：上一步动作结果。
- `{action_histories}`：最近几步的动作历史。

模型做出奇怪决策时，不要凭感觉猜它在想什么。打开 `logs/prompts/prompt_*.txt`，看一眼它真正收到的输入，答案通常一眼就出来了。

## 6. 怎么把成绩提上去（重点）

新手最容易提分的地方不是重写代码，而是改 prompt。

### 6.1 先把失败原因定位清楚

每跑挂一次，先按这几个方向自问一下：

- 模型看清楚物体了吗？
- 用对物体 ID 了吗？
- 动作名是不是写错了？
- 参数名是不是写错了？
- 它是不是太早 `finish_task` 了？
- 同一个失败动作是不是被它反复执行？
- 这是问答题还是行为题？它有没有混用 `submit_answer` 和 `finish_task`？

对应的证据在哪：

- `logs/prompts/perception_*.jpg`
- `logs/prompts/prompt_*.txt`
- 日志里搜 `parsed json message`
- 日志里搜 `action run result`

这些证据比"我感觉模型不太聪明"要值钱得多。

### 6.2 优先改 task prompt

某一类任务老是错，别先动通用 prompt，先去改它的专属提示。

初赛任务现在按任务拆分在：

```text
arenaagent/preliminary_baseline_agent/tasks/<任务名>/prompt.txt
```

入口根据 `subject["task_type"]` 自动选择对应策略。旧的 `task_spec_prompt.json` 仅保留用于兼容，新的开发不要再集中修改它。当前支持的任务目录：

| key | 任务类型 |
| --- | --- |
| `jigsaw` | 拼图任务 |
| `tidyroom` | 整理房间任务 |
| `npc` | NPC 问答任务 |
| `raven` | 瑞文测试任务 |
| `counting` | 计数任务 |

决赛任务在：

```text
arenaagent/final_baseline_agent/prompts/stage_spec_prompts.json
```

`stage_spec_prompts.json` 用在决赛 baseline，匹配字段是 `subject["stage"]`。当前支持的 key：

| key | 阶段类型 |
| --- | --- |
| `jigsaw` | 拼图房间 |
| `raven_room` | 瑞文测试房间 |
| `npc_room` | NPC 房间 |
| `tidy_room` | 整理房间 |
| `counting_room` | 计数房间 |

好用的 task prompt 不需要写得多花哨。把它当成一个靠谱助教坐在模型旁边提醒就行：

- 这道题最终要拿到什么。
- 什么时候才算可以提交。
- 看不到目标怎么办，是转身还是走两步。
- 动作失败了下一步该怎么改。
- 输出用哪个动作、要带哪些参数。

举两个例子。计数题可以这样补：

```text
计数时不要重复计算同一个 object_id。
如果题目问颜色或形状，要同时检查 color 和 shape 字段。
如果视野明显不完整，先转身或移动，不要马上提交答案。
确认后使用 submit_answer，output 只写最终数字。
```

整理房间题可以这样补：

```text
进入房间后先用四次本地转向收集目标、家具、地板和障碍物，不调用 VLM。
本地调度器选择 pending 目标，策略代码同时锁定它的目的地。
拿起后用 object_in_hand 确认；确认成功后直接执行本地规划的 move_and_put_down。
移动点位于家具 AABB 外侧，放置点位于桌面/沙发面或垃圾桶内，X/Y/Z 始终按字段名读取。
放置后刷新目标物和目的地的全局 AABB，几何校验通过才标记 done。
几何校验失败时，自动重拾并轮换接近方向和放置点；连续三次本地失败后才调用 VLM 做异常诊断。
```

### 6.3 通用 prompt 别越写越乱

`react.txt` 是所有任务都会读到的总章程。它适合放真正通用的规则：

- 一次只输出一个动作。
- 只能输出 JSON，别带任何额外解释。
- 动作名必须来自 `api_info`。
- 物体 ID 必须来自右侧语义分割图。
- 上一步失败时不要无脑重复。

千万别把某一个具体任务的细节往 `react.txt` 里塞。通用 prompt 越长，模型注意力越分散，最后所有任务一起变笨。

### 6.4 正确的调试节奏

照这个步骤来，效率会高很多：

1. 锁定一个任务，别频繁切换。
2. 跑一遍，留下失败日志。
3. 只改一个 prompt 文件。
4. 同一个任务再跑一遍。
5. 看失败点变没变。

一口气改十处的话，最后你自己也分不清是哪一处起了作用、哪一处反倒搞砸了。

## 7. 跟任务系统相关的接口说明

**不要修改任何任务接口，否则会导致系统启动失败。**

任务系统的接口定义在：

```text
protocol/arena/agent/arena_agent_service.proto
```

Python 这边主要被 `AgentBase` 封装了。

### 7.1 连接和状态

- `_connect()`：连任务系统。
- `_disconnect()`：断开。
- `_task_ready()`：任务有没有准备好。
- `_get_task_status()`：当前 session 是什么状态。
- `_current_subject_finished()`：当前题做完没。

可能出现的状态：

- `WAITING_FOR_AGENT`：任务系统在等 Agent 接入。
- `PENDING`：任务已准备，等开始。
- `RUNNING`：任务进行中。
- `FINISHED`：正常结束。
- `TERMINATED`：被终止。
- `ERROR`：出错了。

### 7.2 拿题目和反馈

- `_get_agent_spawn_info()`：拿出生点、相机参数等初始化信息。
- `_get_subject_from_task()`：拿当前题。
- `_get_response_from_task()`：拿任务系统的反馈。
- `_get_num_subjects_from_task()`：拿题目总数。

不同任务的 `subject` 字段不完全一样。有的带 `task_type`，有的带 `stage`，有的还塞了 NPC 信息。所以代码里到处都是 `subject.get(...)`，就是为了兼容这些差异。

### 7.3 提交动作和答案

- `_get_action_space()`：拿任务系统当前期望的动作格式。
- `_apply_action()`：把这一轮的动作交上去。
- `speak_to`：和 NPC 说话。
- `_evaluate_subject()`：题做完后让任务系统打分。
- `_get_subject_score()`：拿当前题分数。

有一件事很容易被忽略：`VLMAgent` 是先让 TongSim 执行动作，再把结果上交给任务系统的。整个链路是：

```text
模型输出 JSON
    |
VLMAgent 解析动作
    |
TongSim 执行动作
    |
AgentBase 把执行结果交给任务系统
    |
任务系统更新状态和评分
```

## 8. 看代码的建议顺序

第一次摸这个项目，按这个顺序读会舒服很多：

1. `README.md` + `usage_guid.md`：怎么装、怎么跑。
2. `arenaagent/builder.py`：命令行怎么把 Agent 拉起来。
3. `arenaagent/agent_base.py`：跟任务系统的生命周期。
4. `arenaagent/vlm_agent/vlm_agent.py`：每一步怎么看、怎么问、怎么做。
5. `arenaagent/semantic_mapper.py`：物体 ID 和视觉信息怎么整理出来。
6. `arenaagent/vlm_agent/prompt.py`：prompt 是怎么拼起来的。
7. `arenaagent/*_baseline_agent/prompts/`：开始动手改 prompt 提分。

只想先跑起来，读到第 3 步就够了。想认真提分，重点在 4、6、7。

写在最后：这个项目调试的真正难点，不在"模型怎么这么蠢"，而在"我给模型的信息够不够清楚、动作接口好不好用、失败反馈下一轮有没有用上"。沿这条线去查，分数基本能一点点提高上去，加油吧骚年们~。
