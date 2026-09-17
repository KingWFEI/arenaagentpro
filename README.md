# 高校组：通用智能体任务挑战赛

通用智能体任务挑战赛面向通用智能体在具身仿真环境中的感知、决策与执行能力评测，参赛者需要构建能够完成指定任务的Agent。
本仓库提供比赛 baseline、运行脚本、模型配置和上手文档，帮助参赛者快速搭建环境、调试方案并提交结果。

## 比赛系统下载
- 完整比赛系统下载链接
  - 夸克网盘：https://pan.quark.cn/s/b81737fe757a?pwd=8bcQ
  - 百度网盘：https://pan.baidu.com/s/1uvbGpc3UQIibR_RvJ9grBQ?pwd=bj6f

根据你的系统类型选择对应的比赛系统进行下载，不支持MacOS

## 比赛系统说明
1. 系统支持Windows和Linux平台，位于Linux目录和Windows目录，如图所示：
![](docs/screenshot-20260601-115527.png)
请根据你的电脑类型下载对应比赛系统，不支持MacOS。

1. 比赛系统包含基准Agent代码、仿真客户端、赛题系统、文档，如图所示
![](docs/screenshot-20260601-120436.png)
- 基准Agent：就是本仓库代码，提供比赛baseline，帮助选手快速上手比赛，获得比赛结果。
- 客户端：比赛需要的具身仿真环境
- 赛题系统：负责向智能体出题，并评判最终结果
- 文档：比赛系统说明文档

## 如何开始？
开始前你需要将比赛系统下载到本地，并仔细阅读《【挑战赛】初赛系统使用指南.pdf》，根据指南运行比赛系统和baseline agent。

## 运行方法

### 首次准备

首次克隆后，在 PowerShell 中执行：

```powershell
Copy-Item .env.example .env
Copy-Item config.toml.example config.toml
uv sync
```

随后填写 `.env` 中的模型地址和密钥。`.env` 与 `config.toml` 已加入忽略规则，不会被提交到 GitHub。

### 启动顺序

1. 启动比赛 UE 客户端。
2. 启动赛题系统。`start_test.bat` 会自动拉起 `tongsim_server.exe`，无需手动启动。
3. 在本仓库目录另开一个终端运行智能体。

### 五个任务的运行指令

先切到赛题系统解压目录（路径按实际位置修改）：

```powershell
Set-Location -LiteralPath "D:\Contest\Windows\赛题系统\release\release"
```

`train` 为本地调试，`test` 为正式测试（结果写入 `result_{task_id}.bin`）。五个任务逐一执行：

```powershell
# 整理房间
.\start_test.bat train --task competition-preliminary-tidy-room-task
.\start_test.bat test  --task competition-preliminary-tidy-room-task

# 拼图
.\start_test.bat train --task competition-preliminary-jigsaw-task
.\start_test.bat test  --task competition-preliminary-jigsaw-task

# 计数
.\start_test.bat train --task competition-preliminary-counting-task
.\start_test.bat test  --task competition-preliminary-counting-task

# NPC 对话
.\start_test.bat train --task competition-preliminary-npc-task
.\start_test.bat test  --task competition-preliminary-npc-task

# 瑞文测试
.\start_test.bat train --task competition-preliminary-raven-task
.\start_test.bat test  --task competition-preliminary-raven-task
```

`start_test.bat` 内部已封装 `--resource-pack` 与 `--resource-key`，无需额外传入。五个任务全部跑完后执行 `.\start_test.bat package-results`，生成 `result_package.bin` 用于上传。

### 运行智能体

五个任务共用同一个入口，Agent 按题目中的 `task_type` 自动路由到对应策略。

**方式一：封装脚本**（推荐，脚本会自行切到项目根目录）

在本仓库目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_jigsaw.ps1
```

脚本名中的 `jigsaw` 为历史遗留，实际对五个任务通用。若要检查当前配置可用的视觉模型：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_jigsaw.ps1 -ListModels
```

**方式二：直接调用**（参数更直观）

```powershell
Set-Location -LiteralPath "D:\Contest\Windows\基准Agent\baseline_agent\baseline_agent\arenaagentpro"   # 改成你的仓库路径
uv run arenaagent --agent_name preliminary_baseline_agent --config config.toml --vlm_model VLMGPT5Config --run_times 10
```

`--run_times` 为重复次数，一次运行为一个完整 session（跑完所有题目，直到 session 结束）。直接调用时**必须先切到本仓库目录**，因为 `.env` 是按当前目录查找的，找不到会静默跳过，导致模型配置不生效。

两种方式等价。需要固定依赖环境时给 `uv run` 加 `--no-sync`（封装脚本已默认加上）。

### 拼图专用求解器

当前版本包含本地验证通过的拼图专用求解器，支持图像匹配、固定标准朝向、欧拉角异常纠正，以及在新版 TongSim 接口上的安全双手预抓取。旧版比赛接口无法指定放下哪只手，因此会自动切换到准确率更稳定的单手流程。

## 进一步阅读

[docs/usage_guide.md](docs/usage_guide.md)包含完整上手指南，包含 Agent 工作流程、感知层、动作层、Prompt 调试和提分思路。

[docs/preliminary_task_architecture.md](docs/preliminary_task_architecture.md)说明初赛五个任务的独立开发目录、策略接口、三人分工与测试方法。
