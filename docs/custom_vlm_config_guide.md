# 添加自定义 VLM 配置指南

本指南说明如何在 [arenaagent/vlm_agent/vlm_config.py](../arenaagent/vlm_agent/vlm_config.py) 中添加自定义模型配置，并通过环境变量管理 API Key、API Base 和模型名。

相关速查表见 [docs/vlm_config_table.md](./vlm_config_table.md)。

## 先选添加方式

| 场景 | 推荐方式 | 是否需要改代码 |
|---|---|---|
| 只是临时切换模型名、API 地址或 Key | 使用环境变量覆盖 | 否 |
| 新增一个固定可复用的模型配置 | 新增一个 `VLMConfig` 子类 | 是 |
| 新接入一个已有客户端协议兼容的供应商 | 新增配置类，`client_type` 复用 `openai`、`azure`、`dashscope` 等现有模式 | 是 |
| 新供应商协议不兼容现有客户端 | 新增配置类，并在 `client.py` 注册新的 Client | 是 |

优先建议复用现有 `client_type`。例如很多中转服务、Claude/Gemini/Llama 的 OpenAI 兼容接口，都可以使用 `client_type = "openai"`。

## 方式一：不改代码，直接用环境变量覆盖

如果只是临时换模型，先用通用环境变量：

```powershell
$env:VLM_CLIENT_TYPE="openai"
$env:VLM_CLIENT_CFG_NAME="your-model-name"
$env:VLM_CLIENT_CFG_API_BASE="https://your-api-base/v1"
$env:VLM_CLIENT_CFG_API_KEY="your-api-key"
```

然后运行时任选一个已有配置类作为入口：

```powershell
uv run arenaagent --agent_name final_baseline_agent --config config.toml --vlm_model VLMGPT4o1120Config
```

`VLM_CLIENT_TYPE` 会覆盖配置类里的 `client_type`；`VLM_CLIENT_CFG_*` 会覆盖配置类里的 `client_cfg` 字段。

## 方式二：新增一个 OpenAI 兼容配置类

适用于你的服务兼容 OpenAI Chat Completions 协议，或者已有中转服务暴露的是 OpenAI 风格接口。

### 1. 添加到 `__ALL__`

在 `vlm_config.py` 顶部的 `__ALL__` 列表里加入新类名：

```python
__ALL__ = [
    # ...
    "VLMAcmeVisionConfig",
]
```

`--get_vlm_model` 读取的是这个列表，所以加进去之后命令行才能列出来。

### 2. 新增配置类

在同一个文件中新增配置类：

```python
@configclass
class VLMAcmeVisionConfig(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="acme-vision-model",
        api_base="https://api.acme.example/v1",
        api_key="",
        api_version="2024-10-21",
        message_role="user",
    )
```

字段含义：

| 字段 | 说明 |
|---|---|
| `client_type` | 决定使用哪个客户端实现。OpenAI 兼容接口通常填 `openai` |
| `name` | 默认模型名 |
| `api_base` | 默认 API 地址 |
| `api_key` | 保持为空，通过环境变量传入 |
| `api_version` | Azure 常用；OpenAI 兼容服务通常可以保留或省略 |
| `message_role` | 通常保持 `user` |

### 3. 配置环境变量

OpenAI 兼容模式使用这些变量：

```powershell
$env:OPENAI_API_KEY="your-api-key"
$env:OPENAI_BASE_URL="https://api.acme.example/v1"
```

如果你希望只影响这次配置，不和其他 OpenAI 配置混用，也可以使用优先级更高的通用变量：

```powershell
$env:VLM_CLIENT_CFG_API_KEY="your-api-key"
$env:VLM_CLIENT_CFG_API_BASE="https://api.acme.example/v1"
```

### 4. 运行验证

先确认新配置能被列出来：

```powershell
uv run arenaagent --agent_name vlm_agent --get_vlm_model
```

再指定新配置运行：

```powershell
uv run arenaagent --agent_name final_baseline_agent --config config.toml --vlm_model VLMAcmeVisionConfig
```

## 方式三：新增已有模式的厂商环境变量

如果新供应商仍然走现有客户端，只是想有独立的环境变量名，可以在 `_CLIENT_TYPE_VLM_CLIENT_CFG_ENV_NAMES` 中加一个模式。

示例：

```python
_CLIENT_TYPE_VLM_CLIENT_CFG_ENV_NAMES: dict[str, dict[str, tuple[str, ...]]] = {
    # ...
    "acme": {
        "name": ("ACME_MODEL",),
        "api_base": ("ACME_BASE_URL", "ACME_API_BASE"),
        "api_key": ("ACME_API_KEY",),
        "api_version": ("ACME_API_VERSION",),
    },
}
```

但要注意：只加这里还不够。`client_type` 最终会传给 `ClientFactory().build()`，它必须能在 [arenaagent/vlm_agent/client.py](../arenaagent/vlm_agent/client.py) 中找到同名注册客户端。

如果没有 `@RegisterClient("acme")`，运行时会报：

```text
KeyError: "Client type 'acme' is not registered"
```

所以，除非你也新增了对应 Client，否则配置类里仍然应该复用现有模式，例如：

```python
client_type: str = "openai"
```

## 方式四：新增一种真正的 Client

只有当供应商协议不兼容现有客户端时，才需要新增 Client。

需要做三件事：

1. 在 `client.py` 中新增客户端类，并使用 `@RegisterClient("your_type")` 注册。
2. 在 `vlm_config.py` 的 `_CLIENT_TYPE_VLM_CLIENT_CFG_ENV_NAMES` 中加入对应环境变量映射。
3. 新增一个 `VLMConfig` 子类，设置 `client_type = "your_type"`。

最小结构示例：

```python
@RegisterClient("acme")
class AcmeClient(Client):
    def __init__(self, cfg: Any) -> None:
        self._cfg = cfg
        super().__init__(native_client=None)

    def _invoke(self, message: Any) -> ClientResponse:
        # 在这里调用供应商 SDK 或 HTTP API
        return ClientResponse(text=response_text, raw=raw_response)
```

除非确实需要新协议，不建议走这条路。复用 `openai` 或 `azure` 的维护成本最低。

## 自定义 Prompt 配置

如果这个模型需要不同的 prompt 文件，可以在配置类里覆盖 `prompt_config`：

```python
@configclass
class VLMAcmeVisionConfig(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="acme-vision-model",
        api_base="https://api.acme.example/v1",
        api_key="",
        message_role="user",
    )
    prompt_config: VLMPromptConfig = VLMPromptConfig(
        type="file",
        characteristics="prompts/simple/characteristics.txt",
        interactions="prompts/interaction_info.txt",
        histories="prompts/history.txt",
        instructions="prompts/simple/instructions.txt",
    )
```

Prompt 路径通常相对于 `arenaagent/vlm_agent/` 下的 prompt 搜索目录使用。新增 prompt 文件后，建议先跑一小轮并检查 `logs/prompts/`，确认实际发给模型的内容符合预期。

## 检查清单

提交前快速检查这些点：

| 检查项 | 要求 |
|---|---|
| 类名 | 使用清晰名字，例如 `VLMProviderModelConfig` |
| `__ALL__` | 新类名已加入，方便 `--get_vlm_model` 展示 |
| `client_type` | 优先复用已有模式；新增模式必须有对应 `@RegisterClient` |
| `api_key` | 不写死在代码里，保持空字符串 |
| 环境变量 | 至少配置对应模式的 API Key；必要时配置 API Base 和模型名 |
| 验证命令 | `--get_vlm_model` 能看到配置，实际运行时能初始化 Client |

## 常见问题

### 配置类加了，但 `--get_vlm_model` 看不到

检查是否把类名加入了 `__ALL__`。

### 报 `Client type ... is not registered`

说明 `client_type` 没有对应的 Client。优先把 `client_type` 改成已有模式，例如 `openai`、`azure`、`dashscope`、`zhipu`、`stepai` 或 `lingyi`。

### 报 API Key 未设置

确认使用的 `client_type`，然后设置对应环境变量。例如：

| 模式 | Key 变量 |
|---|---|
| `openai` | `OPENAI_API_KEY` |
| `azure` | `AZURE_OPENAI_API_KEY` |
| `dashscope` | `DASHSCOPE_API_KEY` |
| `zhipu` | `ZHIPU_API_KEY` |
| `stepai` | `STEPAI_API_KEY` |
| `lingyi` | `LINGYI_API_KEY` |

也可以直接使用通用变量：

```powershell
$env:VLM_CLIENT_CFG_API_KEY="your-api-key"
```

### 想临时覆盖某个配置类的默认模型名

使用：

```powershell
$env:VLM_CLIENT_CFG_NAME="another-model-name"
```

或者按模式设置，例如 `openai` 模式：

```powershell
$env:OPENAI_MODEL="another-model-name"
```
