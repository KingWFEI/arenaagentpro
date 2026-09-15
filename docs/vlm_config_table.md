# VLM 配置表

来源：[arenaagent/vlm_agent/vlm_config.py](../arenaagent/vlm_agent/vlm_config.py)

## 环境变量优先级

`VLMConfig.apply_env_overrides()` 会先读取 `VLM_CLIENT_TYPE` 覆盖模式，然后由 `VLMClientCfg.apply_env_overrides()` 覆盖 `client_cfg`。

优先级如下：

| 优先级 | 作用范围 | 环境变量 |
|---|---|---|
| 1 | 覆盖模式 | `VLM_CLIENT_TYPE` |
| 2 | 通用覆盖，所有模式优先读取 | `VLM_CLIENT_CFG_NAME`, `VLM_CLIENT_CFG_REGION`, `VLM_CLIENT_CFG_API_BASE`, `VLM_CLIENT_CFG_API_KEY`, `VLM_CLIENT_CFG_API_VERSION`, `VLM_CLIENT_CFG_MESSAGE_ROLE` |
| 3 | 按模式兜底读取 | 见下方“模式环境变量表” |

## 模式环境变量表

如果没有设置通用 `VLM_CLIENT_CFG_*`，会按当前 `client_type` 读取以下变量。

| 模式 `client_type` | 模型名 | API 地址 | API Key | API 版本 |
|---|---|---|---|---|
| `azure` | `AZURE_OPENAI_MODEL` | `AZURE_OPENAI_ENDPOINT` | `AZURE_OPENAI_API_KEY` | `AZURE_OPENAI_API_VERSION`, `OPENAI_API_VERSION` |
| `openai` | `OPENAI_MODEL` | `OPENAI_BASE_URL`, `OPENAI_API_BASE` | `OPENAI_API_KEY` | - |
| `ark` | `ARK_MODEL` | `ARK_BASE_URL`, `ARK_API_BASE` | `ARK_API_KEY` | - |
| `dashscope` | `DASHSCOPE_MODEL` | `DASHSCOPE_BASE_URL`, `DASHSCOPE_API_BASE` | `DASHSCOPE_API_KEY` | - |
| `zhipu` | `ZHIPU_MODEL` | `ZHIPU_BASE_URL`, `ZHIPU_API_BASE` | `ZHIPU_API_KEY` | - |
| `stepai` | `STEPAI_MODEL` | `STEPAI_BASE_URL`, `STEPAI_API_BASE` | `STEPAI_API_KEY` | - |
| `lingyi` | `LINGYI_MODEL` | `LINGYI_BASE_URL`, `LINGYI_API_BASE` | `LINGYI_API_KEY` | - |
| `gpt-agent` | `GPT_AGENT_MODEL`, `AZURE_OPENAI_MODEL`, `OPENAI_MODEL` | `GPT_AGENT_API_BASE`, `AZURE_OPENAI_ENDPOINT`, `OPENAI_BASE_URL`, `OPENAI_API_BASE` | `AZURE_OPENAI_API_KEY`, `OPENAI_API_KEY` | `AZURE_OPENAI_API_VERSION`, `OPENAI_API_VERSION` |

## 配置选项表

| 配置选项 | 模式 | 默认模型 | 默认 API 地址 |
|---|---|---|---|
| `VLMGPTConfig` | `azure` | `gpt-4o-2024-11-20` | `https://api.openai.com/v1` |
| `VLMGPT52Config` | `azure` | `gpt-5.2-2025-12-11` | `https://api.openai.com/v1` |
| `VLMGPT5Config` | `azure` | `gpt-5.1` | `https://api.openai.com/v1` |
| `VLMGPTAgentConfig` | `azure` | `gpt-agent` | `https://api.openai.com/v1` |
| `VLMGPT4o0806Config` | `azure` | `gpt-4o-2024-08-06` | `https://api.openai.com/v1` |
| `VLMGPT4o0513Config` | `azure` | `gpt-4o-2024-05-13` | `https://api.openai.com/v1` |
| `VLMGPT35TruboConfig` | `azure` | `gpt-35-turbo-0125` | `https://api.openai.com/v1` |
| `VLMGPT4TruboConfig` | `azure` | `gpt-4-turbo-2024-04-09` | `https://api.openai.com/v1` |
| `VLMGPT40125PreviewConfig` | `azure` | `gpt-4-0125-preview` | `https://api.openai.com/v1` |
| `VLMGPT41106PreviewConfig` | `azure` | `gpt-4-1106-preview` | `https://api.openai.com/v1` |
| `VLMGPT4Turbo20240409Config` | `azure` | `gpt-4-turbo-2024-04-09` | `https://api.openai.com/v1` |
| `VLMGPT4VisionPreviewConfig` | `azure` | `gpt-4-vision-preview` | `https://api.openai.com/v1` |
| `VLMGPT40613Config` | `azure` | `gpt-4-0613` | `https://api.openai.com/v1` |
| `VLMGPT410414Config` | `openai` | `gpt-4.1-2025-04-14` | `https://api.openai.com/v1` |
| `VLMGPT41mini0414Config` | `openai` | `gpt-4.1-mini-2025-04-14` | `https://api.openai.com/v1` |
| `VLMGPT41nano0414Config` | `openai` | `gpt-4.1-nano-2025-04-14` | `https://api.openai.com/v1` |
| `VLMGPTo4mini0416Config` | `openai` | `o4-mini-2025-04-16` | `https://api.openai.com/v1` |
| `VLMGPTo3mini0131Config` | `openai` | `o3-mini-2025-01-31` | `https://api.openai.com/v1` |
| `VLMGPTo30416Config` | `openai` | `o3-2025-04-16` | `https://api.openai.com/v1` |
| `VLMGPTo1mini0912Config` | `openai` | `o1-mini-2024-09-12` | `https://api.openai.com/v1` |
| `VLMO30416Config` | `openai` | `o3-2025-04-16` | `https://api.openai.com/v1` |
| `VLMO11217Config` | `openai` | `o1-2024-12-17` | `https://api.openai.com/v1` |
| `VLMGPT45PreviewConfig` | `openai` | `gpt-4.5-preview-2025-02-27` | `https://api.openai.com/v1` |
| `VLMGPT4oMiniConfig` | `openai` | `gpt-4o-mini-2024-07-18` | `https://api.openai.com/v1` |
| `VLMGPT4o1120Config` | `openai` | `gpt-4o-2024-11-20` | `https://litellm.mybigai.ac.cn/` |
| `VLMClaude35AllConfig` | `openai` | `claude-3-5-sonnet-all` | `https://api2.aigcbest.top/v1` |
| `VLMClaude35LatestConfig` | `openai` | `claude-3-5-sonnet-latest` | `https://api2.aigcbest.top/v1` |
| `VLMClaude350620Config` | `openai` | `claude-3-5-sonnet-20240620` | `https://api2.aigcbest.top/v1` |
| `VLMClaude370219Config` | `openai` | `claude-3-7-sonnet-20250219` | `https://api2.aigcbest.top/v1` |
| `VLMClaude37LatestConfig` | `openai` | `claude-3-7-sonnet-latest` | `https://api2.aigcbest.top/v1` |
| `VLMClaude4LatestConfig` | `openai` | `claude-4-sonnet-latest` | `https://api2.aigcbest.top/v1` |
| `VLMClaude40514Config` | `openai` | `claude-sonnet-4-20250514` | `https://api2.aigcbest.top/v1` |
| `VLMClaude45Config` | `openai` | `claude-sonnet-4-5-20250929` | `https://api2.aigcbest.top/v1` |
| `VLMLlamaQddConfig` | `openai` | `llama-3.2-90b-vision-instruct` | `https://api2.aigcbest.top/v1` |
| `VLMGLMConfig` | `openai` | `glm-4v-plus` | `https://api2.aigcbest.top/v1` |
| `VLMQwen2Config` | `openai` | `Qwen/Qwen2-VL-72B-Instruct` | `https://api2.aigcbest.top/v1` |
| `VLMDoubaoSeed18Config` | `openai` | `doubao-seed-1-8-251215` | `https://api2.aigcbest.top/v1` |
| `VLMDoubaoVisionConfig` | `openai` | `Doubao-1.5-vision-pro-32k` | `https://api2.aigcbest.top/v1` |
| `VLMLlama4Maverick17bConfig` | `openai` | `meta/llama-4-maverick-17b-128e-instruct` | `https://api2.aigcbest.top/v1` |
| `VLMLlama4Scout17bConfig` | `openai` | `llama-4-scout-17b-16e-instruct` | `https://api2.aigcbest.top/v1` |
| `Gemini25ProPreview20250605` | `openai` | `gemini-2.5-pro-preview-06-05` | `https://api2.aigcbest.top/v1` |
| `Gemini25ProPreview20250325` | `openai` | `gemini-2.5-pro-preview-03-25` | `https://api2.aigcbest.top/v1` |
| `Gemini20ProExp20250205` | `openai` | `gemini-2.0-pro-exp-20250205` | `https://api2.aigcbest.top/v1` |
| `Gemini20Flash` | `openai` | `gemini-2.0-flash-exp` | `https://api2.aigcbest.top/v1` |
| `Gemini25FlashPreview20250417` | `openai` | `gemini-2.5-flash-preview-04-17` | `https://api2.aigcbest.top/v1` |
| `Gemini3ProPreview` | `openai` | `gemini-3-pro-preview` | `https://api2.aigcbest.top/v1` |
| `VLMGrok3Config` | `openai` | `grok-3` | `https://api2.aigcbest.top/v1` |
| `VLMGrok4Config` | `openai` | `grok-4` | `https://api2.aigcbest.top/v1` |
| `VLMQwenVLMaxConfig` | `dashscope` | `qwen-vl-max-latest` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `VLMQwenVLMax0125Config` | `dashscope` | `qwen-vl-max-2025-01-25` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `VLMQvqMaxConfig` | `dashscope` | `qvq-max-latest` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `VLMQvqMaxLatestConfig` | `dashscope` | `qvq-max-latest` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `VLMQvqMax0325Config` | `dashscope` | `qvq-max-2025-03-25` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `VLMQvq72bPreviewConfig` | `dashscope` | `qvq-72b-preview` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `VLMQwenVLPlusConfig` | `dashscope` | `qwen-vl-plus` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `VLMQwenVLPlusLatestConfig` | `dashscope` | `qwen-vl-plus-latest` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `VLMQwenVLPlus0125Config` | `dashscope` | `qwen-vl-plus-2025-01-25` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `VLMQwen25Vl32bInstruct` | `dashscope` | `qwen2.5-vl-32b-instruct` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `VLMQwen25Vl72bInstruct` | `dashscope` | `qwen2.5-vl-72b-instruct` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `VLMQwen25Vl7bInstruct` | `dashscope` | `qwen2.5-vl-7b-instruct` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `VLMQwen25Vl3bInstruct` | `dashscope` | `qwen2.5-vl-3b-instruct` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `VLMQwen3VLPlus` | `dashscope` | `qwen3-vl-plus` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `VLMLlamaConfig` | `dashscope` | `llama-3.2-90b-vision-instruct` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `VLMQwen25Config` | `dashscope` | `qwen2.5-vl-72b-instruct` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `VLMGLM4VConfig` | `zhipu` | `glm-4v-plus-0111` | `https://open.bigmodel.cn/api/paas/v4/` |
| `VLMStep1oVisionConfig` | `stepai` | `step-1o-vision-32k` | `https://api.stepfun.com/v1` |
| `VLMYiVisionV2Config` | `lingyi` | `yi-vision-v2` | `https://api.lingyiwanwu.com/v1` |
| `VLMHumanConfig` | `human` | `human` | - |

## 使用建议

如果只想切换模型服务，优先使用通用变量：

```bash
VLM_CLIENT_TYPE=openai
VLM_CLIENT_CFG_NAME=gpt-4o-2024-11-20
VLM_CLIENT_CFG_API_BASE=https://api.openai.com/v1
VLM_CLIENT_CFG_API_KEY=...
```

如果想按厂商保留默认映射，就设置对应模式的变量，例如 `openai` 模式设置 `OPENAI_API_KEY`，`dashscope` 模式设置 `DASHSCOPE_API_KEY`。
