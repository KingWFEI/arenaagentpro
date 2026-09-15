import os

from arenaagent.utils.configclass import configclass

__ALL__ = [
    "VLMGPTConfig",
    "VLMGPT52Config",
    "VLMGPT5Config",
    "VLMGPTAgentConfig",
    # "VLMClaudeConfig",
    "VLMLlamaQddConfig",
    "VLMGLMConfig",
    "VLMDoubaoSeed18Config",
    "VLMDoubaoVisionConfig",
    # "VLMHumanConfig",
    "VLMQwenVLMaxConfig",
    "VLMQwenVLPlusConfig",
    "VLMLlamaConfig",
    "VLMQwen25Config",
    "VLMGLM4VConfig",
    "VLMStep1oVisionConfig",
    "VLMYiVisionV2Config",
    "VLMGPT410414Config",
    "VLMGPT41mini0414Config",
    "VLMGPT41nano0414Config",
    "VLMGPTo4mini0416Config",
    "VLMGPTo3mini0131Config",
    "VLMGPTo1mini0912Config",
    "VLMGPTo30416Config",
    "VLMO30416Config",
    "VLMO11217Config",
    "VLMGPT45PreviewConfig",
    "VLMGPT4oMiniConfig",
    "VLMGPT4o1120Config",
    "VLMGPT4o0806Config",
    "VLMGPT4o0513Config",
    "VLMGPT35TruboConfig",
    "VLMGPT4TruboConfig",
    "VLMGPT40125PreviewConfig",
    "VLMGPT41106PreviewConfig",
    "VLMGPT4Turbo20240409Config",
    "VLMGPT4VisionPreviewConfig",
    "VLMGPT40613Config",
    "VLMClaude35AllConfig",
    "VLMClaude35LatestConfig",
    "VLMClaude350620Config",
    "VLMClaude370219Config",
    "VLMClaude37LatestConfig",
    "VLMClaude4LatestConfig",
    "VLMClaude40514Config",
    "VLMClaude45Config",
    "VLMQwen2Config",
    "VLMQvqMaxConfig",
    "VLMQvqMaxLatestConfig",
    "VLMQvqMax0325Config",
    "VLMQvq72bPreviewConfig",
    "VLMQwenVLPlusLatestConfig",
    "VLMQwenVLPlus0125Config",
    "VLMQwen25Vl32bInstruct",
    "VLMQwen25Vl72bInstruct",
    "VLMQwen25Vl7bInstruct",
    "VLMQwen25Vl3bInstruct",
    "VLMQwen3VLPlus",
    "VLMLlama4Maverick17bConfig",
    "VLMLlama4Scout17bConfig",
    "VLMGrok3Config",
    "VLMGrok4Config",
    "Gemini25FlashPreview20250417",
    "Gemini25ProPreview20250605",
    "Gemini3ProPreview"
]


def _first_env(*env_names: str) -> str:
    for env_name in env_names:
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return ""


_GENERIC_VLM_CLIENT_CFG_ENV_NAMES: dict[str, tuple[str, ...]] = {
    "name": ("VLM_CLIENT_CFG_NAME",),
    "region": ("VLM_CLIENT_CFG_REGION",),
    "api_base": ("VLM_CLIENT_CFG_API_BASE",),
    "api_key": ("VLM_CLIENT_CFG_API_KEY",),
    "api_version": ("VLM_CLIENT_CFG_API_VERSION",),
    "message_role": ("VLM_CLIENT_CFG_MESSAGE_ROLE",),
}


_CLIENT_TYPE_VLM_CLIENT_CFG_ENV_NAMES: dict[str, dict[str, tuple[str, ...]]] = {
    "azure": {
        "name": ("AZURE_OPENAI_MODEL",),
        "api_base": ("AZURE_OPENAI_ENDPOINT",),
        "api_key": ("AZURE_OPENAI_API_KEY",),
        "api_version": ("AZURE_OPENAI_API_VERSION", "OPENAI_API_VERSION"),
    },
    "openai": {
        "name": ("OPENAI_MODEL",),
        "api_base": ("OPENAI_BASE_URL", "OPENAI_API_BASE"),
        "api_key": ("OPENAI_API_KEY",),
    },
    "ark": {
        "name": ("ARK_MODEL",),
        "api_base": ("ARK_BASE_URL", "ARK_API_BASE"),
        "api_key": ("ARK_API_KEY",),
    },
    "dashscope": {
        "name": ("DASHSCOPE_MODEL",),
        "api_base": ("DASHSCOPE_BASE_URL", "DASHSCOPE_API_BASE"),
        "api_key": ("DASHSCOPE_API_KEY",),
    },
    "zhipu": {
        "name": ("ZHIPU_MODEL",),
        "api_base": ("ZHIPU_BASE_URL", "ZHIPU_API_BASE"),
        "api_key": ("ZHIPU_API_KEY",),
    },
    "stepai": {
        "name": ("STEPAI_MODEL",),
        "api_base": ("STEPAI_BASE_URL", "STEPAI_API_BASE"),
        "api_key": ("STEPAI_API_KEY",),
    },
    "lingyi": {
        "name": ("LINGYI_MODEL",),
        "api_base": ("LINGYI_BASE_URL", "LINGYI_API_BASE"),
        "api_key": ("LINGYI_API_KEY",),
    },
    "gpt-agent": {
        "name": ("GPT_AGENT_MODEL", "AZURE_OPENAI_MODEL", "OPENAI_MODEL"),
        "api_base": ("GPT_AGENT_API_BASE", "AZURE_OPENAI_ENDPOINT", "OPENAI_BASE_URL", "OPENAI_API_BASE"),
        "api_key": ("AZURE_OPENAI_API_KEY", "OPENAI_API_KEY"),
        "api_version": ("AZURE_OPENAI_API_VERSION", "OPENAI_API_VERSION"),
    },
}


@configclass
class MessageTemplate:
    """Configuration class for agent.

    This class provides the configuration for the agent.

    An example of the message template:
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"}},
                {"type": "text", "text": ""},
            ],
        },

    """

    # The name of the agent, e.g. "gpt-3", "random_agent", etc.
    role: str = "user"
    content: list | str = [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"}},
        {"type": "text", "text": ""},
    ]
    # content


@configclass
class VLMPromptConfig:
    type: str = "file"  # file or text

    # If type is file, the path to the file containing the prompts
    # If type is text, the text of the prompts
    characteristics: str = "prompts/react.txt"
    interactions: str = "prompts/interaction_info.txt"
    histories: str = "prompts/history.txt"
    instructions: str = "prompts/instructions.txt"
    user_message_template: MessageTemplate = MessageTemplate()
    user_image_path = "['content'][0]['image_url']['url']"
    user_text_path = "['content'][1]['text']"

    system_message_template: MessageTemplate = MessageTemplate(
        role="system",
        content="",
    )
    system_text_path = "['content']"


@configclass
class VLMClientCfg:
    name: str = ""
    region: str = "eastus"
    api_base: str = "https://api.openai.com/v1"

    api_key: str = ""
    api_version: str = "2024-10-21"

    message_role: str = "user"

    def __post_init__(self) -> None:
        self.apply_env_overrides()

    def apply_env_overrides(self, client_type: str = "") -> "VLMClientCfg":
        client_type_key = (client_type or "").strip().lower()
        client_env_names = _CLIENT_TYPE_VLM_CLIENT_CFG_ENV_NAMES.get(client_type_key, {})
        for field_name, generic_env_names in _GENERIC_VLM_CLIENT_CFG_ENV_NAMES.items():
            env_value = _first_env(*generic_env_names)
            if not env_value:
                env_value = _first_env(*client_env_names.get(field_name, ()))
            if env_value:
                setattr(self, field_name, env_value)
        return self


@configclass
class VLMConfig:
    client_type: str = "azure"  # openai, openai or ark
    client_cfg: VLMClientCfg = VLMClientCfg()
    prompt_config: VLMPromptConfig = VLMPromptConfig()

    def __post_init__(self) -> None:
        self.apply_env_overrides()

    def apply_env_overrides(self) -> "VLMConfig":
        client_type = _first_env("VLM_CLIENT_TYPE")
        if client_type:
            self.client_type = client_type
        self.client_cfg.apply_env_overrides(self.client_type)
        return self

@configclass
class VLMGPTConfig(VLMConfig):
    client_type: str = "azure"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="gpt-4o-2024-11-20",
        region="eastus",
        api_base="https://api.openai.com/v1",
        api_key="",
        api_version="2024-10-21",
        message_role="user",
    )


@configclass
class VLMGPT52Config(VLMConfig):
    client_type: str = "azure"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="gpt-5.2-2025-12-11",
        region="eastus",
        api_base="https://api.openai.com/v1",
        api_key="",
        api_version="2024-10-21",
        message_role="user",
    )

@configclass
class VLMGPT5Config(VLMConfig):
    client_type: str = "azure"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="gpt-5.1",
        region="eastus",
        api_base="https://api.openai.com/v1",
        api_key="",
        api_version="2024-10-21",
        message_role="user",
    )

@configclass
class VLMGPTAgentConfig(VLMConfig):
    client_type: str = "azure"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="gpt-agent",
        region="eastus",
        api_base="https://api.openai.com/v1",
        api_key="",
        api_version="2024-10-21",
        message_role="user",
    )

@configclass
class VLMGPT410414Config(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="gpt-4.1-2025-04-14",
        region="eastus2",
        api_base="https://api.openai.com/v1",
        api_key="",
        api_version="2024-10-21",
        message_role="user",
    )


@configclass
class VLMGPT41mini0414Config(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="gpt-4.1-mini-2025-04-14",
        region="eastus2",
        api_base="https://api.openai.com/v1",
        api_key="",
        api_version="2024-10-21",
        message_role="user",
    )


@configclass
class VLMGPT41nano0414Config(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="gpt-4.1-nano-2025-04-14",
        region="eastus2",
        api_base="https://api.openai.com/v1",
        api_key="",
        api_version="2024-10-21",
        message_role="user",
    )


@configclass
class VLMGPTo4mini0416Config(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="o4-mini-2025-04-16",
        region="eastus2",
        api_base="https://api.openai.com/v1",
        api_key="",
        api_version="2025-03-01-preview",
        message_role="user",
    )


@configclass
class VLMGPTo3mini0131Config(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="o3-mini-2025-01-31",
        region="eastus",
        api_base="https://api.openai.com/v1",
        api_key="",
        api_version="2025-03-01-preview",
        message_role="user",
    )


@configclass
class VLMGPTo30416Config(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="o3-2025-04-16",
        region="eastus2",
        api_base="https://api.openai.com/v1",
        api_key="",
        api_version="2025-03-01-preview",
        message_role="user",
    )


@configclass
class VLMGPTo1mini0912Config(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="o1-mini-2024-09-12",
        region="westus",
        api_base="https://api.openai.com/v1",
        api_key="",
        api_version="2025-03-01-preview",
        message_role="user",
    )


@configclass
class VLMO30416Config(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="o3-2025-04-16",
        region="eastus2",
        api_base="https://api.openai.com/v1",
        api_key="",
        api_version="2025-03-01-preview",
        message_role="user",
    )


@configclass
class VLMO11217Config(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="o1-2024-12-17",
        region="eastus2",
        api_base="https://api.openai.com/v1",
        api_key="",
        api_version="2025-03-01-preview",
        message_role="user",
    )


@configclass
class VLMGPT45PreviewConfig(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="gpt-4.5-preview-2025-02-27",
        region="eastus2",
        api_base="https://api.openai.com/v1",
        api_key="",
        api_version="2024-10-21",
        message_role="user",
    )


@configclass
class VLMGPT4oMiniConfig(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="gpt-4o-mini-2024-07-18",
        region="eastus",
        api_base="https://api.openai.com/v1",
        api_key="",
        api_version="2024-10-21",
        message_role="user",
    )


@configclass
class VLMGPT4o1120Config(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="gpt-4o-2024-11-20",
        region="eastus",
        api_base="https://api.openai.com/v1",
        api_key="",
        api_version="2024-10-21",
        message_role="user",
    )


@configclass
class VLMGPT4o0806Config(VLMConfig):
    client_type: str = "azure"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="gpt-4o-2024-08-06",
        region="eastus",
        api_base="https://api.openai.com/v1",
        api_key="",
        api_version="2024-10-21",
        message_role="user",
    )


@configclass
class VLMGPT4o0513Config(VLMConfig):
    client_type: str = "azure"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="gpt-4o-2024-05-13",
        region="eastus",
        api_base="https://api.openai.com/v1",
        api_key="",
        api_version="2024-10-21",
        message_role="user",
    )


@configclass
class VLMGPT35TruboConfig(VLMConfig):
    client_type: str = "azure"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="gpt-35-turbo-0125",
        region="canadaeast",
        api_base="https://api.openai.com/v1",
        api_key="",
        api_version="2024-10-21",
        message_role="user",
    )


@configclass
class VLMGPT4TruboConfig(VLMConfig):
    client_type: str = "azure"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="gpt-4-turbo-2024-04-09",
        region="eastus2",
        api_base="https://api.openai.com/v1",
        api_key="",
        api_version="2024-10-21",
        message_role="user",
    )


@configclass
class VLMGPT40125PreviewConfig(VLMConfig):
    client_type: str = "azure"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="gpt-4-0125-preview",
        region="eastus",
        api_base="https://api.openai.com/v1",
        api_key="",
        api_version="2024-10-21",
        message_role="user",
    )


@configclass
class VLMGPT41106PreviewConfig(VLMConfig):
    client_type: str = "azure"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="gpt-4-1106-preview",
        region="australiaeast",
        api_base="https://api.openai.com/v1",
        api_key="",
        api_version="2024-10-21",
        message_role="user",
    )


@configclass
class VLMGPT4Turbo20240409Config(VLMConfig):
    client_type: str = "azure"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="gpt-4-turbo-2024-04-09",
        region="eastus2",
        api_base="https://api.openai.com/v1",
        api_key="",
        api_version="2024-10-21",
        message_role="user",
    )


@configclass
class VLMGPT4VisionPreviewConfig(VLMConfig):
    client_type: str = "azure"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="gpt-4-vision-preview",
        region="australiaeast",
        api_base="https://api.openai.com/v1",
        api_key="",
        api_version="2024-10-21",
        message_role="user",
    )


@configclass
class VLMGPT40613Config(VLMConfig):
    client_type: str = "azure"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="gpt-4-0613",
        region="australiaeast",
        api_base="https://api.openai.com/v1",
        api_key="",
        api_version="2024-10-21",
        message_role="user",
    )


@configclass
class VLMClaude35AllConfig(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="claude-3-5-sonnet-all",
        api_base="https://api2.aigcbest.top/v1",
        api_key="",
        message_role="user",
    )


@configclass
class VLMClaude35LatestConfig(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="claude-3-5-sonnet-latest",
        api_base="https://api2.aigcbest.top/v1",
        api_key="",
        message_role="user",
    )


@configclass
class VLMClaude350620Config(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="claude-3-5-sonnet-20240620",
        api_base="https://api2.aigcbest.top/v1",
        api_key="",
        message_role="user",
    )


@configclass
class VLMClaude370219Config(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="claude-3-7-sonnet-20250219",
        api_base="https://api2.aigcbest.top/v1",
        api_key="",
        message_role="user",
    )


@configclass
class VLMClaude37LatestConfig(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="claude-3-7-sonnet-latest",
        api_base="https://api2.aigcbest.top/v1",
        api_key="",
        message_role="user",
    )


@configclass
class VLMClaude4LatestConfig(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="claude-4-sonnet-latest",
        api_base="https://api2.aigcbest.top/v1",
        api_key="",
        message_role="user",
    )


@configclass
class VLMClaude40514Config(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="claude-sonnet-4-20250514",
        api_base="https://api2.aigcbest.top/v1",
        api_key="",
        message_role="user",
    )

@configclass
class VLMClaude45Config(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="claude-sonnet-4-5-20250929",
        api_base="https://api2.aigcbest.top/v1",
        api_key="",
        message_role="user",
    )


@configclass
class VLMLlamaQddConfig(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="llama-3.2-90b-vision-instruct",
        api_base="https://api2.aigcbest.top/v1",
        api_key="",
        message_role="user",
    )


@configclass
class VLMGLMConfig(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="glm-4v-plus",
        api_base="https://api2.aigcbest.top/v1",
        message_role="user",
    )


@configclass
class VLMQwen2Config(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="Qwen/Qwen2-VL-72B-Instruct",
        api_base="https://api2.aigcbest.top/v1",
        message_role="user",
    )

@configclass
class VLMDoubaoSeed18Config(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="doubao-seed-1-8-251215",
        api_base="https://api2.aigcbest.top/v1",
        message_role="user",
    )

@configclass
class VLMDoubaoVisionConfig(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="Doubao-1.5-vision-pro-32k",
        api_base="https://api2.aigcbest.top/v1",
        message_role="user",
    )


# VLMs below are calling aliyun apis
@configclass
class VLMQwenVLMaxConfig(VLMConfig):
    client_type: str = "dashscope"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="qwen-vl-max-latest",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        message_role="user",
    )


@configclass
class VLMQwenVLMax0125Config(VLMConfig):
    client_type: str = "dashscope"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="qwen-vl-max-2025-01-25",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        message_role="user",
    )


@configclass
class VLMQvqMaxConfig(VLMConfig):
    client_type: str = "dashscope"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="qvq-max-latest",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        message_role="user",
    )


@configclass
class VLMQvqMaxLatestConfig(VLMConfig):
    client_type: str = "dashscope"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="qvq-max-latest",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        message_role="user",
    )


@configclass
class VLMQvqMax0325Config(VLMConfig):
    client_type: str = "dashscope"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="qvq-max-2025-03-25",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        message_role="user",
    )


@configclass
class VLMQvq72bPreviewConfig(VLMConfig):
    client_type: str = "dashscope"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="qvq-72b-preview",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        message_role="user",
    )


@configclass
class VLMQwenVLPlusConfig(VLMConfig):
    client_type: str = "dashscope"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="qwen-vl-plus",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        message_role="user",
    )


@configclass
class VLMQwenVLPlusLatestConfig(VLMConfig):
    client_type: str = "dashscope"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="qwen-vl-plus-latest",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        message_role="user",
    )


@configclass
class VLMQwenVLPlus0125Config(VLMConfig):
    client_type: str = "dashscope"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="qwen-vl-plus-2025-01-25",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        message_role="user",
    )


@configclass
class VLMQwen25Vl32bInstruct(VLMConfig):
    client_type: str = "dashscope"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="qwen2.5-vl-32b-instruct",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        message_role="user",
    )


@configclass
class VLMQwen25Vl72bInstruct(VLMConfig):
    client_type: str = "dashscope"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="qwen2.5-vl-72b-instruct",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        message_role="user",
    )


@configclass
class VLMQwen25Vl7bInstruct(VLMConfig):
    client_type: str = "dashscope"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="qwen2.5-vl-7b-instruct",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        message_role="user",
    )


@configclass
class VLMQwen25Vl3bInstruct(VLMConfig):
    client_type: str = "dashscope"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="qwen2.5-vl-3b-instruct",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        message_role="user",
    )

@configclass
class VLMQwen3VLPlus(VLMConfig):
    client_type: str = "dashscope"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="qwen3-vl-plus",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        message_role="user",
    )


@configclass
class VLMLlama4Maverick17bConfig(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="meta/llama-4-maverick-17b-128e-instruct",
        api_base="https://api2.aigcbest.top/v1",
        message_role="user",
    )


@configclass
class VLMLlama4Scout17bConfig(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="llama-4-scout-17b-16e-instruct",
        api_base="https://api2.aigcbest.top/v1",
        message_role="user",
    )


@configclass
class VLMLlamaConfig(VLMConfig):
    client_type: str = "dashscope"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="llama-3.2-90b-vision-instruct",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        message_role="user",
    )


@configclass
class VLMQwen25Config(VLMConfig):
    client_type: str = "dashscope"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="qwen2.5-vl-72b-instruct",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        message_role="user",
    )


# VLMs below are taken from their official websites
@configclass
class VLMGLM4VConfig(VLMConfig):
    client_type: str = "zhipu"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="glm-4v-plus-0111",
        api_base="https://open.bigmodel.cn/api/paas/v4/",
        message_role="user",
    )
    # website: str = "https://open.bigmodel.cn/"


@configclass
class VLMStep1oVisionConfig(VLMConfig):
    client_type: str = "stepai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="step-1o-vision-32k",
        api_base="https://api.stepfun.com/v1",
        message_role="user",
    )
    # website: str = "https://www.stepfun.com/"


@configclass
class VLMYiVisionV2Config(VLMConfig):
    client_type: str = "lingyi"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="yi-vision-v2",
        api_base="https://api.lingyiwanwu.com/v1",
        message_role="user",
    )
    max_tokens = 2048
    # website: str = "https://www.stepfun.com/"
    prompts = VLMPromptConfig(
        type="file",
        characteristics="prompts/simple/characteristics.txt",
        # interactions="prompts/simple/interaction_info.txt",
        # histories="prompts/yi_vision_v2/history.txt",
        instructions="prompts/simple/instructions.txt",
        # user_message_template=MessageTemplate(
        #     role="user",
        #     content=[
        #         {"type": "text", "text": ""},
        #     ],
        # ),
        # system_message_template=MessageTemplate(
        #     role="system",
        #     content="",
        # ),
    )


@configclass
class VLMHumanConfig(VLMConfig):
    client_type: str = "human"
    client_cfg: VLMClientCfg = VLMClientCfg(name="human")


# todo(@xiehongzhao): Add more VLM variants for the agents with model-specific versions.
@configclass
class Gemini25ProPreview20250605(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="gemini-2.5-pro-preview-06-05",
        api_base="https://api2.aigcbest.top/v1",
        message_role="user",
    )


@configclass
class Gemini25ProPreview20250325(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="gemini-2.5-pro-preview-03-25",
        api_base="https://api2.aigcbest.top/v1",
        message_role="user",
    )


@configclass
class Gemini20ProExp20250205(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="gemini-2.0-pro-exp-20250205",
        api_base="https://api2.aigcbest.top/v1",
        message_role="user",
    )


@configclass
class Gemini20Flash(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="gemini-2.0-flash-exp",
        api_base="https://api2.aigcbest.top/v1",
        message_role="user",
    )


@configclass
class Gemini25FlashPreview20250417(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="gemini-2.5-flash-preview-04-17",
        api_base="https://api2.aigcbest.top/v1",
        message_role="user",
    )

@configclass
class Gemini3ProPreview(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="gemini-3-pro-preview",
        api_base="https://api2.aigcbest.top/v1",
        message_role="user",
    )


@configclass
class VLMGrok3Config(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="grok-3",
        api_base="https://api2.aigcbest.top/v1",
        api_key="",
        message_role="user",
    )

@configclass
class VLMGrok4Config(VLMConfig):
    client_type: str = "openai"
    client_cfg: VLMClientCfg = VLMClientCfg(
        name="grok-4",
        api_base="https://api2.aigcbest.top/v1",
        api_key="",
        message_role="user",
    )
