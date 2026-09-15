from __future__ import annotations

import copy
import os
import string
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from arenaagent.utils.configclass import configclass

from loguru import logger
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


@dataclass
class PromptAssets:
    characteristics: str
    interactions: str
    history: str
    instructions: str


class PromptGenerator:
    def __init__(self, config: VLMPromptConfig, search_paths: list[str] | None = None) -> None:
        self.config = config
        self._search_paths = search_paths or [
            ".",
            os.path.dirname(os.path.abspath(__file__)),
            os.path.expanduser("~"),
        ]
        self._assets: PromptAssets | None = None
        self._file_cache: dict[str, str] = {}

    def Generate(
        self,
        variables: dict[str, Any] | None = None,
        image: str | None = None,
        context_messages: list[dict[str, Any]] | None = None,
        last_json_parse_message: str = "",
        include_instructions: bool = True,
    ) -> list[dict[str, Any]]:
        context_messages = list(context_messages or [])
        assets = self._load_assets()
        render_ctx = self._build_render_context(variables or {})

        characteristics_text = self._render_template(assets.characteristics, render_ctx)

        system_message = self._build_system_message(
            characteristics_text=characteristics_text,
            instructions=assets.instructions,
            include_instructions=include_instructions,
        )

        interaction_text = self._render_template(assets.interactions, render_ctx)

        interaction_text = self._append_hand_status(interaction_text, variables or {})
        if last_json_parse_message:
            interaction_text = f"{interaction_text}\n{last_json_parse_message}"

        user_message = self._build_user_message(
            interaction_text=interaction_text,
            image=image,
        )

        return [system_message] + context_messages + [user_message]

    def _load_assets(self) -> PromptAssets:
        if self._assets is not None:
            return self._assets

        if self.config.type == "file":
            self._assets = self._load_prompts_from_file()
            return self._assets

        if self.config.type == "text":
            self._assets = PromptAssets(
                characteristics=self.config.characteristics,
                interactions=self.config.interactions,
                history=self.config.histories,
                instructions=self.config.instructions or "",
            )
            return self._assets

        raise ValueError(f"Unsupported prompt type: {self.config.type}")

    def _load_prompts_from_file(self) -> PromptAssets:
        characteristics = self._read_prompt_file(self.config.characteristics)
        interactions = self._read_prompt_file(self.config.interactions)
        history = self._read_prompt_file(self.config.histories)
        instructions = ""
        if self.config.instructions:
            instructions = self._read_prompt_file(self.config.instructions)
        return PromptAssets(
            characteristics=characteristics,
            interactions=interactions,
            history=history,
            instructions=instructions,
        )

    def _read_prompt_file(self, relative_path: str) -> str:
        for base in self._search_paths:
            candidate = os.path.join(base, relative_path)
            if os.path.isfile(candidate):
                if candidate in self._file_cache:
                    return self._file_cache[candidate]
                with open(candidate, "r", encoding="utf-8") as handle:
                    content = handle.read()
                    self._file_cache[candidate] = content
                    return content
        raise FileNotFoundError(f"Prompt file not found: {relative_path}")

    def _build_system_message(
        self,
        characteristics_text: str,
        instructions: str,
        include_instructions: bool,
    ) -> dict[str, Any]:
        message = self._copy_template(self.config.system_message_template)
        text = characteristics_text + (instructions if include_instructions else "")
        return self._set_nested(message, self.config.system_text_path, text)

    def _build_user_message(self, interaction_text: str, image: str | None) -> dict[str, Any]:
        message = self._copy_template(self.config.user_message_template)
        message = self._set_nested(message, self.config.user_text_path, interaction_text)
        if image is not None:
            message = self._set_nested(message, self.config.user_image_path, image)
        return message

    def _append_hand_status(self, interaction_text: str, variables: dict[str, Any]) -> str:
        object_in_hand = variables.get("object_in_hand")
        if object_in_hand:
            return f"{interaction_text}\n 当前手上物品为：{object_in_hand}"
        return f"{interaction_text}\n 当前手上无物品。"

    @staticmethod
    def _copy_template(template: MessageTemplate) -> dict[str, Any]:
        if hasattr(template, "to_dict"):
            return copy.deepcopy(template.to_dict())
        return copy.deepcopy(template)

    @staticmethod
    def _build_render_context(variables: dict[str, Any]) -> dict[str, Any]:
        context = dict(variables)
        context.setdefault("self", SimpleNamespace(**variables))
        return _SafeFormatDict(context)

    @staticmethod
    def _render_template(template: str, context: dict[str, Any]) -> str:
        try:
            return template.format_map(context)
        except Exception as e:
            logger.warning("Failed to render template with error: {}", e)
            return template

    @staticmethod
    def _set_nested(message: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
        # 复用通用的设置逻辑，保持与 _set_nested_value 行为一致
        return PromptGenerator._set_nested_value(message, path, value)

    @staticmethod
    def _set_nested_value(obj, path, value):  # noqa: PLR0912
        """
        通过字符串路径设置嵌套字典的值

        Args:
            obj: 要修改的对象(字典)
            path: 访问路径，支持以下格式:
                - "['key1']['key2']"
                - "key1.key2"
                - "key1[0].key2"
            value: 要设置的值

        Returns:
            修改后的对象

        Example:
            data = {"content": [{"type": "image"}, {"type": "text", "text": ""}]}
            set_nested_value(data, "['content'][1]['text']", "hello")
            # 或
            set_nested_value(data, "content.1.text", "hello")
        """
        # 如果是 ['key'] 形式的路径，转换成 key 形式
        if path.startswith("["):
            # 去掉 [] 并分割
            keys = [k.strip("'\"[]") for k in path.split("][")]
        else:
            # 处理 key1.key2 或 key1[0].key2 形式
            keys = []
            current = ""
            for char in path:
                if char == ".":
                    if current:
                        keys.append(current)
                        current = ""
                elif char == "[":
                    if current:
                        keys.append(current)
                    current = ""
                elif char == "]":
                    if current:
                        keys.append(current)
                    current = ""
                else:
                    current += char
            if current:
                keys.append(current)

        # 转换数字索引为整数
        keys = [int(k) if k.isdigit() else k for k in keys]

        # 递归设置值
        current = obj
        for i, key in enumerate(keys[:-1]):
            if isinstance(current, dict):
                if key not in current:
                    # 根据下一个key的类型创建相应的空容器
                    next_key = keys[i + 1]
                    current[key] = [] if isinstance(next_key, int) else {}
                current = current[key]
            elif isinstance(current, list):
                while len(current) <= key:
                    # 根据下一个key的类型创建相应的空容器
                    next_key = keys[i + 1]
                    current.append([] if isinstance(next_key, int) else {})
                current = current[key]
            else:
                raise TypeError(f"Cannot set key '{key}' on object of type {type(current)}")

        # 设置最终值
        last_key = keys[-1]
        if isinstance(current, dict):
            current[last_key] = value
        elif isinstance(current, list):
            while len(current) <= last_key:
                current.append(None)
            current[last_key] = value
        else:
            raise TypeError(f"Cannot set key '{last_key}' on object of type {type(current)}")

        return obj

    @staticmethod
    def get_nested_value(obj, path):  # noqa: PLR0912
        """
        通过字符串路径获取嵌套字典的值

        Args:
            obj: 要访问的对象(字典)
            path: 访问路径，支持以下格式:
                - "['key1']['key2']"
                - "key1.key2"
                - "key1[0].key2"
        Returns:
            获取到的值
        """
        # 如果是 ['key'] 形式的路径，转换成 key 形式
        if path.startswith("["):
            # 去掉 [] 并分割
            keys = [k.strip("'\"[]") for k in path.split("][")]
        else:
            # 处理 key1.key2 或 key1[0].key2 形式
            keys = []
            current = ""
            for char in path:
                if char == ".":
                    if current:
                        keys.append(current)
                        current = ""
                elif char == "[":
                    if current:
                        keys.append(current)
                    current = ""
                elif char == "]":
                    if current:
                        keys.append(current)
                    current = ""
                else:
                    current += char
            if current:
                keys.append(current)

        # 转换数字索引为整数
        keys = [int(k) if k.isdigit() else k for k in keys]

        # 递归获取值
        current = obj
        for key in keys:
            if isinstance(current, dict):
                current = current[key]
            elif isinstance(current, list):
                current = current[key]
            else:
                raise TypeError(f"Cannot get key '{key}' on object of type {type(current)}")

        return current


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _parse_path(path: str) -> list[Any]:
    parts: list[Any] = []
    start = 0
    while True:
        left = path.find("[", start)
        if left == -1:
            break
        right = path.find("]", left)
        if right == -1:
            break
        token = path[left + 1 : right].strip()
        if token.startswith(("'", '"')) and token.endswith(("'", '"')):
            parts.append(token[1:-1])
        else:
            try:
                parts.append(int(token))
            except ValueError:
                parts.append(token)
        start = right + 1
    return parts
