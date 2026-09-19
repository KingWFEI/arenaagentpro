from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Any, Callable
from loguru import logger

_CLIENT_REGISTRY: dict[str, type["Client"]] = {}
_ERROR_RESPONSE = """
[
    {
        "think": "error occurred when sending message to VLA. Terminating the subject.",
        "action": "finish_task",
        "output": 0
    }
]
        """

def RegisterClient(client_type: str) -> Callable[[type["Client"]], type["Client"]]:
    def decorator(cls: type["Client"]) -> type["Client"]:
        key = client_type.lower()
        if key in _CLIENT_REGISTRY and _CLIENT_REGISTRY[key] is not cls:
            raise ValueError(f"Client type '{client_type}' already registered")
        _CLIENT_REGISTRY[key] = cls
        setattr(cls, "_client_type", key)
        return cls

    return decorator


@dataclass
class ClientResponse:
    text: str
    usage: Any | None = None
    raw: Any | None = None
    token_usage: dict[str, int] | None = None


class Client:
    def __init__(self, native_client: Any | None = None) -> None:
        self.native_client = native_client
        self.last_response: ClientResponse | None = None

    def __getattr__(self, item: str) -> Any:
        if self.native_client is None:
            raise AttributeError(item)
        return getattr(self.native_client, item)

    def invoke(self, message: Any, max_retries: int = 3, base_delay: float = 0.5) -> ClientResponse:
        for attempt in range(max_retries):
            try:
                response = self._invoke(message)
                response.token_usage = self._compute_token_usage(response, message)
                self.last_response = response
                return response
            except Exception as exc:
                self._rebuild()
                logger.warning("invoke with error {}", exc)
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt) + random.uniform(0, 0.5)
                    time.sleep(delay)

        error_response = ClientResponse(text=_ERROR_RESPONSE, usage=None, raw=None, token_usage=None)

        return error_response

    def _invoke(self, message: Any) -> ClientResponse:
        raise NotImplementedError()

    def _rebuild(self) -> None:
        return

    @staticmethod
    def _compute_token_usage(response: ClientResponse, message: Any) -> dict[str, int] | None:
        """
        统计 token 使用量：优先读取返回的 usage 字段，缺失时做粗略估算。
        """
        usage = getattr(response, "usage", None)

        def _to_int(val: Any) -> int | None:
            try:
                return int(val)
            except Exception:
                return None

        # 1) 优先从 usage 提取
        if usage is not None:
            prompt = _to_int(getattr(usage, "prompt_tokens", None) or getattr(usage, "prompt_tokens_total", None))
            completion = _to_int(getattr(usage, "completion_tokens", None))
            total = _to_int(getattr(usage, "total_tokens", None))

            if isinstance(usage, dict):
                prompt = prompt or _to_int(usage.get("prompt_tokens") or usage.get("prompt_tokens_total"))
                completion = completion or _to_int(usage.get("completion_tokens"))
                total = total or _to_int(usage.get("total_tokens"))

            if any(v is not None for v in (prompt, completion, total)):
                if total is None and prompt is not None and completion is not None:
                    total = prompt + completion
                if prompt is None and total is not None and completion is not None:
                    prompt = total - completion
                if completion is None and total is not None and prompt is not None:
                    completion = total - prompt
                return {
                    "prompt_tokens": prompt or 0,
                    "completion_tokens": completion or 0,
                    "total_tokens": total or (prompt or 0) + (completion or 0),
                }

        # 2) 简易估算：按字符长度估计
        def _estimate_tokens(text: str) -> int:
            return max(1, len(text) // 4)

        prompt_tokens = 0
        if isinstance(message, list):
            for item in message:
                if isinstance(item, dict):
                    prompt_tokens += _estimate_tokens(str(item.get("content", "")))
                    prompt_tokens += _estimate_tokens(str(item.get("role", "")))
                else:
                    prompt_tokens += _estimate_tokens(str(item))
        else:
            prompt_tokens = _estimate_tokens(str(message))

        completion_tokens = _estimate_tokens(response.text or "")
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }


class ClientFactory:
    def __init__(self) -> None:
        self._registry: dict[str, type[Client]] = {}
        self.collect()

    def collect(self) -> dict[str, type[Client]]:
        self._registry = dict(_CLIENT_REGISTRY)
        return self._registry

    def build(self, client_type: str, *args, **kwargs) -> Client:
        key = (client_type or "").lower()
        if key not in self._registry:
            raise KeyError(f"Client type '{client_type}' is not registered")
        return self._registry[key](*args, **kwargs)


def _resolve_api_key(configured: str | None, env_name: str, fallback: str = "") -> str:
    configured_value = configured or ""
    if configured_value:
        return configured_value
    env_value = os.getenv(env_name, "")
    if env_value:
        return env_value
    if fallback:
        return fallback
    raise AssertionError(f"{env_name} is not set")


def _resolve_model_name(configured: str | None, env_name: str) -> str:
    model_value = (configured or "").strip()
    if model_value:
        return model_value
    env_value = os.getenv(env_name, "").strip()
    if env_value:
        return env_value
    raise AssertionError(f"Model name is empty. Set cfg.name or {env_name}.")


def _build_openai_client(
    api_base: str,
    api_key: str,
    request_timeout_seconds: float | None = None,
    native_max_retries: int | None = None,
) -> Any:
    from openai import OpenAI

    options: dict[str, Any] = {"api_key": api_key, "base_url": f"{api_base}"}
    if request_timeout_seconds is not None:
        options["timeout"] = request_timeout_seconds
    if native_max_retries is not None:
        options["max_retries"] = native_max_retries
    return OpenAI(**options)


def _extract_text(response: Any) -> str:
    choice = response.choices[0]
    message = getattr(choice, "message", None)
    if message is None:
        return ""
    return getattr(message, "content", "")


@RegisterClient("azure")
class AzureClient(Client):
    def __init__(self, cfg: Any) -> None:
        from openai import AzureOpenAI

        self._cfg = cfg
        self._azure_cls = AzureOpenAI
        native_client = self._build_native()
        super().__init__(native_client)

    def _build_native(self) -> Any:
        api_key = _resolve_api_key(getattr(self._cfg, "api_key", ""), "AZURE_OPENAI_API_KEY")
        setattr(self._cfg, "api_key", api_key)
        return self._azure_cls(
            api_key=api_key,
            api_version=getattr(self._cfg, "api_version", None),
            azure_endpoint=f"{self._cfg.api_base}",
        )

    def _rebuild(self) -> None:
        self.native_client = self._build_native()

    def _invoke(self, message: Any) -> ClientResponse:
        model_name = _resolve_model_name(getattr(self._cfg, "name", ""), "AZURE_OPENAI_MODEL")
        response = self.native_client.chat.completions.create(
            model=model_name,
            messages=message,
        )
        return ClientResponse(text=_extract_text(response), usage=getattr(response, "usage", None), raw=response)


@RegisterClient("openai")
class OpenAIClient(Client):
    def __init__(self, cfg: Any) -> None:
        self._cfg = cfg
        super().__init__(self._build_native())

    def _build_native(self) -> Any:
        api_key = _resolve_api_key(getattr(self._cfg, "api_key", ""), "OPENAI_API_KEY")
        setattr(self._cfg, "api_key", api_key)
        return _build_openai_client(
            self._cfg.api_base,
            api_key,
            request_timeout_seconds=getattr(self._cfg, "request_timeout_seconds", None),
            native_max_retries=getattr(self._cfg, "native_max_retries", None),
        )

    def _rebuild(self) -> None:
        self.native_client = self._build_native()

    def _invoke(self, message: Any) -> ClientResponse:
        model_name = _resolve_model_name(getattr(self._cfg, "name", ""), "OPENAI_MODEL")
        chat_completion_kwargs = dict(
            getattr(self._cfg, "chat_completion_kwargs", None) or {}
        )
        response = self.native_client.chat.completions.create(
            model=model_name,
            messages=message,
            **chat_completion_kwargs,
        )
        return ClientResponse(text=_extract_text(response), usage=getattr(response, "usage", None), raw=response)


@RegisterClient("ark")
class ArkClient(Client):
    def __init__(self, cfg: Any) -> None:
        from volcenginesdkarkruntime import Ark

        self._cfg = cfg
        self._ark_cls = Ark
        super().__init__(self._build_native())

    def _build_native(self) -> Any:
        api_key = _resolve_api_key(getattr(self._cfg, "api_key", ""), "ARK_API_KEY")
        setattr(self._cfg, "api_key", api_key)
        return self._ark_cls(api_key=api_key, base_url=f"{self._cfg.api_base}")

    def _rebuild(self) -> None:
        self.native_client = self._build_native()

    def _invoke(self, message: Any) -> ClientResponse:
        model_name = _resolve_model_name(getattr(self._cfg, "name", ""), "ARK_MODEL")
        response = self.native_client.chat.completions.create(
            model=model_name,
            messages=message,
        )
        return ClientResponse(text=_extract_text(response), usage=getattr(response, "usage", None), raw=response)


@RegisterClient("dashscope")
class DashScopeClient(Client):
    def __init__(self, cfg: Any) -> None:
        self._cfg = cfg
        super().__init__(self._build_native())

    def _build_native(self) -> Any:
        api_key = _resolve_api_key(
            getattr(self._cfg, "api_key", ""),
            "DASHSCOPE_API_KEY"
        )
        setattr(self._cfg, "api_key", api_key)
        return _build_openai_client(self._cfg.api_base, api_key)

    def _rebuild(self) -> None:
        self.native_client = self._build_native()

    def _invoke(self, message: Any) -> ClientResponse:
        model_name = _resolve_model_name(getattr(self._cfg, "name", ""), "DASHSCOPE_MODEL")
        response = self.native_client.chat.completions.create(
            model=model_name,
            messages=message,
        )
        return ClientResponse(text=_extract_text(response), usage=getattr(response, "usage", None), raw=response)


@RegisterClient("zhipu")
class ZhiPuClient(Client):
    def __init__(self, cfg: Any) -> None:
        self._cfg = cfg
        super().__init__(self._build_native())

    def _build_native(self) -> Any:
        api_key = _resolve_api_key(getattr(self._cfg, "api_key", ""), "ZHIPU_API_KEY")
        setattr(self._cfg, "api_key", api_key)
        return _build_openai_client(self._cfg.api_base, api_key)

    def _rebuild(self) -> None:
        self.native_client = self._build_native()

    def _invoke(self, message: Any) -> ClientResponse:
        model_name = _resolve_model_name(getattr(self._cfg, "name", ""), "ZHIPU_MODEL")
        response = self.native_client.chat.completions.create(
            model=model_name,
            messages=message,
        )
        return ClientResponse(text=_extract_text(response), usage=getattr(response, "usage", None), raw=response)


@RegisterClient("stepai")
class StepAIClient(Client):
    def __init__(self, cfg: Any) -> None:
        self._cfg = cfg
        super().__init__(self._build_native())

    def _build_native(self) -> Any:
        api_key = _resolve_api_key(getattr(self._cfg, "api_key", ""), "STEPAI_API_KEY")
        setattr(self._cfg, "api_key", api_key)
        return _build_openai_client(self._cfg.api_base, api_key)

    def _rebuild(self) -> None:
        self.native_client = self._build_native()

    def _invoke(self, message: Any) -> ClientResponse:
        model_name = _resolve_model_name(getattr(self._cfg, "name", ""), "STEPAI_MODEL")
        response = self.native_client.chat.completions.create(
            model=model_name,
            messages=message,
        )
        return ClientResponse(text=_extract_text(response), usage=getattr(response, "usage", None), raw=response)


@RegisterClient("lingyi")
class LingyiClient(Client):
    def __init__(self, cfg: Any) -> None:
        self._cfg = cfg
        super().__init__(self._build_native())

    def _build_native(self) -> Any:
        api_key = _resolve_api_key(getattr(self._cfg, "api_key", ""), "LINGYI_API_KEY")
        setattr(self._cfg, "api_key", api_key)
        return _build_openai_client(self._cfg.api_base, api_key)

    def _rebuild(self) -> None:
        self.native_client = self._build_native()

    def _invoke(self, message: Any) -> ClientResponse:
        response = self.native_client.chat.completions.create(
            model=self._cfg.name,
            messages=message,
        )
        return ClientResponse(text=_extract_text(response), usage=getattr(response, "usage", None), raw=response)


@RegisterClient("human")
class HumanClient(Client):
    def __init__(self, cfg: Any | None = None) -> None:
        super().__init__(None)

    def _invoke(self, message: Any) -> ClientResponse:
        hint = getattr(self, "prompt_hint", "")
        prompt = f"{hint} " if hint else ""
        text = input(f"{prompt}Please input the response: ")
        return ClientResponse(text=text)


@RegisterClient("gpt-agent")
class GPTAgentClient(Client):
    def __init__(
        self,
        cfg: Any,
        model: str = "o3-2025-04-16",
        base_url: str = "",
    ) -> None:
        self._cfg = cfg
        self._instructions = (
            "you are a little girl in a virtual world called TongTest. "
            "You can see through the camera images and interact with the objects "
            "in the world to complete your tasks."
        )
        self._default_model = model
        self._default_base_url = base_url
        super().__init__(self._build_native())

    def _build_native(self) -> Any:
        from agents import Agent
        from agents.extensions.models.litellm_model import LitellmModel

        api_key = _resolve_api_key(getattr(self._cfg, "api_key", ""), "AZURE_OPENAI_API_KEY")
        model_name = getattr(self._cfg, "name", "") or self._default_model
        base_url = getattr(self._cfg, "api_base", "") or self._default_base_url
        setattr(self._cfg, "api_key", api_key)
        setattr(self._cfg, "name", model_name)
        setattr(self._cfg, "api_base", base_url)
        return Agent(
            name="litellm-model-agent",
            instructions=self._instructions,
            model=LitellmModel(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
            ),
        )

    def _rebuild(self) -> None:
        self.native_client = self._build_native()

    def _invoke(self, message: Any) -> ClientResponse:
        from agents import Runner

        response = Runner.run_sync(self.native_client, input=message)
        text = getattr(response, "final_output", "")
        return ClientResponse(text=text, usage=getattr(response, "usage", None), raw=response)
