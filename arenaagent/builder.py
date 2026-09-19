from __future__ import annotations

import argparse
import importlib
import pkgutil
from pathlib import Path
from typing import Callable, Iterable

import grpc
import time
from dotenv import load_dotenv

from .agent_base import AgentBase
from arenaagent.generated.arena.agent.arena_agent_service_pb2_grpc import (
    TongTestAgentServiceStub,
)
from arenaagent.utils.config import load_config, set_logger_level
from loguru import logger

_AGENT_REGISTRY: dict[str, type[AgentBase]] = {}
_AGENT_CONFIG_ALIASES: dict[str, tuple[str, ...]] = {
    "preliminary_baseline_agent": ("preliminary_vlm_agent",),
}


def load_project_environment() -> bool:
    """Load project-local secrets without overwriting explicit shell values."""
    env_path = Path.cwd() / ".env"
    if not env_path.is_file():
        return False
    loaded = bool(load_dotenv(dotenv_path=env_path, override=False))
    logger.info("Loaded model environment from {}", env_path)
    return loaded

def Register(name: str) -> Callable[[type[AgentBase]], type[AgentBase]]:
    def decorator(cls: type[AgentBase]) -> type[AgentBase]:
        logger.info("Register.decorator")
        if not issubclass(cls, AgentBase):
            raise TypeError(f"{cls.__name__} must inherit from AgentBase")
        if name in _AGENT_REGISTRY and _AGENT_REGISTRY[name] is not cls:
            raise ValueError(f"Agent name '{name}' already registered")
        _AGENT_REGISTRY[name] = cls
        setattr(cls, "_agent_name", name)
        return cls

    return decorator

class AgentBuilder:
    def __init__(self) -> None:
        self._registry: dict[str, type[AgentBase]] = {}
        self.collect()

    def collect(self) -> dict[str, type[AgentBase]]:
        # Only keep classes that still inherit from AgentBase.
        logger.info("_AGENT_REGISTRY {}", _AGENT_REGISTRY)
        self._registry = {
            name: cls for name, cls in _AGENT_REGISTRY.items() if issubclass(cls, AgentBase)
        }
        return self._registry

    def get(self, name: str) -> type[AgentBase]:
        return self._registry[name]

    def exists(self, name: str) -> bool:
        return name in self._registry

    def build(self, name: str, *args, **kwargs) -> AgentBase:
        if not self.exists(name):
            raise KeyError(f"Agent '{name}' is not registered")
        return self._registry[name](*args, **kwargs)

    def all(self) -> dict[str, type[AgentBase]]:
        return dict(self._registry)

def _try_import_agent_modules(agent_name: str) -> None:
    if not agent_name:
        return

    def _import_and_register(module_path: str) -> None:
        try:
            mod = importlib.import_module(module_path)
        except Exception as e:
            logger.info("import error {}", e)
            return
        # If it's a package (e.g., arenaagent/vlm_agent/ without __init__.py), import its submodules too.
        module_path_is_pkg = hasattr(mod, "__path__")
        if module_path_is_pkg:
            for _, sub_mod_name, _ in pkgutil.iter_modules(mod.__path__, mod.__name__ + "."):
                try:
                    importlib.import_module(sub_mod_name)
                except Exception:
                    continue

    candidates = (
        f"arenaagent.{agent_name}",
        f"arenaagent.{agent_name}.{agent_name}",  # handle nested packages like arenaagent/<name>/<name>.py
        f"arenaagent.{agent_name}_agent",
    )
    for module in candidates:
        _import_and_register(module)
        if _AGENT_REGISTRY:
            return


def _resolve_grpc_target(config: dict) -> str:
    grpc_cfg = config.get("grpc", {}) if isinstance(config, dict) else {}
    if isinstance(grpc_cfg, dict):
        host = grpc_cfg.get("host") or grpc_cfg.get("ip")
        port = grpc_cfg.get("port")
        if host and port:
            return f"{host}:{port}"
        for key in ("endpoint", "address", "target"):
            value = grpc_cfg.get(key)
            if value:
                return value
    for key in (
        "grpc_endpoint",
        "grpc_address",
        "agent_grpc_endpoint",
        "arena_grpc_endpoint",
    ):
        value = config.get(key) if isinstance(config, dict) else None
        if value:
            return value
    return "127.0.0.1:50051"


def _resolve_agent_params(config: dict, agent_name: str) -> dict:
    if not isinstance(config, dict):
        return {}

    candidate_names = (agent_name, *_AGENT_CONFIG_ALIASES.get(agent_name, ()))

    for candidate in candidate_names:
        direct_cfg = config.get(candidate, {})
        if isinstance(direct_cfg, dict) and direct_cfg:
            if candidate != agent_name:
                logger.warning(
                    "Using legacy config section '{}' for agent '{}'. Please rename the section to '[{}]'.",
                    candidate,
                    agent_name,
                    agent_name,
                )
            return direct_cfg

    for section_name, section_cfg in config.items():
        if not isinstance(section_cfg, dict) or not section_cfg:
            continue
        section_agent_name = section_cfg.get("name")
        if section_agent_name not in candidate_names:
            continue
        if section_name != agent_name:
            logger.warning(
                "Using config section '{}' for agent '{}' because its inner name is '{}'.",
                section_name,
                agent_name,
                section_agent_name,
            )
        return section_cfg
    # 优先匹配与 agent_name 同名的顶层配置
    direct_cfg = config.get(agent_name, {})
    if isinstance(direct_cfg, dict) and direct_cfg:
        return direct_cfg
    agents_cfg = config.get("agents", {})
    if isinstance(agents_cfg, dict):
        for candidate in candidate_names:
            agent_params = agents_cfg.get(candidate, {})
            if isinstance(agent_params, dict) and agent_params:
                if candidate != agent_name:
                    logger.warning(
                        "Using legacy nested config section 'agents.{}' for agent '{}'.",
                        candidate,
                        agent_name,
                    )
                return agent_params
    agent_params = config.get("agent", {})
    if isinstance(agent_params, dict):
        return agent_params
    agent_params = config.get("agent_params", {})
    if isinstance(agent_params, dict):
        return agent_params
    return {}


def _create_channel(target: str) -> grpc.Channel:
    return grpc.insecure_channel(
        target,
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ],
    )

def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a registered agent by name.")
    parser.add_argument("--agent_name", required=True, help="Registered agent name")
    parser.add_argument("--config", default="config.toml", help="Path to agent config directory")
    parser.add_argument("--vlm_model", default="VLMGPT5Config", help="Model name used by vlm_agent (must be in vlm_config.__ALL__).")
    parser.add_argument("--get_vlm_model", action="store_true", help="List all available VLM models and exit.")
    parser.add_argument("--run_times", default=1, help="Number of times to run the agent.")
    return parser.parse_args(argv)

def _release_agent_quietly(agent) -> None:
    """Fully unregister a failed agent instead of only releasing its UE state."""
    if agent is None:
        return
    try:
        if bool(getattr(agent, "connected", False)):
            agent._disconnect()
        else:
            agent.deinit()
    except Exception as exc:  # pragma: no cover - 尽力而为的清理
        logger.warning("cleanup after failure failed: {}", exc)


def main() -> None:
    args = _parse_args()
    load_project_environment()

    # 如果只需列出 VLM 模型，提前返回
    if args.get_vlm_model:
        try:
            from arenaagent.vlm_agent import vlm_config as _vc  # noqa: WPS433
            models = getattr(_vc, "__ALL__", [])
            print("\n".join(models))
        except Exception as exc:  # pragma: no cover - 仅用于命令行工具
            logger.error("Failed to load VLM models: {}", exc)
        return

    config = load_config(config_path=args.config)
    log_dir = config.get("log_dir", "logs") if isinstance(config, dict) else "logs"
    set_logger_level(level="debug", log_dir=log_dir)

    _try_import_agent_modules(args.agent_name)

    builder = AgentBuilder()
    if not builder.exists(args.agent_name):
        logger.error(f"Agent '{args.agent_name}' is not registered")
        logger.info("Available agents: {}", ", ".join(builder.all().keys()) or "<none>")
        return

    grpc_target = _resolve_grpc_target(config)

    params = _resolve_agent_params(config, args.agent_name)
    # 在 vlm_agent 系列和 final_baseline_agent 并且传入 --vlm_model 时，创建对应的 VLMConfig 实例并传递给 load
    if args.agent_name in {"vlm_agent", "preliminary_baseline_agent", "final_baseline_agent"} and args.vlm_model:
        try:
            from arenaagent.vlm_agent import vlm_config as _vc  # noqa: WPS433

            if not args.vlm_model or not hasattr(_vc, args.vlm_model):
                logger.error("VLM model '{}' not found in vlm_config.__ALL__", args.vlm_model)
                return
            vlm_cfg_cls = getattr(_vc, args.vlm_model)
            params["vlm_config"] = vlm_cfg_cls()
            logger.info("Using VLM model: {}", args.vlm_model)
        except Exception as exc:  # pragma: no cover - CLI guard
            logger.error("Failed to initialize vlm_model '{}': {}", args.vlm_model, exc)
            return

    runtimes = int(args.run_times) if str(args.run_times).isdigit() else 1
    if runtimes < 1:
        logger.error("run_times must be at least 1")
        return

    # Do not reuse one TongSim client across subjects.  In the 912 runtime each
    # client owns its character camera/capture stream, and close() is the only
    # reliable atomic cleanup path.  Reusing a client requires an explicit
    # character destroy followed by a later client close, which can leave stale
    # capture tasks in UE and cause an access-violation crash.
    completed = 0
    attempts = 0
    max_attempts = max(runtimes * 3, runtimes + 3)
    previous_completed_counting_signature = None
    while completed < runtimes and attempts < max_attempts:
        attempts += 1
        channel = _create_channel(grpc_target)
        stub = TongTestAgentServiceStub(channel)
        time.sleep(2)  # 等待连接稳定
        agent = None
        try:
            agent = builder.build(args.agent_name, stub=stub, channel=channel)
            if previous_completed_counting_signature is not None:
                agent._previous_completed_counting_signature = previous_completed_counting_signature
            agent.load(params)
            if not bool(getattr(agent, "connected", False)):
                raise ConnectionError("Agent did not connect to the task server")
            agent.run()
            completed += 1
            completed_signature = getattr(agent, "_completed_counting_subject_signature", None)
            if isinstance(completed_signature, str):
                previous_completed_counting_signature = completed_signature
        except Exception as e:
            logger.opt(exception=True).warning("error trace back {}", e)
            _release_agent_quietly(agent)
            continue
        finally:
            logger.info("Closing Arena gRPC channel")
            channel.close()
            time.sleep(5)  # 避免短时间内重复创建 Arena 连接

    if completed < runtimes:
        logger.error(
            "Completed only {}/{} requested runs after {} attempts",
            completed,
            runtimes,
            attempts,
        )


if __name__ == "__main__":
    main()
