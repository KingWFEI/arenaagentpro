
import sys
import os

from typing import Any
from loguru import logger
from datetime import datetime
import toml

CONFIG: dict[str, Any] = {}

def set_logger_level(level: str = "INFO", log_dir: str | None = None):
    # pylint: disable=line-too-long
    level = level.upper()
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> "  # noqa: E501
        "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
    )
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_name = os.path.join(
            log_dir, f"arenaagent_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        )
        logger.add(
            log_name,
            level=level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>",
        )

def load_config(config_path="config.toml", extra_params: dict | None = None):
    # pylint: disable=global-statement, global-variable-not-assigned
    global CONFIG  # noqa: PLW0603
    if len(CONFIG) > 0:
        return CONFIG
    config = toml.load(config_path)
    CONFIG = config.copy()
    if extra_params:
        CONFIG.update(extra_params)
    logger.info("Loaded config from {}", config_path)
    logger.info(CONFIG)
    return config


def update_config(update_dict: dict):
    # pylint: disable=global-statement, global-variable-not-assigned
    global CONFIG  # noqa: PLW0602
    CONFIG.update(update_dict)
