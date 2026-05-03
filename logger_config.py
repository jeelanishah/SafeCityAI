"""Logging configuration for SafeCityAI."""

import sys
from pathlib import Path
from loguru import logger as loguru_logger

__all__ = ["get_logger", "setup_logging"]


def setup_logging(environment: str, log_file: Path, log_level: str, log_retention: str) -> None:
    loguru_logger.remove()
    loguru_logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    loguru_logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level,
        rotation="500 MB",
        retention=log_retention,
    )


def get_logger(name: str):
    return loguru_logger.bind(name=name)
