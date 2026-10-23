import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger as _loguru

_LOG_CONFIGURED = False
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _InterceptHandler(logging.Handler):

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = _loguru.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        _loguru.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_logging(level: int = logging.INFO) -> None:
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return
    _loguru.remove()

    _now = datetime.now()
    log_dir = _PROJECT_ROOT / "logs" / _now.strftime("%Y-%m-%d")
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = _now.strftime("%Y%m%d_%H%M%S")
    _loguru.add(
        str(log_dir / f"run_{stamp}.log"),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} {level: <7} [{name}] {message}",
        enqueue=True, backtrace=False, diagnose=False,
    )

    console_level = os.environ.get("LOG_CONSOLE_LEVEL", "WARNING").upper()
    _loguru.add(sys.stderr, level=console_level, colorize=True,
                format="<level>{level: <7}</level> | <level>{message}</level>")

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    _LOG_CONFIGURED = True


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    configure_logging(level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def save_json(path: str | Path, obj: Any, indent: int = 2) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=indent, default=str)
    get_logger("io").info("Saved json file: %s", str(p))
    return p
