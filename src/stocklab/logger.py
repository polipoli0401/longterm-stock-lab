"""Structured logging module.

Every processing step is written as JSON Lines to both stdout and a file
under ``logs/`` so results can be analyzed after the fact.

Each record carries a run id (``run_id``) so a single execution can be
traced end to end. Never pass secrets (tokens, URLs, ...) into the logs.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUN_ID: str = uuid.uuid4().hex[:12]

_EXTRA_KEY = "extra_data"


class JsonFormatter(logging.Formatter):
    """Formats a log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "run_id": RUN_ID,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, _EXTRA_KEY, None)
        if extra:
            payload["data"] = extra
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(log_dir: str | Path = "logs", level: int = logging.INFO) -> Path:
    """Configure the root logger and return the log-file path.

    Args:
        log_dir: Output directory for the JSONL log file.
        level: Log level.

    Returns:
        Path of the created log file.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_path = log_path / f"run_{datetime.now():%Y%m%d_%H%M%S}_{RUN_ID}.jsonl"

    formatter = JsonFormatter()
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(stream)
    root.addHandler(file_handler)

    # Silence noisy third-party loggers.
    for noisy in ("yfinance", "urllib3", "peewee"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return file_path


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name)


def log_event(logger: logging.Logger, message: str, level: int = logging.INFO, **data: Any) -> None:
    """Log an event with attached structured data.

    Args:
        logger: Target logger.
        message: Human-readable message.
        level: Log level.
        **data: Arbitrary structured payload stored under the ``data`` key.
    """
    logger.log(level, message, extra={_EXTRA_KEY: data})
