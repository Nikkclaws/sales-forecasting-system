"""Centralized logging setup."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

_INITIALIZED = False


def setup_logging(log_dir: str | Path = "logs", level: str = "INFO", filename: str = "app.log") -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    formatter = logging.Formatter(fmt)

    root = logging.getLogger()
    root.setLevel(level.upper())
    # Clear any prior handlers (e.g. uvicorn duplicates)
    for h in list(root.handlers):
        root.removeHandler(h)

    stream_h = logging.StreamHandler(sys.stdout)
    stream_h.setFormatter(formatter)
    root.addHandler(stream_h)

    file_h = logging.handlers.RotatingFileHandler(
        log_dir / filename, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_h.setFormatter(formatter)
    root.addHandler(file_h)

    # Tame chatty third-party loggers.
    for noisy in ("cmdstanpy", "prophet", "matplotlib", "optuna"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
