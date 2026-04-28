"""Filesystem I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_joblib(obj: Any, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    joblib.dump(obj, p)
    return p


def load_joblib(path: str | Path) -> Any:
    return joblib.load(path)


def save_json(obj: Any, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    return p


def load_json(path: str | Path) -> Any:
    with open(path) as fh:
        return json.load(fh)
