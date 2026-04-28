"""Model registry: versioned, human-readable layout under `artifacts/registry/`."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .io import ensure_dir, load_json, save_json
from .logging import get_logger

log = get_logger(__name__)


@dataclass
class ModelVersion:
    version: str
    created_at: float
    path: Path

    def to_dict(self) -> dict:
        return {"version": self.version, "created_at": self.created_at, "path": str(self.path)}


class ModelRegistry:
    """Versioned filesystem registry. Layout:

    registry_dir/
        manifest.json                 -> {"current": "v20240417_120033"}
        v20240417_120033/
            states/
                California/
                    arima.joblib
                    sarima.joblib
                    ...
                    metrics.json
                    feature_engineer.joblib
            metadata.json
    """

    def __init__(self, root: str | Path):
        self.root = ensure_dir(root)

    # --- Versions --------------------------------------------------------- #

    def new_version(self) -> ModelVersion:
        ts = time.strftime("v%Y%m%d_%H%M%S")
        path = ensure_dir(self.root / ts)
        return ModelVersion(version=ts, created_at=time.time(), path=path)

    def list_versions(self) -> list[str]:
        return sorted([p.name for p in self.root.iterdir() if p.is_dir() and p.name.startswith("v")])

    def current_version(self) -> str | None:
        manifest = self.root / "manifest.json"
        if manifest.exists():
            return load_json(manifest).get("current")
        versions = self.list_versions()
        return versions[-1] if versions else None

    def set_current(self, version: str) -> None:
        save_json({"current": version}, self.root / "manifest.json")

    def version_path(self, version: str | None = None) -> Path:
        v = version or self.current_version()
        if v is None:
            raise FileNotFoundError("No versions registered yet.")
        return self.root / v

    # --- Per-state state -------------------------------------------------- #

    def state_dir(self, state: str, version: str | None = None) -> Path:
        return ensure_dir(self.version_path(version) / "states" / state)

    def write_metadata(self, metadata: dict, version: str | None = None) -> Path:
        path = self.version_path(version) / "metadata.json"
        save_json(metadata, path)
        return path

    def read_metadata(self, version: str | None = None) -> dict:
        path = self.version_path(version) / "metadata.json"
        if not path.exists():
            return {}
        return load_json(path)

    def gc(self, keep: int = 5) -> list[str]:
        """Remove oldest versions, keeping `keep` most recent."""
        versions = self.list_versions()
        if len(versions) <= keep:
            return []
        to_remove = versions[: len(versions) - keep]
        for v in to_remove:
            shutil.rmtree(self.root / v, ignore_errors=True)
            log.info("GC: removed registry version %s", v)
        return to_remove
