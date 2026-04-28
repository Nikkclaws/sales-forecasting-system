"""Pytest fixtures shared across the test suite."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sales_forecast.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    """Load the project config but redirect log/artifact dirs into a temp directory."""
    os.chdir(REPO_ROOT)
    c = load_config()
    tmp = tmp_path_factory.mktemp("artifacts")
    c.project.log_dir = str(tmp / "logs")
    c.project.artifacts_dir = str(tmp / "artifacts")
    c.project.registry_dir = str(tmp / "artifacts" / "registry")
    c.project_root = REPO_ROOT
    return c


@pytest.fixture(scope="session")
def synthetic_weekly_series() -> pd.Series:
    """Deterministic seasonal+trend weekly series for fast model checks."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2018-01-07", periods=260, freq="W-SUN")
    t = np.arange(len(idx))
    seasonal = 10 * np.sin(2 * np.pi * t / 52.1775)
    trend = 0.1 * t
    noise = rng.normal(0, 1.0, size=len(idx))
    y = 100 + trend + seasonal + noise
    return pd.Series(y, index=idx, name="y")
