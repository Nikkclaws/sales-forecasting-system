"""Smoke tests for the data loading + preprocessing layer on the real dataset."""

from __future__ import annotations

import pandas as pd

from sales_forecast.data import DataLoader, Preprocessor, WalkForwardSplitter


def test_loader_returns_normalized_frame(cfg):
    df = DataLoader(cfg).load()
    assert {cfg.data.date_col, cfg.data.state_col, cfg.data.target_col}.issubset(df.columns)
    assert pd.api.types.is_datetime64_any_dtype(df[cfg.data.date_col])
    assert df[cfg.data.target_col].notna().all()
    assert df[cfg.data.state_col].nunique() >= 5


def test_preprocessor_outputs_per_state_weekly(cfg):
    df = DataLoader(cfg).load()
    out = Preprocessor(cfg).transform(df)
    assert out, "Expected at least one state to survive preprocessing."
    state, weekly = next(iter(out.items()))
    assert weekly.index.is_monotonic_increasing
    assert weekly[cfg.data.target_col].notna().all()
    # weekly grid -> all gaps are exactly 7 days
    diffs = weekly.index.to_series().diff().dropna().dt.days.unique()
    assert set(diffs.tolist()) == {7}, f"Non-weekly grid for {state}: {diffs}"


def test_walk_forward_splitter(cfg):
    df = DataLoader(cfg).load()
    out = Preprocessor(cfg).transform(df)
    weekly = next(iter(out.values()))
    folds = list(WalkForwardSplitter(cfg).split(weekly.index))
    assert folds, "Expected at least one CV fold."
    for f in folds:
        assert len(f.val_idx) == cfg.cv.horizon
        # No overlap between train and val.
        assert f.train_idx.intersection(f.val_idx).empty
