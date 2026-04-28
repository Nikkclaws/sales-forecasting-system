"""Raw data loading with robust schema and date handling."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from ..config import Config
from ..utils.logging import get_logger

log = get_logger(__name__)


class DataLoader:
    """Load raw sales records from CSV/Excel and normalize to a tidy frame.

    Output schema: columns = [date, state, target, category?]; date is `datetime64[ns]`.
    Handles the source quirk where some Excel cells are real datetimes while
    others are `DD-MM-YYYY` strings.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @property
    def source_path(self) -> Path:
        return self.cfg.path(self.cfg.data.source_path)

    def load(self) -> pd.DataFrame:
        path = self.source_path
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")
        log.info("Loading data from %s", path)
        df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
        return self._normalize(df)

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        d = self.cfg.data
        required = [d.date_col, d.state_col, d.target_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

        df = df.copy()
        df[d.date_col] = df[d.date_col].apply(self._parse_date)
        n_bad = df[d.date_col].isna().sum()
        if n_bad:
            log.warning("Dropping %d rows with unparseable dates", n_bad)
            df = df.dropna(subset=[d.date_col])

        df[d.target_col] = pd.to_numeric(df[d.target_col], errors="coerce")
        df = df.dropna(subset=[d.target_col])
        df[d.state_col] = df[d.state_col].astype(str).str.strip()

        keep_cols = [d.date_col, d.state_col, d.target_col]
        if d.category_col and d.category_col in df.columns:
            keep_cols.append(d.category_col)
        df = df[keep_cols].sort_values([d.state_col, d.date_col]).reset_index(drop=True)
        log.info(
            "Loaded %d rows | %d states | range %s..%s",
            len(df),
            df[d.state_col].nunique(),
            df[d.date_col].min().date(),
            df[d.date_col].max().date(),
        )
        return df

    @staticmethod
    def _parse_date(value: object) -> pd.Timestamp | float:
        """Parse mixed datetime objects and DD-MM-YYYY strings."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return pd.NaT
        if isinstance(value, datetime | pd.Timestamp):
            return pd.Timestamp(value)
        s = str(value).strip()
        if not s:
            return pd.NaT
        # try DD-MM-YYYY (dataset's documented format) first, then ISO, then loose.
        for fmt in ("%d-%m-%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return pd.to_datetime(s, format=fmt)
            except (ValueError, TypeError):
                continue
        return pd.to_datetime(s, errors="coerce", dayfirst=True)
