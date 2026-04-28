"""CLI entrypoint: `python -m sales_forecast.training.cli` or `sales-forecast-train`."""

from __future__ import annotations

import argparse

from ..config import load_config
from .pipeline import TrainingPipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train per-state forecasting models.")
    p.add_argument("--config", default=None, help="Path to config.yaml (auto-detected by default)")
    p.add_argument(
        "--states",
        nargs="*",
        default=None,
        help="Optional subset of states to train (default: all states with sufficient history).",
    )
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    pipe = TrainingPipeline(cfg)
    report = pipe.run(states=args.states)
    print(f"Training complete. Version={report.version}; states trained={len(report.states)}.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
