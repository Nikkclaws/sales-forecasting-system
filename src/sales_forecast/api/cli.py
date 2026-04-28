"""CLI launcher for the FastAPI service."""

from __future__ import annotations

import argparse

import uvicorn

from ..config import load_config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the sales forecasting API.")
    p.add_argument("--config", default=None, help="Path to config.yaml (auto-detected by default)")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    host = args.host or cfg.api.host
    port = args.port or cfg.api.port
    uvicorn.run(
        "sales_forecast.api.app:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level=cfg.project.log_level.lower(),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
