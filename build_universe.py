"""Universe rebuild entry point (weekly, before training).

Regenerates ``config/universe.csv`` from the JPX listing plus a liquidity
screen. Fail-safe: any error keeps the existing file and exits 0 so the
weekly pipeline can continue with the previous universe.

Usage:
    python build_universe.py [--config config/config.yaml]
"""

from __future__ import annotations

import argparse
#import logging

from stocklab.config import Config
from stocklab.data.universe_builder import build_universe
from stocklab.logger import get_logger, log_event, setup_logging

logger = get_logger("stocklab.build_universe")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="stocklab universe rebuild")
    parser.add_argument("--config", default="config/config.yaml", help="config file path")
    args = parser.parse_args()

    setup_logging()
    cfg = Config.load(args.config)
    if cfg.universe.mode != "auto":
        log_event(logger, "Universe mode is static; nothing to do")
        return 0
    try:
        universe = build_universe(cfg.universe, cfg.data.universe_file)
        log_event(logger, "Universe rebuild complete", n_tickers=len(universe))
    except Exception:
        logger.exception("Universe rebuild failed; keeping the existing universe.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
