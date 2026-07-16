"""Standalone backtest entry point (manual runs).

Evaluates any single model with purged walk-forward analysis and saves a
report. Useful for experimenting with holding periods and model choices
(the weekly job runs the same evaluation for every candidate
automatically).

Usage:
    python backtest.py --model ridge [--holding-days 63] [--top-n 3]
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime

from stocklab.backtest.engine import BacktestEngine
from stocklab.config import Config
from stocklab.logger import get_logger, log_event, setup_logging
from stocklab.models.trainer import ModelTrainer
from stocklab.pipeline import prepare_training_data
from stocklab.report.generator import ReportGenerator

logger = get_logger("stocklab.backtest")


def run(config_path: str, model_name: str, holding_days: int | None, top_n: int | None) -> int:
    """Run a single-model backtest.

    Args:
        config_path: Path to the config file.
        model_name: Model to evaluate.
        holding_days: Override for the holding period (optional).
        top_n: Override for the number of positions (optional).

    Returns:
        Exit code (0 = success).
    """
    cfg = Config.load(config_path)
    bt_cfg = cfg.backtest
    if holding_days:
        bt_cfg = replace(bt_cfg, holding_days=holding_days)
    if top_n:
        bt_cfg = replace(bt_cfg, top_n=top_n)

    data = prepare_training_data(cfg)
    trainer = ModelTrainer(cfg.model)
    predictions = trainer.walk_forward(data.panel, data.feature_cols, model_name)

    engine = BacktestEngine(bt_cfg, data.prices, data.bench_close, data.filters)
    result = engine.run(predictions, model_name=model_name)

    today = datetime.now().strftime("%Y-%m-%d")
    reporter = ReportGenerator(bt_cfg.report_dir)
    report_path = reporter.save(
        reporter.backtest_markdown(
            result,
            extra={
                "Evaluation": "Purged walk-forward (OOS predictions only)",
                "Run date": today,
                "Holding": f"{bt_cfg.holding_days} trading days / Top{bt_cfg.top_n}",
            },
        ),
        f"backtest/{today}_{model_name}_manual.md",
    )
    equity_path = reporter.dir / "backtest" / f"{today}_{model_name}_equity.csv"
    result.equity.to_csv(equity_path)

    log_event(
        logger,
        "Manual backtest complete",
        report=str(report_path),
        equity_csv=str(equity_path),
    )
    return 0


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="stocklab standalone backtest")
    parser.add_argument("--config", default="config/config.yaml", help="config file path")
    parser.add_argument("--model", default="ridge", help="model name to evaluate")
    parser.add_argument("--holding-days", type=int, default=None, help="holding period override")
    parser.add_argument("--top-n", type=int, default=None, help="number of positions override")
    args = parser.parse_args()

    setup_logging()
    try:
        return run(args.config, args.model, args.holding_days, args.top_n)
    except Exception:
        logger.exception("Backtest terminated abnormally")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
