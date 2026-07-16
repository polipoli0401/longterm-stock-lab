"""Daily analysis pipeline entry point.

Runs data acquisition -> feature engineering -> scoring -> ranking ->
holdings risk analysis -> report generation -> notifications
(the daily flow of spec section 3).

Usage:
    python main.py [--config config/config.yaml] [--no-notify]
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from stocklab.config import Config
from stocklab.data.fetcher import FundamentalFetcher, PriceFetcher
from stocklab.data.storage import Storage
from stocklab.features.builder import FEATURE_LABELS, FeatureBuilder
from stocklab.inputs import load_holdings, load_universe
from stocklab.logger import get_logger, log_event, setup_logging
from stocklab.models.registry import ModelRegistry
from stocklab.notify.notifier import notify_all
from stocklab.report.generator import ReportGenerator
from stocklab.risk.analyzer import RiskAnalyzer
from stocklab.scoring.scorer import Scorer

logger = get_logger("stocklab.main")


def run(config_path: str, no_notify: bool = False) -> int:
    """Run the daily pipeline.

    Args:
        config_path: Path to the config file.
        no_notify: Skip notifications when True (manual runs / testing).

    Returns:
        Exit code (0 = success).
    """
    cfg = Config.load(config_path)
    storage = Storage(cfg.data.db_path)

    # --- inputs --------------------------------------------------------
    universe = load_universe(cfg.data.universe_file)
    tickers = universe["ticker"].tolist()
    name_map = dict(zip(universe["ticker"], universe["name"]))
    holdings = load_holdings()
    holding_tickers = [h["ticker"] for h in holdings]

    # --- data acquisition ------------------------------------------------
    prices = PriceFetcher().fetch(
        sorted(set(tickers) | set(holding_tickers)), cfg.data.price_lookback_days
    )
    storage.upsert_prices(prices)

    if storage.fundamentals_stale(cfg.data.fundamental_refresh_days):
        log_event(logger, "Refreshing fundamentals")
        fundamentals_new, shares = FundamentalFetcher().fetch_many(tickers)
        storage.upsert_fundamentals(fundamentals_new)
        storage.upsert_meta(shares)
    fundamentals = storage.load_fundamentals()
    shares_map = storage.load_meta()

    # --- features --------------------------------------------------------
    builder = FeatureBuilder(cfg.features, cfg.data.publication_lag_days)
    uni_prices = prices[prices["ticker"].isin(tickers)].copy()
    panel = builder.build(uni_prices, fundamentals, shares_map)
    panel = builder.standardize_cross_section(
        panel, builder.model_features, cfg.features.winsorize_sigma
    )
    filters = builder.build_filters(uni_prices)

    last_date = panel["date"].max()
    latest = panel[panel["date"] == last_date].merge(
        filters[filters["date"] == last_date], on=["date", "ticker"], how="left"
    )
    latest["name"] = latest["ticker"].map(name_map)
    latest["name"] = latest["name"].where(latest["name"].notna(), latest["ticker"])

    # --- scoring ----------------------------------------------------------
    registry = ModelRegistry(cfg.model.model_dir)
    model, meta = registry.load_champion()
    feature_cols = builder.model_features
    if model is not None and meta and meta.get("feature_cols"):
        feature_cols = list(meta["feature_cols"])  # infer with the training-time columns
        missing = [c for c in feature_cols if c not in latest.columns]
        if missing:
            log_event(
                logger,
                "Champion features mismatch the current config; switching to fallback",
                level=logging.WARNING,
                missing=missing,
            )
            model, meta = None, None
            feature_cols = builder.model_features
    if model is None:
        log_event(
            logger,
            "No champion model registered; using the equal-weight fallback score",
            level=logging.WARNING,
        )

    scorer = Scorer(feature_cols, pipeline=model, labels=FEATURE_LABELS)
    scored = scorer.score_latest(latest)

    # --- ranking (filter-passing tickers first) ----------------------------
    top_n = cfg.backtest.top_n
    ranked = [s for s in scored if s.filter_pass][:top_n]
    chosen = {s.ticker for s in ranked}
    if len(ranked) < top_n:
        for s in scored:
            if s.ticker in chosen:
                continue
            s.concerns.append("Failed technical filters (listed for reference)")
            ranked.append(s)
            chosen.add(s.ticker)
            if len(ranked) >= top_n:
                break
        ranked.sort(key=lambda s: s.score, reverse=True)

    # --- holdings risk -------------------------------------------------------
    assessments = RiskAnalyzer(cfg.risk).analyze(holdings, prices)
    alerts = [a for a in assessments if a.alert]

    # --- reports & persistence ------------------------------------------------
    run_date = str(pd.Timestamp(last_date).date())
    reporter = ReportGenerator(cfg.backtest.report_dir)
    markdown = reporter.daily_markdown(
        run_date, ranked, assessments, meta, len(universe), len(scored)
    )
    report_path = reporter.save(markdown, f"daily/{run_date}.md")
    reporter.save(markdown, "latest.md")

    storage.save_ranking(run_date, [s.to_dict() for s in ranked])
    storage.log_run(
        "daily",
        {
            "run_date": run_date,
            "model": (meta or {}).get("model_name", "fallback"),
            "top": [s.ticker for s in ranked],
            "alerts": [a.ticker for a in alerts],
        },
    )
    storage.close()

    # --- notifications ----------------------------------------------------------
    model_name = (meta or {}).get("model_name", "fallback (equal weight)")
    summary = reporter.daily_summary(run_date, ranked, alerts, model_name)
    if no_notify:
        log_event(logger, "Notifications skipped (--no-notify)")
    else:
        notify_all(summary)

    log_event(logger, "Daily pipeline complete", report=str(report_path))
    return 0


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="stocklab daily analysis")
    parser.add_argument("--config", default="config/config.yaml", help="config file path")
    parser.add_argument("--no-notify", action="store_true", help="do not send notifications")
    args = parser.parse_args()

    setup_logging()
    try:
        return run(args.config, args.no_notify)
    except Exception:
        logger.exception("Daily pipeline terminated abnormally")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
