"""Shared data preparation for training and backtesting.

``train.py`` and ``backtest.py`` share the same preparation steps
(price fetch -> fundamentals refresh -> features/target/filters).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from stocklab.config import Config
from stocklab.currency import convert_prices_to_jpy, extract_fx_series, usd_tickers
from stocklab.data.fetcher import FundamentalFetcher, PriceFetcher
from stocklab.data.market_calendar import align_prices_to_calendar, jp_calendar
from stocklab.data.storage import Storage
from stocklab.features.builder import FeatureBuilder
from stocklab.inputs import load_universe
from stocklab.logger import get_logger, log_event

logger = get_logger(__name__)


@dataclass
class TrainingData:
    """Everything required for training and backtesting."""

    panel: pd.DataFrame          # date, ticker, close, features (standardized), target
    feature_cols: list[str]      # model feature columns
    filters: pd.DataFrame        # date, ticker, f_*, filter_pass
    bench_close: pd.Series       # benchmark close (index = date)
    prices: pd.DataFrame         # universe prices (long format)
    universe: pd.DataFrame       # ticker, name


def prepare_training_data(cfg: Config) -> TrainingData:
    """Run everything from data acquisition to feature/target generation.

    Args:
        cfg: Application configuration.

    Returns:
        :class:`TrainingData`.

    Raises:
        RuntimeError: When prices or the benchmark cannot be fetched.
    """
    storage = Storage(cfg.data.db_path)
    try:
        universe = load_universe(cfg.data.universe_file)
        tickers = universe["ticker"].tolist()
        bench_tickers = [b for b in (cfg.data.benchmark, cfg.data.benchmark_alt) if b]

        usd = usd_tickers(tickers)
        fetch_set = set(tickers) | set(bench_tickers)
        if usd:
            fetch_set.add(cfg.data.fx_ticker)
        prices = PriceFetcher().fetch(sorted(fetch_set), cfg.data.train_lookback_days)
        if usd:
            fx = extract_fx_series(prices, cfg.data.fx_ticker)
            if fx.empty:
                raise RuntimeError("Could not fetch the USDJPY rate for US tickers")
            prices = convert_prices_to_jpy(prices, fx, usd)
            prices = prices[prices["ticker"] != cfg.data.fx_ticker].reset_index(drop=True)
        prices = align_prices_to_calendar(prices, jp_calendar(prices))
        storage.upsert_prices(prices)

        if storage.fundamentals_stale(cfg.data.fundamental_refresh_days):
            log_event(logger, "Refreshing fundamentals")
            fundamentals_new, shares = FundamentalFetcher().fetch_many(tickers)
            storage.upsert_fundamentals(fundamentals_new)
            storage.upsert_meta(shares)
        fundamentals = storage.load_fundamentals()
        shares_map = storage.load_meta()
    finally:
        storage.close()

    builder = FeatureBuilder(cfg.features, cfg.data.publication_lag_days)
    uni_prices = prices[prices["ticker"].isin(tickers)].copy()
    bench_close = _bench_series(prices, cfg)

    panel = builder.build(uni_prices, fundamentals, shares_map)
    target = builder.build_target(
        uni_prices, bench_close, cfg.model.horizon_days, cfg.model.target_type
    )
    panel = panel.merge(target, on=["date", "ticker"], how="left")
    panel = builder.standardize_cross_section(
        panel, builder.model_features, cfg.features.winsorize_sigma
    )
    filters = builder.build_filters(
        uni_prices, cfg.selection.unit_shares, cfg.selection.max_unit_cost_jpy
    )

    log_event(
        logger,
        "Training data prepared",
        n_rows=len(panel),
        n_tickers=panel["ticker"].nunique(),
        n_realized_targets=int(panel["target"].notna().sum()),
    )
    return TrainingData(
        panel=panel,
        feature_cols=builder.model_features,
        filters=filters,
        bench_close=bench_close,
        prices=uni_prices,
        universe=universe,
    )


def _bench_series(prices: pd.DataFrame, cfg: Config) -> pd.Series:
    """Extract the benchmark close series (primary first, then fallback)."""
    for ticker in (cfg.data.benchmark, cfg.data.benchmark_alt):
        if not ticker:
            continue
        sub = prices[prices["ticker"] == ticker]
        if sub.empty:
            continue
        series = (
            sub.drop_duplicates(subset=["date"], keep="last")
            .set_index("date")["close"]
            .sort_index()
        )
        if ticker != cfg.data.benchmark:
            log_event(
                logger,
                "Primary benchmark unavailable; using the fallback",
                level=logging.WARNING,
                ticker=ticker,
            )
        return series
    raise RuntimeError("Could not fetch benchmark prices")
