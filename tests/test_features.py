"""Tests for feature engineering (targets, filters, standardization)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from stocklab.config import FeatureConfig
from stocklab.features.builder import FeatureBuilder


def _prices(days: int = 300, tickers: tuple[str, ...] = ("AAA", "BBB")) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=days)
    rows = []
    rng = np.random.default_rng(0)
    for t_index, ticker in enumerate(tickers):
        base = 100.0 * (1 + t_index)
        close = base * np.cumprod(1 + rng.normal(0.0005, 0.01, size=days))
        for date, c in zip(dates, close):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": c,
                    "high": c,
                    "low": c,
                    "close": c,
                    "volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows)


def _builder() -> FeatureBuilder:
    return FeatureBuilder(FeatureConfig(), publication_lag_days=90)


def test_absolute_target_known_value():
    """A constant +1%/day series must yield target = 1.01**h - 1."""
    days, horizon = 40, 5
    dates = pd.bdate_range("2023-01-02", periods=days)
    close = 100 * np.cumprod(np.full(days, 1.01))
    prices = pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAA",
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1e6,
        }
    )
    target = _builder().build_target(prices, None, horizon, target_type="absolute")
    first = target.sort_values("date")["target"].iloc[0]
    assert math.isclose(first, 1.01**horizon - 1, rel_tol=1e-9)
    # The last `horizon` rows have no realized target.
    assert target.sort_values("date")["target"].tail(horizon).isna().all()


def test_excess_target_zero_when_equal_to_benchmark():
    """When the stock moves exactly like the benchmark, excess must be ~0."""
    prices = _prices(days=60, tickers=("AAA",))
    bench = prices.set_index("date")["close"]
    target = _builder().build_target(prices, bench, 10, target_type="excess")
    realized = target["target"].dropna()
    assert np.allclose(realized.to_numpy(), 0.0, atol=1e-12)


def test_cross_sectional_standardization_clips():
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-05")] * 5,
            "ticker": list("ABCDE"),
            "x": [0.0, 0.1, 0.2, 0.1, 50.0],  # extreme outlier included
        }
    )
    out = FeatureBuilder.standardize_cross_section(df, ["x"], clip_sigma=3.0)
    assert out["x"].abs().max() <= 3.0 + 1e-9
    assert abs(out["x"].mean()) < 1.5  # roughly centered


def test_filters_columns_and_types():
    prices = _prices(days=260)
    filters = _builder().build_filters(prices)
    for col in ("f_trend", "f_heat", "f_liquid", "filter_pass"):
        assert col in filters.columns
    assert filters["filter_pass"].dtype == bool
    # filter_pass must equal the AND of the individual flags.
    combined = filters[["f_trend", "f_heat", "f_liquid"]].all(axis=1)
    assert (filters["filter_pass"] == combined).all()


def test_build_without_fundamentals_keeps_columns():
    """Without fundamentals, columns still exist (all NaN)."""
    prices = _prices(days=200)
    builder = _builder()
    panel = builder.build(prices, fundamentals=None, shares=None)
    for col in builder.model_features:
        assert col in panel.columns
    assert panel["roe"].isna().all()

def test_momentum_matches_manual_calculation():
    prices = _prices(days=260, tickers=("AAA",))
    builder = _builder()
    feats = builder.build_price_features(prices)
    px = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    dates = px.index
    expected = float(px["AAA"].iloc[-1] / px["AAA"].iloc[-22] - 1.0)
    got = feats[(feats["date"] == dates[-1]) & (feats["ticker"] == "AAA")]["mom_21"].iloc[0]
    assert math.isclose(float(got), expected, rel_tol=1e-10)

def test_build_merges_mixed_datetime_units():
    """Regression: yfinance dates in [s] + fundamentals in [us] must merge.

    pandas>=3 infers datetime units from source data; merge_asof rejects
    mismatched units, which broke the first production run.
    """
    prices = _prices(days=250, tickers=("AAA",))
    prices["date"] = prices["date"].astype("datetime64[s]")  # coarse unit
    fiscal = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "fiscal_end": ["2022-01-31"],  # +90d lag -> usable within range
            "revenue": [1e9],
            "operating_income": [1e8],
            "net_income": [8e7],
            "eps": [10.0],
            "equity": [5e8],
            "total_assets": [1e9],
            "operating_cf": [1.2e8],
            "free_cf": [9e7],
        }
    )
    fiscal["fiscal_end"] = fiscal["fiscal_end"].astype("datetime64[us]")
    builder = _builder()
    panel = builder.build(prices, fiscal, shares={"AAA": 1e6})
    assert panel["roe"].notna().any()
    assert panel["earnings_yield"].notna().any()
