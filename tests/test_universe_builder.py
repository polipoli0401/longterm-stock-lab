"""Tests for automatic universe construction (offline parts)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stocklab.data.universe_builder import (
    filter_segments,
    merge_with_extra,
    screen_by_liquidity,
)

PRIME = "\u30d7\u30e9\u30a4\u30e0\uff08\u5185\u56fd\u682a\u5f0f\uff09"
STANDARD = "\u30b9\u30bf\u30f3\u30c0\u30fc\u30c9\uff08\u5185\u56fd\u682a\u5f0f\uff09"
GROWTH = "\u30b0\u30ed\u30fc\u30b9\uff08\u5185\u56fd\u682a\u5f0f\uff09"
PRIME_FOREIGN = "\u30d7\u30e9\u30a4\u30e0\uff08\u5916\u56fd\u682a\u5f0f\uff09"
ETF = "ETF\u30fbETN"


def test_filter_segments_domestic_only():
    jpx = pd.DataFrame(
        {
            "code": ["7203", "1301", "4488", "9999", "1305"],
            "name": ["Toyota", "Kyokuyo", "AI inside", "Foreign Co", "ETF Fund"],
            "segment": [PRIME, STANDARD, GROWTH, PRIME_FOREIGN, ETF],
        }
    )
    out = filter_segments(jpx, ["prime", "standard"])
    assert out["ticker"].tolist() == ["7203.T", "1301.T"]
    out_all = filter_segments(jpx, ["prime", "standard", "growth"])
    assert "4488.T" in out_all["ticker"].tolist()
    assert "9999.T" not in out_all["ticker"].tolist()  # foreign excluded
    assert "1305.T" not in out_all["ticker"].tolist()  # ETF excluded


def _screen_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=40)
    rows = []
    for ticker, price, volume in (
        ("BIG.T", 1000.0, 1_000_000),   # turnover 1.0e9
        ("MID.T", 500.0, 400_000),      # turnover 2.0e8
        ("TINY.T", 100.0, 100_000),     # turnover 1.0e7 (below floor)
    ):
        for d in dates:
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": float(volume),
                }
            )
    return pd.DataFrame(rows)


def test_screen_by_liquidity_floor_and_cap():
    prices = _screen_prices()
    out = screen_by_liquidity(prices, min_turnover_jpy=1.0e8, max_size=0)
    assert out["ticker"].tolist() == ["BIG.T", "MID.T"]  # sorted desc, floor applied
    capped = screen_by_liquidity(prices, min_turnover_jpy=1.0e8, max_size=1)
    assert capped["ticker"].tolist() == ["BIG.T"]


def test_screen_requires_enough_history():
    prices = _screen_prices()
    short = prices[prices["date"] >= prices["date"].max() - pd.Timedelta(days=10)]
    out = screen_by_liquidity(short, min_turnover_jpy=0.0, max_size=0)
    assert out.empty  # fewer than 20 observations -> excluded


def test_merge_with_extra_dedup_and_names():
    screened = pd.DataFrame({"ticker": ["7203.T", "9999.T"], "turnover": [2.0, 1.0]})
    names = pd.DataFrame({"ticker": ["7203.T"], "name": ["Toyota"]})
    extra = pd.DataFrame(
        {"ticker": ["AAPL", "7203.T"], "name": ["Apple", "Toyota duplicate"]}
    )
    out = merge_with_extra(screened, names, extra)
    assert out["ticker"].tolist() == ["7203.T", "9999.T", "AAPL"]
    assert out.loc[out["ticker"] == "7203.T", "name"].iloc[0] == "Toyota"
    assert out.loc[out["ticker"] == "9999.T", "name"].iloc[0] == "9999.T"  # fallback


def test_screen_handles_nan_only_ticker():
    prices = _screen_prices()
    dates = sorted(prices["date"].unique())
    nan_rows = pd.DataFrame(
        {
            "date": dates,
            "ticker": "NAN.T",
            "open": np.nan,
            "high": np.nan,
            "low": np.nan,
            "close": np.nan,
            "volume": np.nan,
        }
    )
    out = screen_by_liquidity(
        pd.concat([prices, nan_rows], ignore_index=True), 1.0e8, 0
    )
    assert "NAN.T" not in out["ticker"].tolist()
