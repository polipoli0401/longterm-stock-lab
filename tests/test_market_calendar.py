"""Tests for JP/US calendar alignment (regression for the NaN-hole bug)."""

from __future__ import annotations

import pandas as pd

from stocklab.config import FeatureConfig
from stocklab.data.market_calendar import align_prices_to_calendar, jp_calendar
from stocklab.features.builder import FeatureBuilder


def _mixed_prices() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Simulate mixed calendars: US rows exist on JP holidays and one US
    holiday leaves a hole inside the JP calendar."""
    all_days = pd.bdate_range("2023-01-02", periods=320)
    jp_days = all_days.delete([50, 120, 310])  # JP holidays incl. one recent
    us_days = all_days.delete([80])            # one US holiday
    rows = []
    for i, d in enumerate(jp_days):
        rows.append(_row(d, "AAA.T", 1000.0 + i, 1_000_000))  # gently rising
    for _, d in enumerate(us_days):
        rows.append(_row(d, "USD1", 3000.0, 50_000))  # already JPY-converted
    return pd.DataFrame(rows), jp_days


def _row(d, ticker, close, volume):
    return {
        "date": d,
        "ticker": ticker,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "close_native": close,
        "volume": float(volume),
    }


def test_jp_calendar_extraction():
    prices, jp_days = _mixed_prices()
    cal = jp_calendar(prices)
    assert list(cal) == list(jp_days)  # US-only sessions excluded


def test_alignment_fills_foreign_holidays():
    prices, jp_days = _mixed_prices()
    aligned = align_prices_to_calendar(prices, jp_calendar(prices))
    usd_rows = prices[prices["ticker"] == "USD1"]["date"].values
    missing = [d for d in jp_days if d not in usd_rows]
    assert len(missing) == 1  # the US holiday inside the JP calendar
    usd = aligned[aligned["ticker"] == "USD1"].set_index("date")
    assert float(usd.loc[missing[0], "close"]) == 3000.0  # ffilled
    assert float(usd.loc[missing[0], "volume"]) == 0.0
    assert not aligned[["close", "volume"]].isna().any().any()  # no holes left


def test_filters_survive_mixed_calendar():
    """Regression: a JP mega-cap must pass liquidity/trend after alignment."""
    prices, _ = _mixed_prices()
    builder = FeatureBuilder(FeatureConfig(), publication_lag_days=90)

    aligned = align_prices_to_calendar(prices, jp_calendar(prices))
    f = builder.build_filters(aligned)
    last = f[f["date"] == f["date"].max()].set_index("ticker")
    assert bool(last.loc["AAA.T", "f_liquid"]) is True
    assert bool(last.loc["AAA.T", "f_trend"]) is True

    # Pre-fix failure mode: rolling windows poisoned by NaN holes.
    unaligned = builder.build_filters(prices)
    bad = unaligned[unaligned["date"] == unaligned["date"].max()].set_index("ticker")
    assert bool(bad.loc["AAA.T", "f_liquid"]) is False
