"""Currency handling for a mixed JP/US universe.

Design (Option A):
    Everything is valued in JPY. US-listed prices are converted daily with
    the USDJPY rate, so momentum, targets, backtests, the budget filter,
    and reports all reflect what a JPY-based (NISA) investor experiences,
    including currency moves. The benchmark stays TOPIX for every ticker.

    Financial *ratios* (ROE, margins, growth) are currency-agnostic. Value
    metrics (earnings yield etc.) are computed in the native currency via
    the preserved ``close_native`` column, so no FX distortion enters them.
"""

from __future__ import annotations

import pandas as pd

FX_DEFAULT_TICKER = "JPY=X"  # USDJPY on Yahoo Finance


def is_jpy(ticker: str) -> bool:
    """Return True for JPY-denominated tickers.

    Rule of thumb: Tokyo listings end with ``.T``; index symbols we use
    (``^N225``) are also JPY. Everything else is treated as USD.
    """
    return ticker.endswith(".T") or ticker.startswith("^")


def usd_tickers(tickers: list[str]) -> list[str]:
    """Extract the USD-denominated subset of a ticker list."""
    return sorted({t for t in tickers if not is_jpy(t)})


def unit_for(ticker: str, jp_unit_shares: int) -> int:
    """Shares per trading unit: JP lots (default 100), US single shares."""
    return jp_unit_shares if is_jpy(ticker) else 1


def convert_prices_to_jpy(
    prices: pd.DataFrame, fx_close: pd.Series, usd: set[str] | list[str]
) -> pd.DataFrame:
    """Convert USD rows of a long-format price frame to JPY.

    Adds a ``close_native`` column (pre-conversion close for USD rows,
    identical to ``close`` for JPY rows) used for native-currency value
    metrics. Volume stays a share count.

    Args:
        prices: Long-format prices (date, ticker, open..close, volume).
        fx_close: USDJPY close series indexed by date.
        usd: Tickers to convert.

    Returns:
        A converted copy of ``prices``.
    """
    out = prices.copy()
    out["close_native"] = out["close"]
    usd_set = set(usd)
    if not usd_set or fx_close is None or fx_close.empty:
        return out

    rate_by_date = fx_close.sort_index().ffill().bfill()
    mask = out["ticker"].isin(usd_set)
    rates = out.loc[mask, "date"].map(rate_by_date)
    for col in ("open", "high", "low", "close"):
        out.loc[mask, col] = out.loc[mask, col] * rates
    return out


def extract_fx_series(prices: pd.DataFrame, fx_ticker: str) -> pd.Series:
    """Pull the FX close series out of a fetched price frame."""
    sub = prices[prices["ticker"] == fx_ticker]
    if sub.empty:
        return pd.Series(dtype=float)
    return (
        sub.drop_duplicates(subset=["date"], keep="last")
        .set_index("date")["close"]
        .sort_index()
    )
