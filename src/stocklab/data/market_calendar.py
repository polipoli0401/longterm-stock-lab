"""Trading-calendar alignment for a mixed JP/US universe.

JP and US markets observe different holidays. If price frames keep the
union of both calendars, every JP holiday punches a NaN hole into JP
tickers' series (and vice versa), which poisons rolling windows
(moving averages, 20-day turnover, volatility) and position-based
momentum shifts.

Fix: snap every ticker onto one canonical calendar - the JP trading days
present in the data - forward-filling prices over short foreign holidays
(volume filled with 0 so turnover stays conservative).
"""

from __future__ import annotations

import pandas as pd

from stocklab.currency import is_jpy
from stocklab.logger import get_logger, log_event

logger = get_logger(__name__)

PRICE_COLS = ["open", "high", "low", "close"]
FFILL_LIMIT = 5  # max consecutive sessions to carry a stale price (holidays only)


def jp_calendar(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """Extract the JP trading calendar from a fetched price frame.

    Falls back to the union of all dates when no JPY ticker is present
    (e.g. a pure-US configuration).
    """
    mask = prices["ticker"].map(is_jpy)
    dates = prices.loc[mask, "date"] if bool(mask.any()) else prices["date"]
    return pd.DatetimeIndex(sorted(dates.unique())).astype("datetime64[ns]")


def align_prices_to_calendar(
    prices: pd.DataFrame, calendar: pd.DatetimeIndex
) -> pd.DataFrame:
    """Reindex every ticker onto the canonical calendar.

    - Dates outside the calendar (e.g. US-only sessions) are dropped.
    - Missing sessions inside the calendar (e.g. US holidays) get the
      previous close carried forward (up to :data:`FFILL_LIMIT` days)
      with volume 0, so rolling windows never see NaN holes.
    - Leading days before a ticker's first observation stay absent.
    """
    cal = pd.DatetimeIndex(sorted(set(calendar))).astype("datetime64[ns]")
    cols = [c for c in [*PRICE_COLS, "close_native"] if c in prices.columns]
    frames: list[pd.DataFrame] = []
    for ticker, g in prices.groupby("ticker", sort=False):
        g = (
            g.drop_duplicates(subset="date", keep="last")
            .set_index("date")
            .sort_index()
            .reindex(cal)
        )
        g[cols] = g[cols].ffill(limit=FFILL_LIMIT)
        g["volume"] = g["volume"].fillna(0.0)
        g["ticker"] = ticker
        g = g.dropna(subset=["close"])
        frames.append(g.reset_index().rename(columns={"index": "date"}))
    out = pd.concat(frames, ignore_index=True)
    out = out[["date", "ticker", *cols, "volume"]]
    log_event(
        logger,
        "Prices aligned to the JP calendar",
        n_sessions=len(cal),
        n_rows=len(out),
    )
    return out
