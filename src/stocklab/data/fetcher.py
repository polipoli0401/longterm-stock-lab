"""Market and fundamental data acquisition (yfinance).

yfinance, being free, is the default data source.

Known limitations:
    yfinance fundamentals cover only the last 4-5 annual periods and carry
    no true disclosure (point-in-time) dates. This system prevents leakage
    with a conservative approximation: each fiscal period becomes usable
    ``publication_lag_days`` after its fiscal end. For production-grade
    use, switching to point-in-time data such as the J-Quants API is
    recommended (see "Limitations" in the README).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from stocklab.logger import get_logger, log_event

logger = get_logger(__name__)

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]

# yfinance statement item name -> internal column name
STATEMENT_ITEMS: dict[str, dict[str, str]] = {
    "income_stmt": {
        "Total Revenue": "revenue",
        "Operating Income": "operating_income",
        "Net Income": "net_income",
        "Basic EPS": "eps",
    },
    "balance_sheet": {
        "Stockholders Equity": "equity",
        "Total Assets": "total_assets",
    },
    "cashflow": {
        "Operating Cash Flow": "operating_cf",
        "Free Cash Flow": "free_cf",
    },
}
FUNDAMENTAL_COLUMNS = [
    "revenue",
    "operating_income",
    "net_income",
    "eps",
    "equity",
    "total_assets",
    "operating_cf",
    "free_cf",
]


class PriceFetcher:
    """Fetches daily (adjusted) prices."""

    def __init__(self, retries: int = 3, pause_sec: float = 3.0) -> None:
        """Args:
        retries: Number of retries on download failure.
        pause_sec: Delay between retries (seconds).
        """
        self.retries = retries
        self.pause_sec = pause_sec

    def fetch(self, tickers: list[str], lookback_days: int) -> pd.DataFrame:
        """Fetch daily OHLCV for multiple tickers in long format.

        Args:
            tickers: Tickers to fetch (e.g. ``7203.T``).
            lookback_days: History length to fetch (calendar days).

        Returns:
            DataFrame with columns ``date, ticker, open, high, low, close,
            volume``.
        """
        tickers = sorted(set(tickers))
        start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        raw: pd.DataFrame | None = None
        for attempt in range(1, self.retries + 1):
            try:
                raw = yf.download(
                    tickers=tickers,
                    start=start,
                    auto_adjust=True,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )
            except Exception:
                logger.exception("Price download raised (attempt=%d)", attempt)
                raw = None
            if raw is not None and not raw.empty:
                break
            time.sleep(self.pause_sec)
        if raw is None or raw.empty:
            raise RuntimeError("Failed to download price data")

        df = self._to_long(raw, tickers)
        fetched = set(df["ticker"].unique())
        missing = sorted(set(tickers) - fetched)
        if missing:
            log_event(logger, "Some tickers returned no price data", missing=missing)
        log_event(
            logger,
            "Price download complete",
            n_tickers=len(fetched),
            n_rows=len(df),
            date_from=str(df["date"].min().date()),
            date_to=str(df["date"].max().date()),
        )
        return df

    @staticmethod
    def _to_long(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
        """Normalize the wide yfinance output to long format."""
        frames: list[pd.DataFrame] = []
        if isinstance(raw.columns, pd.MultiIndex):
            available = set(raw.columns.get_level_values(0))
            for ticker in tickers:
                if ticker not in available:
                    continue
                sub = raw[ticker].copy()
                sub.columns = [str(c).lower() for c in sub.columns]
                sub = sub.reindex(columns=PRICE_COLUMNS)
                sub["ticker"] = ticker
                frames.append(sub.reset_index())
        else:
            sub = raw.copy()
            sub.columns = [str(c).lower() for c in sub.columns]
            sub = sub.reindex(columns=PRICE_COLUMNS)
            sub["ticker"] = tickers[0]
            frames.append(sub.reset_index())

        df = pd.concat(frames, ignore_index=True)
        date_col = "Date" if "Date" in df.columns else "index"
        df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"])
        if getattr(df["date"].dt, "tz", None) is not None:
            df["date"] = df["date"].dt.tz_localize(None)
        df = df.dropna(subset=["close"]).sort_values(["ticker", "date"])
        return df[["date", "ticker", *PRICE_COLUMNS]].reset_index(drop=True)


class FundamentalFetcher:
    """Fetches annual financial statements (revenue, income, CF, equity...)."""

    def __init__(self, pause_sec: float = 0.7) -> None:
        """Args:
        pause_sec: Delay between per-ticker requests (rate-limit friendly).
        """
        self.pause_sec = pause_sec

    def fetch_many(
        self, tickers: list[str]
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        """Fetch fundamentals and shares outstanding for multiple tickers.

        Returns:
            Tuple of (fundamentals DataFrame, {ticker: shares outstanding}).
            The DataFrame columns are ``ticker, fiscal_end`` + statement items.
        """
        frames: list[pd.DataFrame] = []
        shares: dict[str, float] = {}
        failed: list[str] = []
        for ticker in tickers:
            try:
                df, n_shares = self.fetch_one(ticker)
                if not df.empty:
                    frames.append(df)
                if n_shares:
                    shares[ticker] = n_shares
            except Exception:
                logger.exception("Fundamental fetch failed: %s", ticker)
                failed.append(ticker)
            time.sleep(self.pause_sec)
        result = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["ticker", "fiscal_end", *FUNDAMENTAL_COLUMNS])
        )
        log_event(
            logger,
            "Fundamental download complete",
            n_tickers_ok=len({t for t in result.get("ticker", [])}),
            n_records=len(result),
            failed=failed,
        )
        return result, shares

    def fetch_one(self, ticker: str) -> tuple[pd.DataFrame, float | None]:
        """Fetch annual fundamentals and shares outstanding for one ticker."""
        tk = yf.Ticker(ticker)
        records: dict[str, dict[str, float]] = {}
        for attr, mapping in STATEMENT_ITEMS.items():
            stmt = getattr(tk, attr, None)
            if stmt is None or getattr(stmt, "empty", True):
                continue
            for src, dst in mapping.items():
                if src not in stmt.index:
                    continue
                for col, val in stmt.loc[src].items():
                    key = pd.Timestamp(col).strftime("%Y-%m-%d")
                    if pd.notna(val):
                        records.setdefault(key, {})[dst] = float(val)

        rows = [
            {"ticker": ticker, "fiscal_end": fiscal_end, **values}
            for fiscal_end, values in sorted(records.items())
        ]
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.reindex(columns=["ticker", "fiscal_end", *FUNDAMENTAL_COLUMNS])
        return df, self._fetch_shares(tk)

    @staticmethod
    def _fetch_shares(tk: yf.Ticker) -> float | None:
        """Fetch shares outstanding (tries fast_info first, then info)."""
        try:
            value = tk.fast_info["shares"]
            if value:
                return float(value)
        except Exception:  # noqa: BLE001 - missing data is tolerated
            pass
        try:
            value = tk.info.get("sharesOutstanding")
            if value:
                return float(value)
        except Exception:  # noqa: BLE001
            pass
        return None
