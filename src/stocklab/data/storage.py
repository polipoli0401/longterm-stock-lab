"""SQLite persistence module.

Caches price and fundamental data, and stores daily rankings and run logs.
When operated on GitHub Actions, this DB file is committed back to the
repository so data carries over between runs (keeping API load low via
incremental fetches).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from stocklab.logger import RUN_ID, get_logger, log_event

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    date   TEXT NOT NULL,
    ticker TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (date, ticker)
);
CREATE TABLE IF NOT EXISTS fundamentals (
    ticker     TEXT NOT NULL,
    fiscal_end TEXT NOT NULL,
    revenue REAL, operating_income REAL, net_income REAL, eps REAL,
    equity REAL, total_assets REAL, operating_cf REAL, free_cf REAL,
    PRIMARY KEY (ticker, fiscal_end)
);
CREATE TABLE IF NOT EXISTS ticker_meta (
    ticker TEXT PRIMARY KEY,
    shares_outstanding REAL,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS rankings (
    run_date TEXT NOT NULL,
    rank     INTEGER NOT NULL,
    ticker   TEXT NOT NULL,
    score    REAL,
    detail   TEXT,
    PRIMARY KEY (run_date, rank)
);
CREATE TABLE IF NOT EXISTS run_log (
    run_id  TEXT,
    ts      TEXT,
    kind    TEXT,
    payload TEXT
);
"""

_FUND_COLS = [
    "revenue",
    "operating_income",
    "net_income",
    "eps",
    "equity",
    "total_assets",
    "operating_cf",
    "free_cf",
]


class Storage:
    """SQLite storage; one instance owns one connection."""

    def __init__(self, db_path: str | Path) -> None:
        """Open the DB and initialize the schema.

        Args:
            db_path: SQLite file path (parent directories are created).
        """
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._path = path

    def close(self) -> None:
        """Close the connection."""
        self._conn.close()

    # ---------------------------------------------------------- prices
    def upsert_prices(self, prices: pd.DataFrame) -> int:
        """Upsert price rows and return the number processed."""
        if prices.empty:
            return 0
        rows = [
            (
                pd.Timestamp(r.date).strftime("%Y-%m-%d"),
                r.ticker,
                _f(r.open),
                _f(r.high),
                _f(r.low),
                _f(r.close),
                _f(r.volume),
            )
            for r in prices.itertuples(index=False)
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO prices VALUES (?,?,?,?,?,?,?)", rows
        )
        self._conn.commit()
        log_event(logger, "Prices stored", n_rows=len(rows), db=str(self._path))
        return len(rows)

    def load_prices(
        self, tickers: list[str] | None = None, start: str | None = None
    ) -> pd.DataFrame:
        """Load prices (the ``date`` column is parsed to datetime)."""
        query = "SELECT date, ticker, open, high, low, close, volume FROM prices"
        conds: list[str] = []
        params: list[Any] = []
        if tickers:
            conds.append(f"ticker IN ({','.join('?' * len(tickers))})")
            params.extend(tickers)
        if start:
            conds.append("date >= ?")
            params.append(start)
        if conds:
            query += " WHERE " + " AND ".join(conds)
        query += " ORDER BY ticker, date"
        df = pd.read_sql_query(query, self._conn, params=params or None)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ---------------------------------------------------- fundamentals
    def upsert_fundamentals(self, fundamentals: pd.DataFrame) -> int:
        """Upsert fundamental rows."""
        if fundamentals.empty:
            return 0
        rows = [
            (r["ticker"], r["fiscal_end"], *[_f(r.get(c)) for c in _FUND_COLS])
            for r in fundamentals.to_dict("records")
        ]
        placeholders = ",".join("?" * (2 + len(_FUND_COLS)))
        self._conn.executemany(
            f"INSERT OR REPLACE INTO fundamentals VALUES ({placeholders})", rows
        )
        self._conn.commit()
        log_event(logger, "Fundamentals stored", n_rows=len(rows))
        return len(rows)

    def load_fundamentals(self) -> pd.DataFrame:
        """Load all fundamental rows."""
        return pd.read_sql_query(
            "SELECT * FROM fundamentals ORDER BY ticker, fiscal_end", self._conn
        )

    def upsert_meta(self, shares: dict[str, float]) -> None:
        """Store per-ticker metadata such as shares outstanding."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = [(t, float(v), now) for t, v in shares.items() if v]
        if rows:
            self._conn.executemany(
                "INSERT OR REPLACE INTO ticker_meta VALUES (?,?,?)", rows
            )
            self._conn.commit()

    def load_meta(self) -> dict[str, float]:
        """Return {ticker: shares outstanding}."""
        df = pd.read_sql_query(
            "SELECT ticker, shares_outstanding FROM ticker_meta", self._conn
        )
        return {
            r["ticker"]: float(r["shares_outstanding"])
            for r in df.to_dict("records")
            if r["shares_outstanding"]
        }

    def fundamentals_stale(self, max_age_days: int) -> bool:
        """Return True when fundamentals are missing or older than the limit."""
        row = self._conn.execute("SELECT MAX(updated_at) FROM ticker_meta").fetchone()
        latest = row[0] if row else None
        if not latest:
            return True
        try:
            updated = datetime.fromisoformat(latest)
        except ValueError:
            return True
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - updated > timedelta(days=max_age_days)

    # -------------------------------------------------------- results
    def save_ranking(self, run_date: str, records: list[dict[str, Any]]) -> None:
        """Store the daily ranking (scores with attached reasoning)."""
        rows = [
            (
                run_date,
                rank,
                rec.get("ticker"),
                _f(rec.get("score")),
                json.dumps(rec, ensure_ascii=False, default=str),
            )
            for rank, rec in enumerate(records, start=1)
        ]
        if rows:
            self._conn.executemany(
                "INSERT OR REPLACE INTO rankings VALUES (?,?,?,?,?)", rows
            )
            self._conn.commit()

    def log_run(self, kind: str, payload: dict[str, Any]) -> None:
        """Record a run summary in the DB as well (in addition to JSONL logs)."""
        self._conn.execute(
            "INSERT INTO run_log VALUES (?,?,?,?)",
            (
                RUN_ID,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                kind,
                json.dumps(payload, ensure_ascii=False, default=str),
            ),
        )
        self._conn.commit()


def _f(value: Any) -> float | None:
    """NaN-tolerant float conversion."""
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
