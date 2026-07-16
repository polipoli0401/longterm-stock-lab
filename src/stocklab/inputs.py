"""Input data loading (universe and holdings).

Holdings are personal financial information, so they can stay secret even
in a public repository. They are loaded in the following priority order:

1. Environment variable ``HOLDINGS_JSON`` (GitHub Secrets recommended)
2. ``config/holdings.csv`` (a local file already in ``.gitignore``)
3. Otherwise an empty list (risk analysis is skipped)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from stocklab.config import get_secret
from stocklab.logger import get_logger, log_event

logger = get_logger(__name__)

HOLDINGS_ENV = "HOLDINGS_JSON"
HOLDINGS_FILE = Path("config/holdings.csv")


def load_universe(path: str | Path) -> pd.DataFrame:
    """Load the analysis universe.

    Args:
        path: Path to a CSV with ``ticker,name`` columns.

    Returns:
        DataFrame with ``ticker`` / ``name`` columns (duplicates removed).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Universe file not found: {p}")
    df = pd.read_csv(p, dtype=str).dropna(subset=["ticker"])
    if not {"ticker", "name"}.issubset(df.columns):
        raise ValueError("universe.csv must contain ticker,name columns")
    df["ticker"] = df["ticker"].str.strip()
    df = df.drop_duplicates(subset="ticker").reset_index(drop=True)
    log_event(logger, "Universe loaded", n_tickers=len(df), path=str(p))
    return df


def load_holdings() -> list[dict[str, Any]]:
    """Load holdings (HOLDINGS_JSON -> holdings.csv -> empty).

    Returns:
        List of ``{"ticker": str, "name": str, "quantity": float|None,
        "avg_price": float|None}`` dicts.
    """
    raw = get_secret(HOLDINGS_ENV)
    if raw:
        try:
            items = json.loads(raw)
            holdings = [_normalize(h) for h in items if h.get("ticker")]
            log_event(logger, "Holdings loaded from secret", n_holdings=len(holdings))
            return holdings
        except (json.JSONDecodeError, AttributeError, TypeError):
            logger.exception("Failed to parse HOLDINGS_JSON (value is not logged)")
            return []

    if HOLDINGS_FILE.exists():
        try:
            df = pd.read_csv(HOLDINGS_FILE)
            holdings = [_normalize(row) for row in df.to_dict("records") if row.get("ticker")]
            log_event(logger, "Holdings loaded from local CSV", n_holdings=len(holdings))
            return holdings
        except (OSError, ValueError):
            logger.exception("Failed to read holdings.csv")
            return []

    log_event(logger, "No holdings configured; skipping risk analysis")
    return []


def _normalize(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single holdings entry."""
    return {
        "ticker": str(item["ticker"]).strip(),
        "name": str(item.get("name") or item["ticker"]).strip(),
        "quantity": _to_float(item.get("quantity")),
        "avg_price": _to_float(item.get("avg_price")),
    }


def _to_float(value: Any) -> float | None:
    """Convert to float only when possible."""
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
