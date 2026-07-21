"""Automatic universe construction (JPX listing + liquidity screen).

Weekly flow:
    1. Download the official JPX listed-companies file (monthly Excel).
    2. Keep the configured market segments (Prime/Standard/Growth,
       domestic common stocks only).
    3. Fetch recent prices and keep names whose 20-day average turnover
       clears a tradability floor; optionally cap the universe size by
       turnover rank.
    4. Merge the always-include extra file (US stocks, manual picks) and
       write ``config/universe.csv``.

The JPX file uses Japanese segment labels; the constants below match on
those labels (data values, not code).
"""

from __future__ import annotations

import io
import time

import pandas as pd

from stocklab.config import UniverseConfig
from stocklab.logger import get_logger, log_event

logger = get_logger(__name__)

# Japanese segment prefixes in the JPX file (data literals, not UI text).
SEGMENT_PREFIXES: dict[str, str] = {
    "prime": "\u30d7\u30e9\u30a4\u30e0",        # プライム
    "standard": "\u30b9\u30bf\u30f3\u30c0\u30fc\u30c9",  # スタンダード
    "growth": "\u30b0\u30ed\u30fc\u30b9",       # グロース
}
DOMESTIC_MARKER = "\u5185\u56fd\u682a\u5f0f"     # 内国株式 (domestic common stock)

JPX_COLUMNS = {"code": "\u30b3\u30fc\u30c9", "name": "\u9298\u67c4\u540d",
               "segment": "\u5e02\u5834\u30fb\u5546\u54c1\u533a\u5206"}


def fetch_jpx_list(url: str) -> pd.DataFrame:
    """Download and parse the JPX listing into (code, name, segment)."""
    import requests  # lazy import (not required in test envs)

    response = requests.get(
        url, headers={"User-Agent": "Mozilla/5.0 (universe-builder)"}, timeout=60
    )
    response.raise_for_status()
    raw = pd.read_excel(io.BytesIO(response.content), dtype=str)
    missing = [c for c in JPX_COLUMNS.values() if c not in raw.columns]
    if missing:
        raise RuntimeError(f"JPX file format changed; missing columns: {missing}")
    df = raw.rename(columns={v: k for k, v in JPX_COLUMNS.items()})
    df = df[["code", "name", "segment"]].dropna(subset=["code"])
    df["code"] = df["code"].str.strip()
    df["name"] = df["name"].str.strip().str.replace(",", " ")
    log_event(logger, "JPX listing downloaded", n_rows=len(df))
    return df


def filter_segments(jpx: pd.DataFrame, segments: list[str]) -> pd.DataFrame:
    """Keep domestic common stocks in the requested segments."""
    prefixes = tuple(
        SEGMENT_PREFIXES[s] for s in segments if s in SEGMENT_PREFIXES
    )
    if not prefixes:
        raise ValueError(f"No valid segments in {segments}")
    seg = jpx["segment"].fillna("")
    mask = seg.str.startswith(prefixes) & seg.str.contains(DOMESTIC_MARKER)
    out = jpx[mask].copy()
    out["ticker"] = out["code"] + ".T"
    log_event(logger, "Segments filtered", segments=list(segments), n_candidates=len(out))
    return out[["ticker", "name"]]


def fetch_screen_prices(
    tickers: list[str], lookback_days: int, batch_size: int = 250, pause_sec: float = 2.0
) -> pd.DataFrame:
    """Fetch recent prices for the screen in batches (failures tolerated)."""
    from stocklab.data.fetcher import PriceFetcher  # lazy: pulls in yfinance

    frames: list[pd.DataFrame] = []
    fetcher = PriceFetcher(retries=2, pause_sec=2.0)
    for start in range(0, len(tickers), batch_size):
        batch = tickers[start : start + batch_size]
        try:
            frames.append(fetcher.fetch(batch, lookback_days))
        except RuntimeError:
            log_event(logger, "Screen batch failed; skipping", batch_start=start)
        time.sleep(pause_sec)
    if not frames:
        raise RuntimeError("Liquidity screen could not fetch any prices")
    return pd.concat(frames, ignore_index=True)


def screen_by_liquidity(
    prices: pd.DataFrame, min_turnover_jpy: float, max_size: int
) -> pd.DataFrame:
    """Rank by 20-day average turnover and apply floor/size cap.

    Returns:
        DataFrame with ``ticker, turnover`` sorted by turnover descending.
    """
    dedup = prices.drop_duplicates(subset=["date", "ticker"], keep="last")
    px = dedup.pivot(index="date", columns="ticker", values="close").sort_index()
    vol = dedup.pivot(index="date", columns="ticker", values="volume").sort_index()
    turnover = (px * vol).rolling(20).mean().iloc[-1]
    enough = px.notna().sum() >= 20  # require at least 20 observations
    turnover = turnover[enough & turnover.notna()]
    passed = turnover[turnover >= min_turnover_jpy].sort_values(ascending=False)
    if max_size and max_size > 0:
        passed = passed.head(max_size)
    out = passed.rename("turnover").reset_index().rename(columns={"index": "ticker"})
    log_event(
        logger,
        "Liquidity screen complete",
        n_passed=len(out),
        floor_jpy=min_turnover_jpy,
        max_size=max_size,
    )
    return out


def merge_with_extra(
    screened: pd.DataFrame, names: pd.DataFrame, extra: pd.DataFrame | None
) -> pd.DataFrame:
    """Attach names and append the always-include extra list (deduped)."""
    out = screened[["ticker"]].merge(names, on="ticker", how="left")
    out["name"] = out["name"].fillna(out["ticker"])
    if extra is not None and not extra.empty:
        extra = extra[["ticker", "name"]].copy()
        extra["ticker"] = extra["ticker"].astype(str).str.strip()
        out = pd.concat([out, extra], ignore_index=True)
    out = out.drop_duplicates(subset="ticker", keep="first").reset_index(drop=True)
    return out


def build_universe(cfg: UniverseConfig, output_file: str) -> pd.DataFrame:
    """Run the full auto-build and write the universe CSV."""
    jpx = fetch_jpx_list(cfg.jpx_url)
    candidates = filter_segments(jpx, cfg.segments)
    prices = fetch_screen_prices(candidates["ticker"].tolist(), cfg.screen_lookback_days)
    screened = screen_by_liquidity(prices, cfg.min_turnover_jpy, cfg.max_size)

    extra = None
    from pathlib import Path

    extra_path = Path(cfg.extra_file)
    if extra_path.exists():
        extra = pd.read_csv(extra_path, dtype=str)

    universe = merge_with_extra(screened, candidates, extra)
    universe.to_csv(output_file, index=False)
    log_event(
        logger,
        "Universe file written",
        path=output_file,
        n_total=len(universe),
        n_screened=len(screened),
    )
    return universe
