"""Holdings risk analysis module.

Detects risk factors per holding, aggregates them into a score, and raises
an alert when the score crosses a threshold (spec section 11). It never
forces a sell; it only provides decision-support material.

Detected risk factors (computable from price/volume):
    - Below the long-term (200-day) moving average
    - Volatility spike
    - Volume dry-up
    - Large decline from the 52-week high
    - Upcoming earnings announcement (yfinance; ignored on failure)

Downward guidance revisions and margin-trading balance are not implemented
because no stable free data source exists (planned once a source such as
J-Quants is integrated; see README).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from stocklab.config import RiskConfig
from stocklab.logger import get_logger, log_event

logger = get_logger(__name__)


@dataclass
class RiskAssessment:
    """Risk-evaluation result for one holding."""

    ticker: str
    name: str
    score: int
    reasons: list[str] = field(default_factory=list)
    alert: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dict for JSON storage."""
        return {
            "ticker": self.ticker,
            "name": self.name,
            "score": self.score,
            "reasons": self.reasons,
            "alert": self.alert,
        }


class RiskAnalyzer:
    """Computes risk scores for current holdings."""

    REASON_LABELS = {
        "below_ma200": "Below the long-term (200-day) moving average",
        "vol_spike": "Volatility has spiked",
        "volume_dry": "Trading volume has dried up",
        "large_drawdown": "Large decline from the 52-week high",
        "earnings_soon": "Earnings announcement approaching",
    }

    def __init__(self, cfg: RiskConfig) -> None:
        """Args:
        cfg: Risk-analysis settings.
        """
        self.cfg = cfg

    def analyze(
        self, holdings: list[dict[str, Any]], prices: pd.DataFrame
    ) -> list[RiskAssessment]:
        """Evaluate risk for the given holdings.

        Args:
            holdings: List of ``{"ticker", "name", ...}`` dicts.
            prices: Long-format price data (must include the holdings).

        Returns:
            List of :class:`RiskAssessment` (empty when no holdings).
        """
        if not holdings:
            return []
        dedup = prices.drop_duplicates(subset=["date", "ticker"], keep="last")
        px = dedup.pivot(index="date", columns="ticker", values="close").sort_index()
        vol = dedup.pivot(index="date", columns="ticker", values="volume").sort_index()

        results: list[RiskAssessment] = []
        for holding in holdings:
            ticker = holding["ticker"]
            name = holding.get("name") or ticker
            if ticker not in px.columns or px[ticker].dropna().empty:
                results.append(
                    RiskAssessment(
                        ticker, name, 0, ["Price data unavailable (needs review)"], False
                    )
                )
                continue
            close = px[ticker].dropna()
            volume = vol[ticker].dropna() if ticker in vol.columns else pd.Series(dtype=float)
            flags = self._detect_flags(ticker, close, volume)
            score = int(sum(self.cfg.weights.get(flag, 0) for flag in flags))
            reasons = [self.REASON_LABELS[flag] for flag in flags]
            results.append(
                RiskAssessment(
                    ticker=ticker,
                    name=name,
                    score=score,
                    reasons=reasons,
                    alert=score >= self.cfg.notify_threshold,
                )
            )
        log_event(
            logger,
            "Holdings risk analysis complete",
            n_holdings=len(results),
            n_alerts=sum(1 for r in results if r.alert),
        )
        return results

    def _detect_flags(
        self, ticker: str, close: pd.Series, volume: pd.Series
    ) -> list[str]:
        """Detect risk-factor flags for a single ticker."""
        flags: list[str] = []

        if len(close) >= 200:
            ma200 = float(close.rolling(200).mean().iloc[-1])
            if float(close.iloc[-1]) < ma200:
                flags.append("below_ma200")

        rets = (close / close.shift(1) - 1.0).dropna()
        if len(rets) >= 120:
            recent = float(rets.tail(20).std())
            base = float(rets.tail(120).std())
            if base > 0 and recent / base >= self.cfg.vol_spike_ratio:
                flags.append("vol_spike")

        if len(volume) >= 120:
            recent_volume = float(volume.tail(20).mean())
            base_volume = float(volume.tail(120).mean())
            if base_volume > 0 and recent_volume / base_volume <= self.cfg.volume_dry_ratio:
                flags.append("volume_dry")

        if len(close) >= 30:
            high = float(close.tail(252).max())
            if high > 0 and float(close.iloc[-1]) / high - 1.0 <= -self.cfg.drawdown_threshold:
                flags.append("large_drawdown")

        if self.cfg.check_earnings and self._earnings_soon(ticker):
            flags.append("earnings_soon")
        return flags

    def _earnings_soon(self, ticker: str) -> bool:
        """Check whether an earnings date is near (False on any failure)."""
        try:
            import yfinance as yf  # lazy import (not required in test envs)

            calendar = yf.Ticker(ticker).calendar
            dates = calendar.get("Earnings Date") if isinstance(calendar, dict) else None
            if not dates:
                return False
            today = pd.Timestamp.today().normalize()
            for raw in dates:
                delta_days = (pd.Timestamp(raw) - today).days
                if 0 <= delta_days <= self.cfg.earnings_within_days:
                    return True
        except Exception:
            logger.debug("Could not fetch earnings date: %s", ticker)
        return False
