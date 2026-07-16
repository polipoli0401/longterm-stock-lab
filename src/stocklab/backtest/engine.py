"""Backtest engine.

Takes only walk-forward out-of-sample predictions as input and evaluates
the strategy "on each prediction date, buy the top-N ranked tickers with
equal weight and rotate after the holding period" (spec section 8).

No look-ahead:
    - Predictions are expected to come from the purged walk-forward.
    - Selection on each entry date uses only that date's predictions and
      filters.
    - No trade is generated when the exit price does not exist yet.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stocklab.backtest.metrics import equity_curve, summarize
from stocklab.config import BacktestConfig
from stocklab.logger import get_logger, log_event

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """Full backtest output."""

    model_name: str
    metrics: dict[str, float]
    trades: pd.DataFrame
    equity: pd.DataFrame


class BacktestEngine:
    """Backtests the top-N rotation strategy."""

    def __init__(
        self,
        cfg: BacktestConfig,
        prices: pd.DataFrame,
        benchmark_close: pd.Series | None,
        filters: pd.DataFrame | None = None,
    ) -> None:
        """Args:
        cfg: Backtest settings.
        prices: Long-format price data (the universe).
        benchmark_close: Benchmark close series (index = date).
        filters: Optional filters with ``date, ticker, filter_pass``.
        """
        self.cfg = cfg
        px = prices.drop_duplicates(subset=["date", "ticker"], keep="last")
        self.px = px.pivot(index="date", columns="ticker", values="close").sort_index()
        self.bench = (
            benchmark_close.sort_index().reindex(self.px.index).ffill()
            if benchmark_close is not None and not benchmark_close.empty
            else None
        )
        self.filters = (
            filters[["date", "ticker", "filter_pass"]].copy() if filters is not None else None
        )

    def run(self, predictions: pd.DataFrame, model_name: str = "model") -> BacktestResult:
        """Run the backtest.

        Args:
            predictions: OOS predictions with ``date, ticker, y_pred``.
            model_name: Model name used in reports.

        Returns:
            :class:`BacktestResult`.

        Raises:
            ValueError: When no trade can be generated (period too short).
        """
        preds = predictions.copy()
        preds["date"] = pd.to_datetime(preds["date"])
        index = self.px.index
        pred_dates = sorted(d for d in preds["date"].unique() if d in index)
        if not pred_dates:
            raise ValueError("Prediction dates do not match the price data")

        holding = self.cfg.holding_days
        trades: list[dict] = []
        pos = 0
        while pos < len(pred_dates):
            entry = pred_dates[pos]
            i = index.get_loc(entry)
            j = i + holding
            if j >= len(index):
                break
            exit_date = index[j]

            day = preds[preds["date"] == entry].dropna(subset=["y_pred"])
            day = self._apply_filters(day, entry)
            day = day.sort_values("y_pred", ascending=False).head(self.cfg.top_n)

            selected: list[str] = []
            rets: list[float] = []
            for ticker in day["ticker"]:
                if ticker not in self.px.columns:
                    continue
                p0 = self.px.at[entry, ticker]
                p1 = self.px.at[exit_date, ticker]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    selected.append(str(ticker))
                    rets.append(float(p1 / p0 - 1.0))

            trades.append(
                {
                    "entry_date": entry,
                    "exit_date": exit_date,
                    "tickers": ",".join(selected),
                    "n_positions": len(selected),
                    "return": float(np.mean(rets)) if rets else 0.0,
                    "benchmark_return": self._bench_return(entry, exit_date),
                }
            )
            pos = bisect.bisect_left(pred_dates, exit_date)

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            raise ValueError("Backtest period too short to generate any trades")

        returns = pd.Series(
            trades_df["return"].to_numpy(),
            index=pd.DatetimeIndex(trades_df["exit_date"]),
            name="strategy",
        )
        bench_returns = pd.Series(
            trades_df["benchmark_return"].to_numpy(), index=returns.index, name="benchmark"
        )
        metrics = summarize(returns, bench_returns, holding)
        equity = pd.DataFrame(
            {
                "strategy": equity_curve(returns),
                "benchmark": equity_curve(bench_returns.fillna(0.0)),
            }
        )
        log_event(
            logger,
            "Backtest complete",
            model=model_name,
            n_trades=len(trades_df),
            cagr=_round(metrics["cagr"]),
            sharpe=_round(metrics["sharpe"]),
            max_drawdown=_round(metrics["max_drawdown"]),
            excess_return=_round(metrics["excess_return"]),
        )
        return BacktestResult(model_name, metrics, trades_df, equity)

    def _apply_filters(self, day: pd.DataFrame, entry: pd.Timestamp) -> pd.DataFrame:
        """Apply entry-date technical filters."""
        if self.filters is None:
            return day
        f = self.filters[self.filters["date"] == entry]
        merged = day.merge(f, on=["date", "ticker"], how="left")
        mask = merged["filter_pass"].map(lambda v: bool(v) if pd.notna(v) else False)
        return merged[mask.to_numpy()]

    def _bench_return(self, entry: pd.Timestamp, exit_date: pd.Timestamp) -> float:
        """Benchmark return over the same period."""
        if self.bench is None:
            return float("nan")
        b0 = self.bench.loc[entry]
        b1 = self.bench.loc[exit_date]
        if pd.notna(b0) and pd.notna(b1) and b0 > 0:
            return float(b1 / b0 - 1.0)
        return float("nan")


def _round(value: float, digits: int = 4) -> float | None:
    """Safe rounding for log payloads."""
    try:
        if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None
