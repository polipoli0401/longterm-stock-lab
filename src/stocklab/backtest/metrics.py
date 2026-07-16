"""Backtest performance metrics.

Computes CAGR, Sharpe, Sortino, Calmar, profit factor, win rate, max
drawdown, excess return, etc. from a series of per-period returns
(one rebalance = one period; spec section 8).
"""

from __future__ import annotations

import math

import pandas as pd

TRADING_DAYS_PER_YEAR = 252

METRIC_KEYS = [
    "n_periods",
    "years",
    "cagr",
    "annual_return",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "profit_factor",
    "win_rate",
    "excess_return",
    "benchmark_cagr",
]


def equity_curve(returns: pd.Series) -> pd.Series:
    """Build an equity curve (starting at 1) from per-period returns."""
    return (1.0 + returns).cumprod()


def cagr(equity: pd.Series, years: float) -> float:
    """Compound annual growth rate; NaN when it cannot be computed."""
    if equity.empty or years <= 0:
        return float("nan")
    final = float(equity.iloc[-1])
    if final <= 0:
        return float("nan")
    return final ** (1.0 / years) - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown (a negative value)."""
    if equity.empty:
        return float("nan")
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def sharpe_ratio(returns: pd.Series, periods_per_year: float, risk_free: float = 0.0) -> float:
    """Annualized Sharpe ratio; NaN for <2 samples or zero variance."""
    if len(returns) < 2:
        return float("nan")
    excess = returns - risk_free / periods_per_year
    std = float(excess.std(ddof=1))
    if not std or math.isnan(std):
        return float("nan")
    return float(excess.mean()) / std * math.sqrt(periods_per_year)


def sortino_ratio(returns: pd.Series, periods_per_year: float, risk_free: float = 0.0) -> float:
    """Annualized Sortino ratio (downside-deviation based)."""
    if len(returns) < 2:
        return float("nan")
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    if len(downside) < 2:
        return float("nan")
    downside_std = float(downside.std(ddof=1))
    if not downside_std or math.isnan(downside_std):
        return float("nan")
    return float(excess.mean()) / downside_std * math.sqrt(periods_per_year)


def calmar_ratio(cagr_value: float, mdd: float) -> float:
    """Calmar ratio (CAGR / |max drawdown|)."""
    if math.isnan(cagr_value) or math.isnan(mdd) or mdd >= 0:
        return float("nan")
    return cagr_value / abs(mdd)


def profit_factor(returns: pd.Series) -> float:
    """Profit factor (gross gains / gross losses); inf when there are no losses."""
    if returns.empty:
        return float("nan")
    gains = float(returns[returns > 0].sum())
    losses = float(-returns[returns < 0].sum())
    if losses <= 0:
        return float("inf") if gains > 0 else float("nan")
    return gains / losses


def win_rate(returns: pd.Series) -> float:
    """Win rate (fraction of periods with a positive return)."""
    if returns.empty:
        return float("nan")
    return float((returns > 0).mean())


def summarize(
    returns: pd.Series, bench_returns: pd.Series, holding_days: int
) -> dict[str, float]:
    """Compute the full metric set from per-period returns.

    Args:
        returns: Strategy per-period returns (index = period end date).
        bench_returns: Benchmark returns over the same periods.
        holding_days: Trading days per period (used for annualization).

    Returns:
        Dict keyed by :data:`METRIC_KEYS`.
    """
    periods_per_year = TRADING_DAYS_PER_YEAR / holding_days
    n = len(returns)
    if n == 0:
        return {key: float("nan") for key in METRIC_KEYS}

    equity = equity_curve(returns)
    years = n / periods_per_year
    cagr_value = cagr(equity, years)
    mdd = max_drawdown(equity)

    bench = bench_returns.reindex(returns.index)
    excess = (returns - bench).dropna()
    bench_equity = equity_curve(bench.dropna())

    return {
        "n_periods": float(n),
        "years": round(years, 3),
        "cagr": cagr_value,
        "annual_return": float(returns.mean()) * periods_per_year,
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "sortino": sortino_ratio(returns, periods_per_year),
        "max_drawdown": mdd,
        "calmar": calmar_ratio(cagr_value, mdd),
        "profit_factor": profit_factor(returns),
        "win_rate": win_rate(returns),
        "excess_return": float(excess.mean()) * periods_per_year
        if not excess.empty
        else float("nan"),
        "benchmark_cagr": cagr(bench_equity, years) if not bench_equity.empty else float("nan"),
    }
