"""Feature engineering module.

Builds the following from price series and fundamental data:

- Model features: growth, financial quality, value, momentum
- Technical filters: moving average / RSI / liquidity (never used as
  learned weights; spec section 5)
- Target variable: return over the next ``horizon`` trading days
  (excess vs benchmark, or absolute)

Data-leak prevention:
    Fundamentals become usable only from ``fiscal end + publication_lag_days``
    (a point-in-time approximation). Targets are computed from future prices,
    so unrealized periods are automatically NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stocklab.config import FeatureConfig
from stocklab.logger import get_logger, log_event

logger = get_logger(__name__)

# Fundamental-derived features (cross-sectionally standardized, fed to the model)
FUNDAMENTAL_BASE_FEATURES = [
    "rev_growth",       # revenue growth (YoY)
    "eps_growth",       # EPS growth (YoY)
    "op_margin_delta",  # change in operating margin
    "roe",
    "roa",
    "op_margin",
    "ocf_margin",       # operating cash-flow margin
    "equity_ratio",     # equity-to-assets ratio
]
VALUE_FEATURES = [
    "earnings_yield",  # earnings yield (= 1/PER; more outlier-robust than PER)
    "book_to_price",   # inverse of price-to-book (= 1/PBR)
    "fcf_yield",       # free-cash-flow yield
]
FUNDAMENTAL_FEATURES = [*FUNDAMENTAL_BASE_FEATURES, *VALUE_FEATURES]

FEATURE_LABELS: dict[str, str] = {
    "mom_21": "1-month momentum",
    "mom_63": "3-month momentum",
    "mom_126": "6-month momentum",
    "volatility": "Volatility",
    "rev_growth": "Revenue growth",
    "eps_growth": "EPS growth",
    "op_margin_delta": "Margin improvement",
    "roe": "ROE",
    "roa": "ROA",
    "op_margin": "Operating margin",
    "ocf_margin": "Operating CF margin",
    "equity_ratio": "Equity ratio",
    "earnings_yield": "Value (earnings yield)",
    "book_to_price": "Value (book-to-price)",
    "fcf_yield": "FCF yield",
}

FILTER_LABELS: dict[str, str] = {
    "f_trend": "Below the long-term (200-day) moving average",
    "f_heat": "RSI in overheated territory",
    "f_liquid": "Liquidity (turnover) below threshold",
}


class FeatureBuilder:
    """Builds model features, technical filters, and the target variable."""

    def __init__(self, cfg: FeatureConfig, publication_lag_days: int = 90) -> None:
        """Args:
        cfg: Feature settings.
        publication_lag_days: Publication lag in days (leak prevention).
        """
        self.cfg = cfg
        self.publication_lag_days = publication_lag_days

    @property
    def model_features(self) -> list[str]:
        """Feature columns fed to the model (follows the configuration)."""
        moms = [f"mom_{w}" for w in self.cfg.momentum_windows]
        return [*moms, "volatility", *FUNDAMENTAL_FEATURES]

    # ------------------------------------------------------------ build
    def build(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame | None,
        shares: dict[str, float] | None,
    ) -> pd.DataFrame:
        """Build a feature panel combining prices and fundamentals.

        Args:
            prices: Long-format price data (date, ticker, close, ...).
            fundamentals: Fundamental data (ticker, fiscal_end, ...); may be
                empty.
            shares: {ticker: shares outstanding}, used for value metrics.

        Returns:
            DataFrame with ``date, ticker, close`` + :attr:`model_features`.
        """
        price_feats = self.build_price_features(prices)
        fund = self.build_fundamental_features(fundamentals)

        fund_cols = [*FUNDAMENTAL_BASE_FEATURES, "net_income", "equity", "free_cf"]
        if fund.empty:
            merged = price_feats.copy()
            for col in fund_cols:
                merged[col] = np.nan
        else:
            merged = pd.merge_asof(
                price_feats.sort_values("date"),
                fund[["ticker", "effective_date", *fund_cols]].sort_values("effective_date"),
                left_on="date",
                right_on="effective_date",
                by="ticker",
                direction="backward",
            )

        shares = shares or {}
        merged["mcap"] = merged["close"] * merged["ticker"].map(shares)
        merged["earnings_yield"] = merged["net_income"] / merged["mcap"]
        merged["book_to_price"] = merged["equity"] / merged["mcap"]
        merged["fcf_yield"] = merged["free_cf"] / merged["mcap"]
        merged = merged.replace([np.inf, -np.inf], np.nan)

        out = merged.reindex(columns=["date", "ticker", "close", *self.model_features])
        log_event(
            logger,
            "Feature panel built",
            n_rows=len(out),
            n_features=len(self.model_features),
            has_fundamentals=not fund.empty,
        )
        return out

    # --------------------------------------------------- price features
    def build_price_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Build price-derived features (momentum, volatility, ...)."""
        px = _pivot(prices, "close")
        rets = px / px.shift(1) - 1.0

        feats: dict[str, pd.DataFrame] = {"close": px}
        for w in self.cfg.momentum_windows:
            feats[f"mom_{w}"] = px / px.shift(w) - 1.0
        feats["volatility"] = rets.rolling(self.cfg.volatility_window).std() * np.sqrt(252.0)

        long = _merge_wide(feats)
        return long.dropna(subset=["close"]).reset_index(drop=True)

    # --------------------------------------------- fundamental features
    def build_fundamental_features(
        self, fundamentals: pd.DataFrame | None
    ) -> pd.DataFrame:
        """Derive fundamental metrics and attach their usable-from date.

        Note:
            EPS growth has an unstable sign when the prior-period EPS is
            negative; the downstream winsorized cross-sectional
            standardization limits the impact.
        """
        out_cols = [
            "ticker",
            "effective_date",
            *FUNDAMENTAL_BASE_FEATURES,
            "net_income",
            "equity",
            "free_cf",
        ]
        if fundamentals is None or fundamentals.empty:
            return pd.DataFrame(columns=out_cols)

        f = fundamentals.copy()
        f["fiscal_end"] = pd.to_datetime(f["fiscal_end"])
        f = f.sort_values(["ticker", "fiscal_end"]).reset_index(drop=True)
        g = f.groupby("ticker")

        f["rev_growth"] = f["revenue"] / g["revenue"].shift(1) - 1.0
        f["eps_growth"] = f["eps"] / g["eps"].shift(1) - 1.0
        f["op_margin"] = f["operating_income"] / f["revenue"]
        f["op_margin_delta"] = f["op_margin"] - f.groupby("ticker")["op_margin"].shift(1)
        f["roe"] = f["net_income"] / f["equity"]
        f["roa"] = f["net_income"] / f["total_assets"]
        f["ocf_margin"] = f["operating_cf"] / f["revenue"]
        f["equity_ratio"] = f["equity"] / f["total_assets"]
        f["effective_date"] = f["fiscal_end"] + pd.Timedelta(days=self.publication_lag_days)
        f = f.replace([np.inf, -np.inf], np.nan)
        return f[out_cols]

    # ----------------------------------------------------------- filter
    def build_filters(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Build technical filters (never used as learned weights).

        - f_trend: above the long moving average (max of ma_windows)
        - f_heat: RSI below the overheat threshold (rsi_overheat)
        - f_liquid: 20-day average turnover at/above the threshold

        Returns:
            Columns ``date, ticker, f_trend, f_heat, f_liquid, filter_pass``.
        """
        px = _pivot(prices, "close")
        vol = _pivot(prices, "volume")

        long_ma = px.rolling(max(self.cfg.ma_windows)).mean()
        rsi = _rsi(px, self.cfg.rsi_window)
        turnover = (px * vol).rolling(20).mean()

        flags = {
            "f_trend": px > long_ma,
            "f_heat": rsi < self.cfg.rsi_overheat,
            "f_liquid": turnover >= self.cfg.min_turnover_jpy,
        }
        long = _merge_wide(flags)
        for col in flags:
            long[col] = long[col].astype(bool)
        long["filter_pass"] = long[list(flags)].all(axis=1)
        return long

    # ----------------------------------------------------------- target
    def build_target(
        self,
        prices: pd.DataFrame,
        benchmark_close: pd.Series | None,
        horizon_days: int,
        target_type: str = "excess",
    ) -> pd.DataFrame:
        """Build the target (return over the next ``horizon_days``).

        Args:
            prices: Long-format price data.
            benchmark_close: Benchmark close series (index = date).
            horizon_days: Look-ahead period (trading days).
            target_type: ``excess`` (vs benchmark) or ``absolute``.

        Returns:
            Columns ``date, ticker, target``; unrealized periods are NaN.
        """
        px = _pivot(prices, "close")
        fwd = px.shift(-horizon_days) / px - 1.0
        if target_type == "excess":
            if benchmark_close is None or benchmark_close.empty:
                raise ValueError("target_type=excess requires a benchmark series")
            bench = benchmark_close.sort_index().reindex(px.index).ffill()
            bench_fwd = bench.shift(-horizon_days) / bench - 1.0
            fwd = fwd.sub(bench_fwd, axis=0)
        elif target_type != "absolute":
            raise ValueError(f"Unknown target_type: {target_type}")
        return _to_long(fwd, "target")

    # ---------------------------------------------------- normalization
    @staticmethod
    def standardize_cross_section(
        df: pd.DataFrame, cols: list[str], clip_sigma: float = 3.0
    ) -> pd.DataFrame:
        """Z-score features within each date's cross-section and clip outliers.

        Only same-day cross-sectional information is used, so no
        time-direction leakage occurs. Also tames scale differences and
        outliers for linear models.
        """
        out = df.copy()
        grouped = out.groupby("date")[cols]
        z = (out[cols] - grouped.transform("mean")) / grouped.transform("std")
        z = z.replace([np.inf, -np.inf], np.nan)
        out[cols] = z.clip(-clip_sigma, clip_sigma)
        return out


# ------------------------------------------------------------------ util
def _pivot(prices: pd.DataFrame, value: str) -> pd.DataFrame:
    """Convert long-format prices to wide format (date x ticker)."""
    df = prices.drop_duplicates(subset=["date", "ticker"], keep="last")
    return df.pivot(index="date", columns="ticker", values=value).sort_index()


def _to_long(wide: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Convert wide format (index=date, columns=ticker) to long format."""
    df = wide.reset_index().melt(id_vars="date", var_name="ticker", value_name=value_name)
    return df


def _merge_wide(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge multiple wide frames into one long frame keyed by (date, ticker)."""
    long: pd.DataFrame | None = None
    for name, wide in frames.items():
        part = _to_long(wide, name)
        long = part if long is None else long.merge(part, on=["date", "ticker"], how="left")
    assert long is not None
    return long


def _rsi(px: pd.DataFrame, window: int) -> pd.DataFrame:
    """Wilder-style RSI (EWM approximation)."""
    delta = px.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / window, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)
