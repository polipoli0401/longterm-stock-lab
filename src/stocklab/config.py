"""Configuration module.

Keeps YAML configuration (non-secret) and environment variables (secrets)
strictly separated:

- Every tunable parameter is centralized in ``config/config.yaml``.
- Secrets (webhook URLs, tokens, holdings) are read from environment
  variables only (GitHub Secrets on Actions) and never stored in config
  files or the repository.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _filter_kwargs(cls: type, data: dict[str, Any] | None) -> dict[str, Any]:
    """Keep only keys that exist as dataclass fields (ignore unknown keys)."""
    data = data or {}
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in valid}


@dataclass(frozen=True)
class DataConfig:
    """Data acquisition and storage settings."""

    universe_file: str = "config/universe.csv"
    price_lookback_days: int = 1100
    train_lookback_days: int = 3700
    fundamental_refresh_days: int = 7
    publication_lag_days: int = 90
    benchmark: str = "1306.T"
    benchmark_alt: str = "^N225"
    db_path: str = "data/stocklab.db"


@dataclass(frozen=True)
class FeatureConfig:
    """Feature engineering settings."""

    momentum_windows: list[int] = field(default_factory=lambda: [21, 63, 126])
    volatility_window: int = 63
    ma_windows: list[int] = field(default_factory=lambda: [20, 75, 200])
    rsi_window: int = 14
    rsi_overheat: float = 80.0
    winsorize_sigma: float = 3.0
    min_turnover_jpy: float = 1.0e8


@dataclass(frozen=True)
class SelectionConfig:
    """Candidate-selection constraints (budget, lot size)."""

    unit_shares: int = 100
    max_unit_cost_jpy: float = 0.0  # 0 disables the affordability filter


@dataclass(frozen=True)
class ModelConfig:
    """Model training / walk-forward settings."""

    horizon_days: int = 126
    target_type: str = "excess"
    candidates: list[str] = field(
        default_factory=lambda: ["ridge", "lasso", "elasticnet", "linear"]
    )
    min_train_days: int = 750
    step_days: int = 63
    model_dir: str = "models"
    params: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest settings."""

    top_n: int = 3
    holding_days: int = 126
    report_dir: str = "reports"


@dataclass(frozen=True)
class RiskConfig:
    """Holdings risk-analysis settings."""

    vol_spike_ratio: float = 1.8
    volume_dry_ratio: float = 0.5
    drawdown_threshold: float = 0.25
    notify_threshold: int = 40
    check_earnings: bool = True
    earnings_within_days: int = 7
    weights: dict[str, int] = field(
        default_factory=lambda: {
            "below_ma200": 25,
            "vol_spike": 25,
            "volume_dry": 15,
            "large_drawdown": 25,
            "earnings_soon": 10,
        }
    )


@dataclass(frozen=True)
class AdoptionConfig:
    """Model adoption (composite score) settings.

    composite = w_sharpe*Sharpe + w_cagr*CAGR + w_excess*excess return
                - w_mdd*|max drawdown|
    """

    w_sharpe: float = 0.4
    w_cagr: float = 0.2
    w_excess: float = 0.3
    w_mdd: float = 0.1
    min_improvement: float = 0.0


@dataclass(frozen=True)
class Config:
    """Application-wide configuration."""

    data: DataConfig
    features: FeatureConfig
    selection: SelectionConfig
    model: ModelConfig
    backtest: BacktestConfig
    risk: RiskConfig
    adoption: AdoptionConfig

    @classmethod
    def load(cls, path: str | Path = "config/config.yaml") -> Config:
        """Load the YAML config file; fall back to defaults if it is absent."""
        raw: dict[str, Any] = {}
        p = Path(path)
        if p.exists():
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return cls(
            data=DataConfig(**_filter_kwargs(DataConfig, raw.get("data"))),
            features=FeatureConfig(**_filter_kwargs(FeatureConfig, raw.get("features"))),
            selection=SelectionConfig(**_filter_kwargs(SelectionConfig, raw.get("selection"))),
            model=ModelConfig(**_filter_kwargs(ModelConfig, raw.get("model"))),
            backtest=BacktestConfig(**_filter_kwargs(BacktestConfig, raw.get("backtest"))),
            risk=RiskConfig(**_filter_kwargs(RiskConfig, raw.get("risk"))),
            adoption=AdoptionConfig(**_filter_kwargs(AdoptionConfig, raw.get("adoption"))),
        )


def get_secret(name: str) -> str | None:
    """Read a secret from an environment variable.

    Returns ``None`` when the variable is unset or empty.
    Never write the returned value to logs or reports.
    """
    value = os.environ.get(name, "").strip()
    return value or None
