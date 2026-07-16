"""Scoring and explainability module.

Scores the latest cross-section with the champion model and always attaches
the reasoning (per-feature contributions; spec section 12). Because a
linear model's prediction is the sum of "coefficient x standardized
feature", contributions decompose additively.

- Composite score: percentile (0-100) of the prediction within the universe
- Contribution: coefficient x standardized feature, shown in predicted
  return percentage points
- No trained model: falls back to an equal-weight average of all features
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from stocklab.features.builder import FILTER_LABELS
from stocklab.logger import get_logger, log_event

logger = get_logger(__name__)


@dataclass
class ScoredTicker:
    """Scoring result for a single ticker."""

    ticker: str
    name: str
    score: float
    raw_pred: float
    contributions: list[tuple[str, float]] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    filter_pass: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dict for JSON storage."""
        return asdict(self)


class Scorer:
    """Computes composite scores with contribution breakdowns."""

    CONTRIB_EPS = 1e-4

    def __init__(
        self,
        feature_cols: list[str],
        pipeline: Any | None = None,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Args:
        feature_cols: Feature columns (same order as at training time).
        pipeline: Fitted pipeline; None triggers the fallback score.
        labels: Mapping from feature name to display label.
        """
        self.feature_cols = list(feature_cols)
        self.pipeline = pipeline
        self.labels = labels or {}

    def score_latest(self, latest: pd.DataFrame) -> list[ScoredTicker]:
        """Score the latest cross-section (one row per ticker).

        Args:
            latest: DataFrame with ``ticker, name`` + feature columns
                (cross-sectionally standardized) + optional filter columns.

        Returns:
            List of :class:`ScoredTicker`, sorted by score descending.
        """
        if latest.empty:
            return []
        data = latest.reset_index(drop=True)
        x_raw = data[self.feature_cols]

        if self.pipeline is not None:
            imputer = self.pipeline.named_steps["imputer"]
            estimator = self.pipeline.named_steps["model"]
            x_imp = np.asarray(imputer.transform(x_raw), dtype=float)
            coef = np.ravel(estimator.coef_)
            raw_pred = np.asarray(self.pipeline.predict(x_raw), dtype=float)
        else:
            x_imp = np.nan_to_num(x_raw.to_numpy(dtype=float), nan=0.0)
            coef = np.full(len(self.feature_cols), 1.0 / len(self.feature_cols))
            raw_pred = x_imp @ coef

        contrib = x_imp * coef  # (n tickers, n features)
        score = pd.Series(raw_pred).rank(pct=True, method="average") * 100.0

        results: list[ScoredTicker] = []
        for i in range(len(data)):
            row = data.iloc[i]
            pairs = sorted(
                zip(self.feature_cols, contrib[i]), key=lambda p: -abs(p[1])
            )
            contributions = [
                (self._label(name), round(float(value) * 100.0, 1))
                for name, value in pairs
                if abs(value) > self.CONTRIB_EPS
            ]
            results.append(
                ScoredTicker(
                    ticker=str(row["ticker"]),
                    name=str(row.get("name", row["ticker"])),
                    score=round(float(score.iloc[i]), 1),
                    raw_pred=round(float(raw_pred[i]), 5),
                    contributions=contributions,
                    concerns=self._concerns(row, pairs),
                    filter_pass=_filter_pass(row),
                )
            )
        results.sort(key=lambda s: s.score, reverse=True)
        log_event(
            logger,
            "Scoring complete",
            n_tickers=len(results),
            mode="model" if self.pipeline is not None else "fallback",
        )
        return results

    def _label(self, name: str) -> str:
        return self.labels.get(name, name)

    def _concerns(
        self, row: pd.Series, pairs: list[tuple[str, float]]
    ) -> list[str]:
        """List concerns (negative contributions, failed filters, missing data)."""
        concerns: list[str] = []
        negatives = [(n, v) for n, v in pairs if v < -1e-3][:3]
        for name, value in negatives:
            concerns.append(f"{self._label(name)} contributes negatively ({value * 100:+.1f})")
        for col, message in FILTER_LABELS.items():
            if col in row.index:
                value = row[col]
                if value is not None and not pd.isna(value) and not bool(value):
                    concerns.append(message)
        n_missing = int(row[self.feature_cols].isna().sum())
        if n_missing >= max(1, len(self.feature_cols) // 3):
            concerns.append(f"Many missing features ({n_missing}/{len(self.feature_cols)})")
        return concerns


def _filter_pass(row: pd.Series) -> bool:
    """Safely read the filter flag (missing values count as a fail)."""
    if "filter_pass" not in row.index:
        return True
    value = row["filter_pass"]
    if value is None or pd.isna(value):
        return False
    return bool(value)
