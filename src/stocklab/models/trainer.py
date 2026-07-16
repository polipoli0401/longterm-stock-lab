"""Model training and walk-forward evaluation.

The initial implementation provides the linear family (Linear / Ridge /
Lasso / ElasticNet). Gradient-boosting models (e.g. LightGBM) can be added
simply by registering them in ``MODEL_BUILDERS``.

Data-leak prevention (spec section 10):
    - Training/evaluation strictly follow time order; no random shuffling.
    - Purged walk-forward: the target is "the return over the next
      ``horizon`` trading days from day t", so training samples whose label
      window overlaps the test period are removed (for a test start index
      i, training data is limited to dates[i - horizon] and earlier).
    - The final model is fitted only on samples whose target has realized.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from stocklab.config import ModelConfig
from stocklab.logger import get_logger, log_event

logger = get_logger(__name__)

MODEL_BUILDERS: dict[str, type] = {
    "linear": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elasticnet": ElasticNet,
}


def build_pipeline(name: str, params: dict[str, Any] | None = None) -> Pipeline:
    """Build an sklearn pipeline of imputation + estimator.

    Args:
        name: Model name (a key of ``MODEL_BUILDERS``).
        params: Hyperparameters passed to the estimator.

    Raises:
        ValueError: If the model name is unknown.
    """
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model name: {name} (choices: {sorted(MODEL_BUILDERS)})")
    estimator = MODEL_BUILDERS[name](**(params or {}))
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", estimator),
        ]
    )


def feature_importance(pipeline: Pipeline, feature_cols: list[str]) -> dict[str, float]:
    """Return feature importances (coefficients for linear models)."""
    estimator = pipeline.named_steps["model"]
    coef = getattr(estimator, "coef_", None)
    if coef is None:
        coef = getattr(estimator, "feature_importances_", None)
    if coef is None:
        return {}
    return {c: float(w) for c, w in zip(feature_cols, np.ravel(coef))}


@dataclass(frozen=True)
class Fold:
    """A single walk-forward split."""

    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class WalkForwardSplitter:
    """Purged walk-forward splitter."""

    def __init__(self, min_train_days: int, step_days: int, horizon_days: int) -> None:
        """Args:
        min_train_days: Trading days required before the first test window.
        step_days: Test-window length = retraining interval (trading days).
        horizon_days: Target look-ahead (trading days) = purge width.

        Raises:
            ValueError: If min_train_days is not greater than horizon_days.
        """
        if min_train_days <= horizon_days:
            raise ValueError("min_train_days must be greater than horizon_days")
        if step_days <= 0:
            raise ValueError("step_days must be positive")
        self.min_train_days = min_train_days
        self.step_days = step_days
        self.horizon_days = horizon_days

    def split(self, dates: Sequence[Any]) -> list[Fold]:
        """Create Folds from a date sequence (time-ordered, deduplicated)."""
        idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(dates)).unique())).sort_values()
        folds: list[Fold] = []
        i = self.min_train_days
        n = len(idx)
        while i < n:
            j = min(i + self.step_days, n)
            train_end_pos = i - self.horizon_days
            if train_end_pos > 0:
                folds.append(Fold(idx[train_end_pos], idx[i], idx[j - 1]))
            i = j
        return folds


class ModelTrainer:
    """Handles walk-forward training and the final fit."""

    MIN_TRAIN_ROWS = 50

    def __init__(self, cfg: ModelConfig) -> None:
        """Args:
        cfg: Model settings.
        """
        self.cfg = cfg
        self.splitter = WalkForwardSplitter(
            cfg.min_train_days, cfg.step_days, cfg.horizon_days
        )

    def _pipeline(self, name: str) -> Pipeline:
        return build_pipeline(name, self.cfg.params.get(name))

    def walk_forward(
        self, panel: pd.DataFrame, feature_cols: list[str], model_name: str
    ) -> pd.DataFrame:
        """Generate out-of-sample predictions via walk-forward.

        Args:
            panel: Panel with ``date, ticker, target`` + feature columns.
            feature_cols: Feature columns to use.
            model_name: Model name.

        Returns:
            DataFrame with columns ``date, ticker, y_pred, target``.

        Raises:
            RuntimeError: If no fold has enough training data.
        """
        folds = self.splitter.split(panel["date"].unique())
        preds: list[pd.DataFrame] = []
        used = 0
        for fold in folds:
            train = panel[(panel["date"] <= fold.train_end) & panel["target"].notna()]
            test = panel[(panel["date"] >= fold.test_start) & (panel["date"] <= fold.test_end)]
            if len(train) < self.MIN_TRAIN_ROWS or test.empty:
                continue
            pipe = self._pipeline(model_name)
            pipe.fit(train[feature_cols], train["target"])
            out = test[["date", "ticker", "target"]].copy()
            out["y_pred"] = pipe.predict(test[feature_cols])
            preds.append(out)
            used += 1
        if not preds:
            raise RuntimeError("Not enough training data for walk-forward")
        result = pd.concat(preds, ignore_index=True)
        log_event(
            logger,
            "Walk-forward complete",
            model=model_name,
            n_folds=used,
            n_predictions=len(result),
        )
        return result

    def predict_frozen(
        self,
        pipeline: Pipeline,
        panel: pd.DataFrame,
        feature_cols: list[str],
        after: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Predict with an already-fitted model, no retraining (champion eval).

        Args:
            pipeline: Fitted pipeline.
            panel: Panel to predict on (must include a target column).
            feature_cols: Same feature columns used at training time.
            after: Only predict on dates strictly after this (excludes the
                training period).
        """
        data = panel if after is None else panel[panel["date"] > after]
        if data.empty:
            return pd.DataFrame(columns=["date", "ticker", "y_pred", "target"])
        out = data[["date", "ticker", "target"]].copy()
        out["y_pred"] = pipeline.predict(data[feature_cols])
        return out

    def fit_final(
        self, panel: pd.DataFrame, feature_cols: list[str], model_name: str
    ) -> tuple[Pipeline, dict[str, float], pd.Timestamp]:
        """Fit the final model on all data whose target has realized.

        Returns:
            (fitted pipeline, feature importances, last training date).
        """
        train = panel[panel["target"].notna()]
        if train.empty:
            raise RuntimeError("No data available for the final fit")
        pipe = self._pipeline(model_name)
        pipe.fit(train[feature_cols], train["target"])
        importance = feature_importance(pipe, feature_cols)
        train_end = pd.Timestamp(train["date"].max())
        log_event(
            logger,
            "Final model fitted",
            model=model_name,
            n_rows=len(train),
            train_end=str(train_end.date()),
        )
        return pipe, importance, train_end
