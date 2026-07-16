"""Tests for the walk-forward splitter (leak prevention is the key check)."""

from __future__ import annotations

import pandas as pd
import pytest

from stocklab.models.trainer import WalkForwardSplitter


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


def test_purge_gap_between_train_and_test():
    """The purge must keep >= horizon trading days between train end and test start."""
    horizon = 10
    splitter = WalkForwardSplitter(min_train_days=50, step_days=20, horizon_days=horizon)
    dates = _dates(120)
    folds = splitter.split(dates)
    assert folds, "at least one fold should be produced"
    lookup = {d: i for i, d in enumerate(dates)}
    for fold in folds:
        train_pos = lookup[fold.train_end]
        test_pos = lookup[fold.test_start]
        # Purge guarantee: samples whose label window overlaps the test set
        # are excluded from training.
        assert train_pos + horizon <= test_pos


def test_folds_cover_the_tail():
    splitter = WalkForwardSplitter(min_train_days=50, step_days=30, horizon_days=5)
    dates = _dates(200)
    folds = splitter.split(dates)
    assert folds[-1].test_end == dates[-1]
    # Test windows are contiguous and non-overlapping.
    for prev, cur in zip(folds, folds[1:]):
        assert prev.test_end < cur.test_start


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        WalkForwardSplitter(min_train_days=100, step_days=20, horizon_days=100)
    with pytest.raises(ValueError):
        WalkForwardSplitter(min_train_days=100, step_days=0, horizon_days=10)

def test_walk_forward_learns_signal():
    """OOS predictions must recover a known linear signal (no shuffling)."""
    import numpy as np

    from stocklab.config import ModelConfig
    from stocklab.models.trainer import ModelTrainer

    dates = _dates(200)
    rng = np.random.default_rng(0)
    rows = []
    for ticker in ("A", "B", "C"):
        x = rng.normal(size=len(dates))
        noise = rng.normal(scale=0.01, size=len(dates))
        for d, xv, nv in zip(dates, x, noise):
            rows.append({"date": d, "ticker": ticker, "x1": xv, "target": 0.5 * xv + nv})
    panel = pd.DataFrame(rows)

    cfg = ModelConfig(horizon_days=10, min_train_days=60, step_days=20, candidates=["linear"])
    predictions = ModelTrainer(cfg).walk_forward(panel, ["x1"], "linear")

    assert {"date", "ticker", "y_pred", "target"} <= set(predictions.columns)
    corr = float(np.corrcoef(predictions["y_pred"], predictions["target"])[0, 1])
    assert corr > 0.9
    # Predictions only exist after min_train_days (no in-sample mixing).
    assert predictions["date"].min() >= dates[60]
