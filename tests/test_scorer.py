"""Tests for scoring and explainability."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from stocklab.models.trainer import build_pipeline
from stocklab.scoring.scorer import Scorer

FEATURES = ["f1", "f2", "f3"]


def _fitted_pipeline(n: int = 200) -> object:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(n, len(FEATURES)))
    y = 0.5 * x[:, 0] - 0.2 * x[:, 1] + rng.normal(0, 0.01, size=n)
    pipe = build_pipeline("ridge", {"alpha": 1e-6})
    pipe.fit(pd.DataFrame(x, columns=FEATURES), y)
    assert isinstance(pipe.named_steps["model"], Ridge)
    return pipe


def _latest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["T00", "T01", "T02"],
            "name": ["Alpha", "Beta", "Gamma"],
            "f1": [1.0, 0.0, -1.0],
            "f2": [0.0, 0.0, 0.0],
            "f3": [0.0, 0.0, 0.0],
            "filter_pass": [True, True, False],
        }
    )


def test_contributions_sum_to_prediction():
    """Contribution sum + intercept must reproduce the prediction (linear)."""
    pipe = _fitted_pipeline()
    scorer = Scorer(FEATURES, pipeline=pipe)
    results = scorer.score_latest(_latest())
    intercept = float(pipe.named_steps["model"].intercept_)
    for s in results:
        contrib_sum = sum(v for _, v in s.contributions) / 100.0
        assert math.isclose(contrib_sum + intercept, s.raw_pred, abs_tol=1e-2)


def test_ranking_follows_coefficients():
    """With coef f1>0, the ticker with the largest f1 must rank first."""
    scorer = Scorer(FEATURES, pipeline=_fitted_pipeline())
    results = scorer.score_latest(_latest())
    assert results[0].ticker == "T00"
    assert results[0].score >= results[-1].score
    assert results[-1].filter_pass is False  # T02 fails the filter


def test_fallback_without_model():
    """With no model, the equal-weight fallback still returns scores."""
    scorer = Scorer(FEATURES, pipeline=None)
    results = scorer.score_latest(_latest())
    assert len(results) == 3
    assert all(0.0 <= s.score <= 100.0 for s in results)


def test_missing_features_flagged_as_concern():
    latest = _latest()
    noise = latest.copy()
    noise.loc[0, "f2"] = np.nan
    noise.loc[0, "f3"] = np.nan
    scorer = Scorer(FEATURES, pipeline=_fitted_pipeline())
    scored = {s.ticker: s for s in scorer.score_latest(noise)}
    assert any("missing" in c.lower() for c in scored["T00"].concerns)

def test_filter_pass_nan_treated_as_fail():
    latest = _latest()
    latest["filter_pass"] = [True, np.nan, False]
    scorer = Scorer(FEATURES, pipeline=None)
    scored = {s.ticker: s for s in scorer.score_latest(latest)}
    assert scored["T00"].filter_pass is True
    assert scored["T01"].filter_pass is False  # NaN counts as a fail
    assert scored["T02"].filter_pass is False
