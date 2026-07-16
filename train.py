"""Model training, evaluation and adoption entry point (weekly).

Evaluates candidate models out-of-sample with purged walk-forward analysis
and compares them against the current champion using the composite backtest
score. A new model is adopted only when it improves on the champion
(spec section 9).

Fair-comparison design:
    The current champion is evaluated frozen (no retraining) only on the
    period after its training-data end date (``train_end``); candidates are
    re-aggregated over the same window. When that window has too few trades,
    the current model is kept (fail-safe).

Usage:
    python train.py [--config config/config.yaml] [--notify]
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from typing import Any

import pandas as pd

from stocklab.backtest.engine import BacktestEngine
from stocklab.config import Config
from stocklab.logger import get_logger, log_event, setup_logging
from stocklab.models.registry import ModelRegistry
from stocklab.models.trainer import ModelTrainer
from stocklab.notify.notifier import notify_all
from stocklab.pipeline import prepare_training_data
from stocklab.report.generator import ReportGenerator

logger = get_logger("stocklab.train")

MIN_COMPARE_PERIODS = 2  # minimum trades for comparison (Sharpe needs >= 2)


def run(config_path: str, notify: bool = False) -> int:
    """Run training, evaluation and the adoption decision.

    Args:
        config_path: Path to the config file.
        notify: Send the decision as a notification when True.

    Returns:
        Exit code (0 = success).
    """
    cfg = Config.load(config_path)
    data = prepare_training_data(cfg)
    trainer = ModelTrainer(cfg.model)
    engine = BacktestEngine(cfg.backtest, data.prices, data.bench_close, data.filters)
    registry = ModelRegistry(cfg.model.model_dir)
    reporter = ReportGenerator(cfg.backtest.report_dir)
    today = datetime.now().strftime("%Y-%m-%d")

    # --- walk-forward evaluation of candidates ------------------------------
    candidates: list[dict[str, Any]] = []
    for name in cfg.model.candidates:
        try:
            predictions = trainer.walk_forward(data.panel, data.feature_cols, name)
            result = engine.run(predictions, model_name=name)
            composite = registry.composite_score(result.metrics, cfg.adoption)
            candidates.append(
                {
                    "model_name": name,
                    "metrics": result.metrics,
                    "composite": composite,
                    "predictions": predictions,
                }
            )
            reporter.save(
                reporter.backtest_markdown(
                    result,
                    extra={
                        "Evaluation": "Purged walk-forward (OOS predictions only)",
                        "Run date": today,
                    },
                ),
                f"backtest/{today}_{name}.md",
            )
        except Exception:
            logger.exception("Candidate evaluation failed: %s", name)
    if not candidates:
        log_event(logger, "No candidate model could be evaluated", level=logging.ERROR)
        return 1

    best_full = max(candidates, key=lambda c: c["composite"])

    # --- comparison with the champion & decision -----------------------------
    champion_model, champion_meta = registry.load_champion()
    champion_eval: dict[str, Any] | None = None
    decision: dict[str, Any]

    if champion_meta is None or champion_model is None:
        decision = _adopt(best_full, "First training run - adopted (no champion to compare)")
    else:
        compatible = _champion_compatible(champion_meta, data.panel, cfg)
        if not compatible:
            decision = _adopt(
                best_full,
                "Feature set or target definition changed - adopting the retrained model",
            )
        else:
            train_end = pd.Timestamp(champion_meta["train_end"])
            champion_eval = _evaluate_champion(
                trainer, engine, registry, champion_model, champion_meta, data.panel, cfg, train_end
            )
            compare = _compare_on_window(engine, registry, candidates, cfg, train_end)
            decision = _decide(compare, champion_eval, champion_meta, cfg)

    # --- on adoption: final fit and save ---------------------------------------
    if decision["adopt"]:
        adopted_name = decision["model_name"]
        final_model, importance, train_end = trainer.fit_final(
            data.panel, data.feature_cols, adopted_name
        )
        adopted = next(c for c in candidates if c["model_name"] == adopted_name)
        meta = {
            "model_name": adopted_name,
            "params": cfg.model.params.get(adopted_name, {}),
            "feature_cols": data.feature_cols,
            "horizon_days": cfg.model.horizon_days,
            "target_type": cfg.model.target_type,
            "metrics": adopted["metrics"],
            "composite": adopted["composite"],
            "importance": importance,
            "train_end": str(train_end.date()),
            "adopted_at": today,
            "reason": decision["reason"],
        }
        registry.save_champion(final_model, meta)

    # --- record, report, notify -------------------------------------------------
    registry.record_decision(
        {
            **{k: v for k, v in decision.items() if k != "predictions"},
            "candidates": [
                {"model_name": c["model_name"], "composite": c["composite"]}
                for c in candidates
            ],
            "champion_eval": (
                {"composite": champion_eval["composite"]} if champion_eval else None
            ),
        }
    )
    reporter.save(
        reporter.decision_markdown(decision, candidates, champion_eval),
        f"model_decision/{today}.md",
    )
    if notify:
        verdict = "adopt new model" if decision["adopt"] else "keep current model"
        notify_all(
            f"🧪 Weekly model evaluation {today}\n"
            f"Decision: {verdict} ({decision['model_name']})\n"
            f"Reason: {decision['reason']}"
        )
    log_event(
        logger,
        "Weekly training pipeline complete",
        adopt=decision["adopt"],
        model=decision["model_name"],
    )
    return 0


def _adopt(candidate: dict[str, Any], reason: str) -> dict[str, Any]:
    """Build an adopt=True decision."""
    return {"adopt": True, "model_name": candidate["model_name"], "reason": reason}


def _champion_compatible(
    meta: dict[str, Any], panel: pd.DataFrame, cfg: Config
) -> bool:
    """Check whether the champion is compatible with the current config/panel."""
    feature_cols = meta.get("feature_cols") or []
    if not feature_cols or any(c not in panel.columns for c in feature_cols):
        return False
    if meta.get("horizon_days") != cfg.model.horizon_days:
        return False
    if meta.get("target_type") != cfg.model.target_type:
        return False
    return bool(meta.get("train_end"))


def _evaluate_champion(
    trainer: ModelTrainer,
    engine: BacktestEngine,
    registry: ModelRegistry,
    model: Any,
    meta: dict[str, Any],
    panel: pd.DataFrame,
    cfg: Config,
    train_end: pd.Timestamp,
) -> dict[str, Any] | None:
    """Evaluate the current champion frozen, on the post-training period only."""
    realized = panel[panel["target"].notna()]
    predictions = trainer.predict_frozen(
        model, realized, list(meta["feature_cols"]), after=train_end
    )
    metrics = _window_metrics(engine, predictions, f"champion:{meta.get('model_name')}")
    if metrics is None:
        log_event(logger, "Champion's post-training evaluation window is insufficient")
        return None
    composite = registry.composite_score(metrics, cfg.adoption)
    if composite == float("-inf"):
        return None
    return {
        "model_name": str(meta.get("model_name")),
        "metrics": metrics,
        "composite": composite,
    }


def _compare_on_window(
    engine: BacktestEngine,
    registry: ModelRegistry,
    candidates: list[dict[str, Any]],
    cfg: Config,
    after: pd.Timestamp,
) -> list[dict[str, Any]]:
    """Re-aggregate candidates over the common post-champion-training window."""
    compare: list[dict[str, Any]] = []
    for c in candidates:
        window = c["predictions"]
        window = window[pd.to_datetime(window["date"]) > after]
        metrics = _window_metrics(engine, window, f"{c['model_name']} (compare window)")
        if metrics is None:
            continue
        composite = registry.composite_score(metrics, cfg.adoption)
        if composite == float("-inf"):
            continue
        compare.append({"model_name": c["model_name"], "composite": composite})
    return compare


def _window_metrics(
    engine: BacktestEngine, predictions: pd.DataFrame, label: str
) -> dict[str, float] | None:
    """Backtest metrics over the comparison window (None when too short)."""
    if predictions.empty:
        return None
    try:
        result = engine.run(predictions, model_name=label)
    except ValueError:
        return None
    if result.metrics.get("n_periods", 0) < MIN_COMPARE_PERIODS:
        return None
    return result.metrics


def _decide(
    compare: list[dict[str, Any]],
    champion_eval: dict[str, Any] | None,
    champion_meta: dict[str, Any],
    cfg: Config,
) -> dict[str, Any]:
    """Adopt or keep, based on composite scores over the common window."""
    keep_name = str(champion_meta.get("model_name"))
    if champion_eval is None or not compare:
        return {
            "adopt": False,
            "model_name": keep_name,
            "reason": (
                "Champion's post-training window is too short for a fair comparison - "
                "keeping the current model (will re-evaluate as data accrues)"
            ),
        }
    best = max(compare, key=lambda c: c["composite"])
    improvement = best["composite"] - champion_eval["composite"]
    if improvement > cfg.adoption.min_improvement:
        return {
            "adopt": True,
            "model_name": best["model_name"],
            "reason": (
                f"Composite score improved by {improvement:+.4f} over the champion "
                f"on the common window "
                f"({champion_eval['composite']:.4f} -> {best['composite']:.4f})"
            ),
            "improvement": improvement,
        }
    return {
        "adopt": False,
        "model_name": keep_name,
        "reason": (
            f"Best candidate's improvement {improvement:+.4f} does not exceed the "
            f"threshold ({cfg.adoption.min_improvement}) - keeping the current model"
        ),
        "improvement": improvement,
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="stocklab weekly training")
    parser.add_argument("--config", default="config/config.yaml", help="config file path")
    parser.add_argument("--notify", action="store_true", help="notify the decision")
    args = parser.parse_args()

    setup_logging()
    try:
        return run(args.config, args.notify)
    except Exception:
        logger.exception("Weekly training pipeline terminated abnormally")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
