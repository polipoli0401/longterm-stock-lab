"""Model registry module.

Handles saving/loading/archiving the champion model (the currently adopted
model), computing the composite adoption score, and recording adopt/keep
decisions. Every decision is appended with its reason to
``models/decisions.jsonl`` to keep the process transparent
(spec sections 9 and 13).
"""

from __future__ import annotations

import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from stocklab.config import AdoptionConfig
from stocklab.logger import get_logger, log_event

logger = get_logger(__name__)


class ModelRegistry:
    """Persists the champion model and evaluates adoption decisions."""

    CHAMPION_META = "champion.json"
    CHAMPION_MODEL = "champion_model.joblib"
    DECISION_LOG = "decisions.jsonl"

    def __init__(self, model_dir: str | Path) -> None:
        """Args:
        model_dir: Directory used for model storage.
        """
        self.dir = Path(model_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------- load
    def load_champion(self) -> tuple[Any | None, dict[str, Any] | None]:
        """Load the champion model and its metadata; (None, None) if absent."""
        meta_path = self.dir / self.CHAMPION_META
        model_path = self.dir / self.CHAMPION_MODEL
        if not meta_path.exists() or not model_path.exists():
            return None, None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            model = joblib.load(model_path)
            return model, meta
        except Exception:
            logger.exception("Failed to load the champion model")
            return None, None

    # ----------------------------------------------------------- save
    def save_champion(self, model: Any, meta: dict[str, Any]) -> None:
        """Save a new champion; the previous one is moved to the archive."""
        if (self.dir / self.CHAMPION_META).exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive = self.dir / "archive" / stamp
            archive.mkdir(parents=True, exist_ok=True)
            for fname in (self.CHAMPION_META, self.CHAMPION_MODEL):
                src = self.dir / fname
                if src.exists():
                    shutil.copy2(src, archive / fname)
        joblib.dump(model, self.dir / self.CHAMPION_MODEL)
        (self.dir / self.CHAMPION_META).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        log_event(logger, "Champion model updated", model=meta.get("model_name"))

    # ------------------------------------------------------- decision
    def composite_score(self, metrics: dict[str, float], cfg: AdoptionConfig) -> float:
        """Compute the composite score used for adoption decisions.

        composite = w_sharpe*Sharpe + w_cagr*CAGR + w_excess*excess return
                    - w_mdd*|max drawdown|

        Returns ``-inf`` when any input metric is missing (NaN/inf), which
        removes the candidate from consideration.
        """
        values: dict[str, float] = {}
        for key in ("sharpe", "cagr", "excess_return", "max_drawdown"):
            v = metrics.get(key)
            if v is None or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                return float("-inf")
            values[key] = float(v)
        return (
            cfg.w_sharpe * values["sharpe"]
            + cfg.w_cagr * values["cagr"]
            + cfg.w_excess * values["excess_return"]
            - cfg.w_mdd * abs(values["max_drawdown"])
        )

    def record_decision(self, decision: dict[str, Any]) -> None:
        """Append the adopt/keep decision to the append-only log."""
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            **decision,
        }
        with open(self.dir / self.DECISION_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        log_event(
            logger,
            "Model adoption decision recorded",
            adopt=decision.get("adopt"),
            model=decision.get("model_name"),
            reason=decision.get("reason"),
        )
