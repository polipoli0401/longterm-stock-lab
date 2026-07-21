"""Report generation module.

Generates the daily report (top-N buy candidates + holdings risk), backtest
reports, and model-adoption-decision reports in Markdown (spec sections 12
and 14). Every score is always accompanied by its reasoning (contribution
breakdown).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from stocklab.backtest.engine import BacktestResult
from stocklab.logger import get_logger, log_event
from stocklab.risk.analyzer import RiskAssessment
from stocklab.scoring.scorer import ScoredTicker

logger = get_logger(__name__)

DISCLAIMER = (
    "This report is for informational purposes only and is not a recommendation "
    "to buy or sell any security. All investment decisions are your own responsibility."
)

_METRIC_LABELS = [
    ("cagr", "CAGR", "pct"),
    ("annual_return", "Annualized return", "pct"),
    ("excess_return", "Excess return (ann.)", "pct"),
    ("benchmark_cagr", "Benchmark CAGR", "pct"),
    ("max_drawdown", "Max drawdown", "pct"),
    ("sharpe", "Sharpe ratio", "num"),
    ("sortino", "Sortino ratio", "num"),
    ("calmar", "Calmar ratio", "num"),
    ("profit_factor", "Profit factor", "num"),
    ("win_rate", "Win rate", "pct"),
    ("n_periods", "Periods", "int"),
    ("years", "Years tested", "num"),
]


class ReportGenerator:
    """Generates and saves Markdown reports."""

    def __init__(self, report_dir: str | Path) -> None:
        """Args:
        report_dir: Output directory for reports.
        """
        self.dir = Path(report_dir)

    # ------------------------------------------------------------ daily
    def daily_markdown(
        self,
        run_date: str,
        ranked: list[ScoredTicker],
        risks: list[RiskAssessment],
        model_meta: dict[str, Any] | None,
        universe_size: int,
        analyzed: int,
        unit_shares: int = 100,
    ) -> str:
        """Generate the daily analysis report (Markdown)."""
        model_desc = _model_description(model_meta)
        lines = [
            f"# Daily Analysis Report {run_date}",
            "",
            f"- Model: {model_desc}",
            f"- Universe: {universe_size} tickers / scored: {analyzed}",
            "",
            f"## Top {len(ranked)} Buy Candidates",
            "",
        ]
        for rank, s in enumerate(ranked, start=1):
            lines.append(f"### #{rank} {s.name} ({s.ticker}) - composite score {s.score:.1f}")
            lines.append("")
            if s.price is not None:
                lot = s.price * unit_shares
                lines.append(
                    f"Price: {s.price:,.0f} JPY / min lot ({unit_shares} sh): {lot:,.0f} JPY"
                )
                lines.append("")
            lines.append("Why (contribution, in predicted excess-return %):")
            for label, value in s.contributions[:6]:
                lines.append(f"- {value:+.1f} {label}")
            lines.append("")
            lines.append("Concerns:")
            if s.concerns:
                for concern in s.concerns:
                    lines.append(f"- {concern}")
            else:
                lines.append("- No notable concerns detected")
            lines.append("")

        lines.append("## Holdings Risk")
        lines.append("")
        if risks:
            for r in sorted(risks, key=lambda x: -x.score):
                mark = "⚠️ " if r.alert else ""
                reasons = " / ".join(r.reasons) if r.reasons else "no risk flags detected"
                lines.append(f"- {mark}{r.name} ({r.ticker}) score {r.score}: {reasons}")
        else:
            lines.append("- Skipped: no holdings configured (see README)")

        lines += ["", "---", f"*{DISCLAIMER}*", ""]
        return "\n".join(lines)

    def daily_summary(
        self,
        run_date: str,
        ranked: list[ScoredTicker],
        alerts: list[RiskAssessment],
        model_name: str,
    ) -> str:
        """Generate the short summary used for notifications."""
        lines = [
            f"📈 Long-Term Investing Daily Report {run_date}",
            f"Model: {model_name}",
            "",
            f"◆ Top {len(ranked)} buy candidates",
        ]
        for rank, s in enumerate(ranked, start=1):
            tops = ", ".join(label for label, value in s.contributions[:2] if value > 0)
            suffix = f" / {tops}" if tops else ""
            price_part = f" @ {s.price:,.0f} JPY" if s.price is not None else ""
            lines.append(f"{rank}. {s.name} ({s.ticker}) score {s.score:.0f}{price_part}{suffix}")
        lines.append("")
        if alerts:
            lines.append("⚠️ Holdings risk alerts")
            for a in alerts:
                lines.append(f"- {a.name}: {' / '.join(a.reasons)} (score {a.score})")
        else:
            lines.append("Holdings: no risks above the alert threshold")
        lines += ["", DISCLAIMER]
        return "\n".join(lines)

    # --------------------------------------------------------- backtest
    def backtest_markdown(
        self, result: BacktestResult, extra: dict[str, Any] | None = None
    ) -> str:
        """Generate a backtest report (Markdown)."""
        lines = [f"# Backtest Report: {result.model_name}", ""]
        if extra:
            for key, value in extra.items():
                lines.append(f"- {key}: {value}")
            lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | ---: |")
        for key, label, kind in _METRIC_LABELS:
            lines.append(f"| {label} | {_fmt(result.metrics.get(key), kind)} |")
        lines.append("")
        lines.append("## Recent Trades (up to 10)")
        lines.append("")
        lines.append("| Entry | Exit | Tickers | Return | Benchmark |")
        lines.append("| --- | --- | --- | ---: | ---: |")
        tail = result.trades.tail(10)
        for row in tail.to_dict("records"):
            lines.append(
                "| {entry} | {exit} | {tickers} | {ret} | {bench} |".format(
                    entry=str(row["entry_date"])[:10],
                    exit=str(row["exit_date"])[:10],
                    tickers=row["tickers"] or "(none selected)",
                    ret=_fmt(row["return"], "pct"),
                    bench=_fmt(row["benchmark_return"], "pct"),
                )
            )
        lines += ["", "---", f"*{DISCLAIMER}*", ""]
        return "\n".join(lines)

    # --------------------------------------------------------- decision
    def decision_markdown(
        self,
        decision: dict[str, Any],
        candidates: list[dict[str, Any]],
        champion_eval: dict[str, Any] | None,
    ) -> str:
        """Generate the model-adoption-decision report (Markdown)."""
        lines = ["# Model Adoption Decision", ""]
        verdict = "✅ Adopt new model" if decision.get("adopt") else "⏸ Keep current model"
        lines.append(f"- Decision: {verdict} (target: {decision.get('model_name')})")
        lines.append(f"- Reason: {decision.get('reason')}")
        lines.append("")
        lines.append("## Candidate Evaluation (full walk-forward period)")
        lines.append("")
        lines.append("| Model | CAGR | Sharpe | Max DD | Excess | Composite |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for c in candidates:
            m = c["metrics"]
            lines.append(
                "| {name} | {cagr} | {sharpe} | {mdd} | {excess} | {comp} |".format(
                    name=c["model_name"],
                    cagr=_fmt(m.get("cagr"), "pct"),
                    sharpe=_fmt(m.get("sharpe"), "num"),
                    mdd=_fmt(m.get("max_drawdown"), "pct"),
                    excess=_fmt(m.get("excess_return"), "pct"),
                    comp=_fmt(c.get("composite"), "num"),
                )
            )
        lines.append("")
        if champion_eval:
            m = champion_eval["metrics"]
            lines.append("## Current Champion (evaluated on the post-training period only)")
            lines.append("")
            lines.append(
                f"- {champion_eval['model_name']}: CAGR {_fmt(m.get('cagr'), 'pct')} / "
                f"Sharpe {_fmt(m.get('sharpe'), 'num')} / "
                f"composite {_fmt(champion_eval.get('composite'), 'num')}"
            )
            lines.append("")
        lines += ["---", f"*{DISCLAIMER}*", ""]
        return "\n".join(lines)

    # ------------------------------------------------------------- save
    def save(self, text: str, relative_path: str) -> Path:
        """Save a report and return the destination path."""
        path = self.dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        log_event(logger, "Report saved", path=str(path))
        return path


def _model_description(meta: dict[str, Any] | None) -> str:
    """Build the model description shown in reports."""
    if not meta:
        return "Fallback (equal weight - no trained model)"
    return (
        f"{meta.get('model_name')} (adopted: {meta.get('adopted_at')} / "
        f"train end: {meta.get('train_end')} / target: "
        f"{meta.get('horizon_days')} trading days ahead, {meta.get('target_type')})"
    )


def _fmt(value: Any, kind: str) -> str:
    """NaN/inf-safe formatter."""
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(v):
        return "—"
    if math.isinf(v):
        return "∞"
    if kind == "pct":
        return f"{v * 100:.2f}%"
    if kind == "int":
        return f"{int(v)}"
    return f"{v:.3f}"
