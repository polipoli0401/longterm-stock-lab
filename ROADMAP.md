# Roadmap

Direction of travel for stocklab. Items are grouped into phases; within a phase, order is flexible.

**Guiding principles** (these never change):

1. Decision *support*, not decision *making* — no automated order execution, ever.
2. No data leakage — every evaluation must be defensible as truly out-of-sample.
3. Explainability first — a score without a reason is a bug.

**Status legend:** ✅ shipped · 🔜 planned next · 💡 idea / needs research

---

## Phase 0 — Foundation (current)

- ✅ Daily pipeline on GitHub Actions: fetch → features → score → rank → risk → report → notify
- ✅ Purged walk-forward evaluation with leak tests
- ✅ Linear model family (Linear / Ridge / Lasso / ElasticNet) with additive contribution breakdown
- ✅ Champion/challenger adoption: adopt only on composite-score improvement over a common out-of-sample window
- ✅ Top-N rotation backtest vs. TOPIX/Nikkei benchmark (CAGR, Sharpe, Sortino, Calmar, PF, win rate, MDD, excess return)
- ✅ Holdings risk scoring with Discord/LINE alerts, secrets-based holdings (`HOLDINGS_JSON`)
- ✅ SQLite persistence, JSONL structured logging, results auto-committed

## Phase 1 — Data Quality

- 🔜 **J-Quants API integration** — true point-in-time fundamentals with real disclosure dates; replaces the conservative 90-day publication-lag approximation. Implemented as an alternative fetcher behind the existing `data/fetcher.py` interface.
- 🔜 **Quarterly fundamentals** — move from annual-only statements to quarterly, improving freshness of growth/quality factors.
- 🔜 **Transaction costs & slippage in backtests** — configurable commission (bps) and simple spread model, so reported metrics are closer to realizable returns.
- 💡 **Dividends / total-return handling** — use total-return series or add dividend adjustment to the backtest.
- 💡 **Universe hygiene** — delisting handling and survivorship-bias notes; periodic auto-refresh of `universe.csv` from an index constituent list.
- 💡 **Additional risk inputs** — earnings downward revisions and margin-trading balance (requires a paid/stable source), feeding `risk/analyzer.py`.

## Phase 2 — Modeling

- 🔜 **Gradient boosting (LightGBM)** — register in `MODEL_BUILDERS`; compared under the exact same purged walk-forward and adoption rules as the linear family.
- 🔜 **SHAP-based contributions** — keep the "reasons attached to every score" guarantee for non-linear models.
- 🔜 **Hyperparameter search inside walk-forward** — small grid/random search per fold, so tuning never sees future data.
- 💡 **Multi-horizon experiments** — systematic comparison of 30/63/126/252-day targets; possibly an ensemble across horizons.
- 💡 **Feature expansion** — accruals quality, dilution (share-count change), analyst-free quality composites; each gated by out-of-sample improvement.
- 💡 **Regime awareness** — market-state features (volatility regime, breadth) as inputs, not as discretionary switches.

## Phase 3 — Portfolio & Risk

- 🔜 **Position sizing** — inverse-volatility or equal-risk weighting as an alternative to equal weight.
- 💡 **Diversification constraints** — sector caps and simple correlation limits in candidate selection.
- 💡 **Drawdown-aware overlay** — reduce exposure rules evaluated in backtest (never auto-executed live; surfaced as guidance only).
- 💡 **Holdings integration** — "replace X with Y?" style comparisons between current holdings and top-ranked candidates, with reasoning.

## Phase 4 — Operations & UX

- 🔜 **Web dashboard** — static site (GitHub Pages) rendered from `reports/`: latest ranking, equity curves, decision history.
- 🔜 **Chart attachments in notifications** — equity-curve and score-breakdown images in Discord messages.
- 💡 **Data-quality monitors** — alerts on missing tickers, stale fundamentals, or abnormal fetch failures.
- 💡 **Weekly digest** — one summary message combining model decision, backtest deltas, and portfolio risk trend.
- 💡 **Backfill & reproducibility tooling** — one-command historical re-run with pinned data snapshots.

## Non-Goals

- ❌ Automated trade execution or broker API integration
- ❌ Intraday / high-frequency strategies
- ❌ Recommendations without an attached explanation
