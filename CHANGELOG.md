# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-07-16

### Added

- Daily analysis pipeline (`main.py`): price/fundamental ingestion via
  yfinance, cross-sectionally standardized features, champion-model scoring
  with per-feature contribution breakdowns, top-N buy-candidate ranking,
  holdings risk alerts, Markdown reports, and Discord/LINE notifications.
- Weekly training pipeline (`train.py`): purged walk-forward evaluation of
  linear-family candidates (Linear/Ridge/Lasso/ElasticNet), composite-score
  adoption decision against a frozen champion on a common comparison
  window, decision log, and champion registry with archiving.
- Standalone backtester (`backtest.py`) with holding-period / top-N
  overrides and equity-curve CSV export.
- SQLite persistence for prices, fundamentals, rankings, and run logs.
- GitHub Actions workflows: CI (ruff + pytest), daily run (weekdays
  18:30 JST), weekly training (Sundays 06:00 JST), with results committed
  back to the repository.
- Secrets-only handling of Discord webhook, LINE credentials, and holdings
  (`HOLDINGS_JSON`).
- Test suite (26 tests) covering metrics, purged walk-forward splitting,
  feature/target construction, scoring explainability, and the backtest
  engine.
