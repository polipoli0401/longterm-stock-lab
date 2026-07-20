# stocklab — Long-Term Stock Investment Support System

An automated analysis system that supports **stock-selection decisions** for long-term investing in Japanese equities. It runs daily on GitHub Actions and delivers a machine-learning-based ranking of buy candidates plus a risk analysis of your current holdings — always with the reasoning attached, as reports and notifications.

> ⚠️ This system is for informational purposes only. It never executes trades automatically, and nothing it outputs is a recommendation to buy or sell any security. All investment decisions are your own responsibility.

## Key Features

- **Daily analysis (weekdays 18:30 JST)** — Fetches price and fundamental data, then scores and ranks the Top 3 buy candidates. Every pick comes with an additive breakdown of *which factors contributed how much* plus a list of concerns, so the model is never a black box.
- **Holdings risk analysis** — Detects price below the 200-day moving average, volatility spikes, drying-up volume, large drawdowns from the 52-week high, and upcoming earnings. Sends an alert when the risk score crosses a threshold.
- **Weekly training (Sundays 06:00 JST)** — Evaluates multiple models (Linear / Ridge / Lasso / ElasticNet) out-of-sample with purged walk-forward analysis, and **adopts a new model only if it beats the current champion**. Every adopt/keep decision is logged with its reason to `models/decisions.jsonl`.
- **Backtesting** — Evaluates a Top-3 rotation strategy with CAGR, Sharpe, Sortino, Calmar, Profit Factor, win rate, max drawdown, and excess return vs. a benchmark (TOPIX ETF / Nikkei 225).
- **Notifications** — Discord and LINE. Channels that are not configured are skipped automatically.
- **Structured logging** — Every step is recorded as JSON Lines with a per-run `run_id` for end-to-end traceability.

## Architecture

```
GitHub Actions (scheduled)
│
├─ daily.yml (weekdays 18:30 JST) ──► main.py
│    fetch data → build features → score → rank
│    → holdings risk analysis → report → Discord/LINE
│
├─ weekly_train.yml (Sundays 06:00 JST) ──► train.py
│    purged walk-forward training → backtest → composite score
│    → update champion only on improvement → decision report
│
└─ ci.yml (push/PR) ──► ruff + pytest

Outputs (data/ reports/ models/) are persisted via auto-commit.
```

| Module | Responsibility |
| --- | --- |
| `stocklab/data` | Price & fundamentals via yfinance, SQLite persistence |
| `stocklab/features` | Model features, technical filters, target variable |
| `stocklab/models` | Walk-forward training, model registry & adoption decisions |
| `stocklab/scoring` | Scoring with additive contribution breakdown (explainability) |
| `stocklab/backtest` | Backtest engine and performance metrics |
| `stocklab/risk` | Risk scoring of current holdings |
| `stocklab/report` / `notify` | Markdown report generation, Discord/LINE delivery |

## Directory Layout

```
.
├── main.py / train.py / backtest.py   # entry points
├── config/
│   ├── config.yaml          # all tunable parameters (non-secret)
│   ├── universe.csv         # analysis universe (ticker,name)
│   └── holdings.example.csv # sample holdings file (never commit the real one)
├── src/stocklab/            # the package
├── tests/                   # pytest suite (incl. leak checks)
├── .github/workflows/       # ci / daily / weekly_train
├── data/                    # SQLite DB (auto-committed)
├── reports/                 # daily / backtest / model_decision
├── models/                  # champion model + decision history
└── logs/                    # JSONL logs (uploaded as artifacts on Actions)
```

## Setup

### 1. Create the repository

Push this project to a GitHub repository.

### 2. Register Secrets (important)

Go to **Settings → Secrets and variables → Actions → New repository secret**. URLs, tokens, and your holdings must never be written into code or config files — always manage them as Secrets.

| Secret name | Contents | Required |
| --- | --- | --- |
| `DISCORD_WEBHOOK_URL` | Discord webhook URL | Optional (for notifications) |
| `LINE_CHANNEL_ACCESS_TOKEN` | LINE Messaging API channel access token | Optional (for notifications) |
| `LINE_TO` | LINE recipient user ID (`U...`) | Required if using LINE |
| `HOLDINGS_JSON` | Your holdings as JSON (example below) | Optional (for risk analysis) |

Example value for `HOLDINGS_JSON` (register as a single line):

```json
[{"ticker":"7203.T","name":"Toyota Motor","quantity":100,"avg_price":2500},{"ticker":"8306.T","name":"MUFG","quantity":200,"avg_price":1200}]
```

Holdings are personal financial information, so they are kept in Secrets even for public repositories. For local-only use you may instead put them in `config/holdings.csv` (already in `.gitignore`).

### 3. Grant Actions write permission

Set **Settings → Actions → General → Workflow permissions** to **Read and write permissions** (required for auto-committing results).

### 4. First run

1. Manually run **Actions → Weekly Train & Backtest → Run workflow** (initial model training; expect ~30 minutes since it fetches ~10 years of data).
2. Manually run **Actions → Daily Analysis → Run workflow** and check the report and notifications.

After that, everything runs on schedule. If the daily job runs before any model has been trained, it falls back to an equal-weight score — clearly labeled as such in the report.

### Local usage

```bash
pip install -e ".[dev]"
cp .env.example .env   # edit as needed and export as environment variables
python train.py        # initial training
python main.py --no-notify
pytest                 # run the test suite
```

## Configuration (config/config.yaml)

| Key | Default | Description |
| --- | --- | --- |
| `data.universe_file` | config/universe.csv | List of tickers to analyze |
| `data.benchmark` | 1306.T | Benchmark (TOPIX ETF; falls back to `benchmark_alt` = ^N225) |
| `data.publication_lag_days` | 90 | Publication lag applied to fundamentals (leak prevention) |
| `model.horizon_days` | 126 | Target look-ahead in trading days (configurable, ~30–365 calendar days) |
| `model.target_type` | excess | `excess` (vs. benchmark) or `absolute` |
| `model.candidates` | ridge, etc. (4) | Models compared in the weekly run |
| `backtest.top_n` / `holding_days` | 3 / 126 | Portfolio size and holding period |
| `risk.*` | — | Risk detection thresholds, weights, alert threshold |
| `adoption.*` | — | Composite-score weights and minimum improvement for adoption |

Generated reports, notifications, and logs are written in English. To localize them, edit the label constants in `src/stocklab/features/builder.py`, `src/stocklab/risk/analyzer.py`, and the templates in `src/stocklab/report/generator.py`.

## Data-Leak Prevention

Treated as the most critical requirement of the design document:

1. **Strict chronological order** — All training and evaluation respect time order; no random shuffling anywhere.
2. **Purged walk-forward** — The target is "the return over the next *h* trading days from day *t*", so for a test window starting at index *i*, training data is restricted to `dates[i - h]` and earlier. This removes label overlap with the test period and is verified by `tests/test_walk_forward.py`.
3. **Publication lag on fundamentals** — yfinance provides no true disclosure dates, so each fiscal period becomes usable only after `fiscal_end + 90 days` (configurable) — a conservative point-in-time approximation.
4. **Unrealized targets excluded** — Targets within the most recent horizon are NaN and never used for training.
5. **Cross-sectional standardization** — Feature z-scores are computed within each date's cross-section only; no time-direction statistics are used.

## Model Adoption Logic

```
composite = 0.4×Sharpe + 0.2×CAGR + 0.3×excess return − 0.1×|max drawdown|
```

- The current champion is evaluated **frozen** (no retraining) on the period *after* its training-data end date; candidate models are re-aggregated over the same window, giving a truly out-of-sample, apples-to-apples comparison.
- If the comparison window has too few trades, the current model is kept (fail-safe).
- If the feature set or target definition changed, the newly trained model is adopted.
- Every decision is recorded with its reason in `models/decisions.jsonl` and `reports/model_decision/`.

## Design Decisions & Improvements over the Spec

| Decision | Rationale | Effect | Trade-off / mitigation |
| --- | --- | --- | --- |
| Target = excess return vs. benchmark | Learn stock-picking skill, not market direction | Rankings robust to market regime | Set `target_type: absolute` if absolute return matters more |
| Earnings yield, B/P, FCF yield instead of PER/PBR | Inverting avoids division-by-zero and wild values for loss-making firms | Robust to outliers | Read as "higher = cheaper" |
| Cross-sectional z-score + ±3σ winsorize | Linear models are sensitive to scale and outliers | Coefficients become interpretable contributions | Discards some distribution info (acceptable) |
| Technical indicators are filters only (not learned) | Spec §5 requirement; also reduces overfitting | Excludes overheated names | If all fail, filled back in labeled "reference only" |
| 90-day publication lag | yfinance lacks disclosure dates | Eliminates fundamental leakage | Freshness suffers (solved by J-Quants later) |
| Holdings via `HOLDINGS_JSON` secret | Prevents leaking personal assets in a public repo | Repo can be public | Update via Secrets editing |
| Persist results by git commit | Actions filesystems are ephemeral | Free persistence of DB/reports/models | History growth (squash periodically if needed) |

## Limitations

- **yfinance constraints** — Only 4–5 annual fiscal periods, no disclosure dates, and rate limits. For production-grade use, switch to a point-in-time source such as the **J-Quants API** (the `data/fetcher.py` interface is designed to be swappable).
- Downward revisions and margin-trading balance are not implemented (no stable free source); add them in `risk/analyzer.py` once a data source exists.
- The backtest is simplified: no transaction costs, slippage, or dividends.

See [ROADMAP.md](ROADMAP.md) for planned improvements.

## Development

```bash
ruff check . && ruff format .   # lint / format
pytest                          # tests (leak checks, known-value metric checks)
```

## Acknowledgements

Developed with AI assistance (Claude by Anthropic).
Design decisions, verification, and operation
by the repository owner.

---

*Nothing in this repository constitutes investment advice. Investing involves risk.*
