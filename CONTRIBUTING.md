# Contributing

Thanks for your interest in improving this project! Small fixes and focused
improvements are very welcome.

## Development Setup

```bash
git clone https://github.com/<owner>/longterm-stock-lab.git
cd longterm-stock-lab
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before Opening a PR

Run the same checks as CI:

```bash
ruff check .
pytest -q
```

Both must pass. Style rules (line length 100, import order, etc.) are
enforced by ruff and configured in `pyproject.toml`.

## Guidelines

- Keep PRs small and focused on a single change.
- Add or update tests for any behavior change (`tests/`).
- Do not hard-code tunable values; add them to `config/config.yaml`.
- Never commit secrets. Webhooks, tokens, and holdings belong in
  environment variables / GitHub Secrets (see `.env.example`).
- Docstrings, comments, logs, and report strings are written in English.
- No look-ahead: any change to features, targets, or backtesting must
  preserve the leak-prevention design (purged walk-forward, publication
  lag, point-in-time joins). Please explain your reasoning in the PR.

## Commit Messages

Conventional-commit style is appreciated but not required, e.g.
`feat: add LightGBM candidate model` or `fix: handle empty benchmark`.

## Reporting Bugs / Ideas

Open an issue with reproduction steps or a clear motivation. For security
problems, follow [SECURITY.md](SECURITY.md) instead of a public issue.
