# Security Policy

## Supported Versions

Only the latest release on the `main` branch is supported with security
updates.

## Reporting a Vulnerability

Please **do not** open a public issue for security problems.

Use GitHub's private vulnerability reporting instead:
**Security tab > Report a vulnerability**. You should receive an initial
response within 7 days.

When reporting, please include:

- A description of the issue and its impact
- Steps to reproduce (or a proof of concept)
- Affected files / versions if known

## Scope Notes

- This project performs **analysis only** and never places trades, so the
  primary risks are credential leakage and supply-chain issues.
- All secrets (Discord webhook, LINE token, holdings) are injected via
  GitHub Secrets / environment variables. If you find any code path that
  could write a secret to logs, reports, or committed files, please report
  it as a vulnerability.
- Dependency vulnerabilities are monitored via Dependabot and CodeQL;
  reports for issues already flagged there are still welcome if they
  include an exploitable scenario specific to this project.
