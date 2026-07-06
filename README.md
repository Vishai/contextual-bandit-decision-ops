# Contextual Bandit Decision Ops

An end-to-end, deterministic reference project for contextual bandit decisions:
simulate, learn, evaluate logged policies, apply safety gates, serve decisions, and
monitor local/staging behavior.

**Release 0.1.0:** portfolio-ready for local review; **HOLD** for real-world policy
promotion. All reward evidence is synthetic, and the API is deliberately
staging-only.

## What this demonstrates

- Reproducible comparison of baselines, LinUCB, and Linear Thompson Sampling
- Propensity-aware replay, IPS, SNIPS, and doubly robust evaluation
- Promotion decisions that include exploration, coverage, capacity, support, and
  regret—not reward alone
- A typed FastAPI decision/feedback contract with local observability
- CPU-only Docker, CI, release metadata, and one-command quality/demo paths

Start with the [0.1.0 release summary](reports/portfolio/release_0.1.0.md), then
read the [policy card](docs/policy_card.md) for the evidence and launch decision.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
make install
make release-check
make demo
```

Show installed release metadata with `contextual-bandit-info`; print only the version with
`contextual-bandit-info --version`.

## Reviewer path

1. Read the [portfolio release summary](reports/portfolio/release_0.1.0.md).
2. Run `make release-check` for tests, lint, metadata, API smoke, and release files.
3. Run `make demo` to regenerate all six deterministic Markdown reports.
4. Read [the promotion gate](reports/policy_promotion_gate.md) and
   [observability report](reports/staging_observability_report.md).
5. Optionally run `make docker-smoke` when Docker is available.

## Command summary

| Command | Purpose |
| --- | --- |
| `make install` | Install the package and development tools |
| `make check` | Run pytest and Ruff |
| `make smoke` | Exercise the API entirely in process |
| `make demo` | Regenerate tracked deterministic reports |
| `make release-check` | Run lightweight release verification without Docker |
| `make docker-build` | Build the CPU-only local service image |
| `make docker-smoke` | Build, run, health-check, and clean up the image |
| `contextual-bandit-info` | Print package/release metadata |
| `contextual-bandit-service --smoke-test` | Exercise the API entirely in process |

## Generate a synthetic log

```bash
contextual-bandit-generate
```

The command uses a fixed default seed and writes:

- `data/processed/synthetic_bandit_log.csv`
- `reports/synthetic_bandit_log_summary.md`

Use `contextual-bandit-generate --help` to change the row count, seed, or output paths.

## Compare baseline policies

```bash
contextual-bandit-compare
```

This replays a shared deterministic simulation stream across random-uniform, fixed-action,
greedy-oracle, and configurable epsilon-greedy baselines. It writes:

- `reports/baseline_policy_comparison.md`
- `artifacts/baseline_policy_comparison.json`

The comparison is an offline simulator benchmark, not real online learning or causal
off-policy evaluation. Use `contextual-bandit-compare --help` to change its configuration.

## Compare contextual learning policies

```bash
contextual-bandit-learn
```

This runs a deterministic sequential simulation comparing random-uniform, online
epsilon-greedy, LinUCB, linear Thompson Sampling, and a greedy-oracle upper bound. It writes:

- `reports/contextual_learning_policy_comparison.md`
- `artifacts/contextual_learning_policy_comparison.json`

LinUCB and Thompson Sampling update from each observed reward. This is still an online-style
simulation, not a production online-learning service. Use `contextual-bandit-learn --help`
to change its configuration.

## Evaluate policies from logged data

```bash
contextual-bandit-ope
```

This generates deterministic uniform-policy logs and compares target policies using the
direct logged average, replay matching, IPS, SNIPS, and doubly robust estimators. It writes:

- `reports/off_policy_evaluation.md`
- `artifacts/off_policy_evaluation.json`

The report includes matching and effective-sample-size diagnostics plus simulator truth as a
synthetic reference. It is not evidence of production policy quality. Use
`contextual-bandit-ope --help` to change its configuration.

## Run the policy promotion gate

```bash
contextual-bandit-gate
```

This applies blocked-action, action-share, exploration, coverage, OPE support, improvement,
and regret checks to candidate policies. It writes:

- `reports/policy_promotion_gate.md`
- `artifacts/policy_promotion_gate.json`

The default decision is deliberately conservative: synthetic evidence alone cannot authorize
promotion. This is simulated launch-readiness logic, not real-world safety certification. Use
`contextual-bandit-gate --help` to inspect configurable thresholds.

## Run the local decision API

```bash
contextual-bandit-service --host 127.0.0.1 --port 8000
```

The local/staging-only FastAPI service exposes `GET /health`, `GET /policy`, `POST /decide`,
`POST /feedback`, and `GET /metrics`. Decisions and feedback are appended to ignored local
JSONL files under `logs/`.

Run the deterministic API smoke path entirely in process:

```bash
contextual-bandit-service --smoke-test
```

This serving layer has no production authentication, deployment, or monitoring guarantees.

## Generate a staging observability report

```bash
contextual-bandit-observe
```

This compares deterministic reference/current decision and feedback windows for action,
reward, context, propensity, feedback-coverage, and exploration changes. It writes:

- `reports/staging_observability_report.md`
- `artifacts/staging_observability_report.json`

Pass four `--reference-*` and `--current-*` JSONL paths to analyze local service logs instead
of fixtures. The checks are transparent local/staging diagnostics, not production monitoring.

## Docker

```bash
make docker-build
make docker-smoke
```

The image uses Python 3.11 slim, installs only CPU dependencies, runs as an unprivileged user,
exposes port 8000, and includes a `/health` check. Docker is optional for local quality checks.

## Development

```bash
make check
```

## Documentation

| Document | Reviewer question |
| --- | --- |
| [Architecture](docs/architecture.md) | How do simulation, evaluation, serving, and monitoring connect? |
| [Policy card](docs/policy_card.md) | What is intended, measured, constrained, and approved? |
| [Evaluation methodology](docs/evaluation_methodology.md) | What do the estimators and regret metrics mean? |
| [Operations](docs/operations.md) | How is the local/staging system run and diagnosed? |
| [Release checklist](docs/release_checklist.md) | What is verified, and what is intentionally incomplete? |
| [Interview notes](docs/interview_notes.md) | What are the main design choices and tradeoffs? |

## Current limitations

- All evaluation, promotion, and drift evidence is synthetic or local staging data.
- The API has no production authentication, durable database, or distributed coordination.
- Monitoring is report-based rather than a live alerting system.
- The Docker image is a local demonstration target, not a cloud deployment.
- CI covers install, tests, lint, and CLI smoke behavior; it does not publish releases.
