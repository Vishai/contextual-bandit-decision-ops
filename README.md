# Contextual Bandit Decision Ops

A small, production-style foundation for deterministic contextual bandit simulation.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
make install
make check
make demo
```

Show installed release metadata with `contextual-bandit-info`; print only the version with
`contextual-bandit-info --version`.

## Reviewer path

1. Run `make check` for the complete test and lint suite.
2. Run `make demo` to regenerate all six deterministic Markdown reports.
3. Read `reports/policy_promotion_gate.md` and
   `reports/staging_observability_report.md` for the safety/operations story.
4. Run `contextual-bandit-service --smoke-test` for the in-process API path.
5. Optionally run `make docker-smoke` when Docker is available.

## Command summary

| Command | Purpose |
| --- | --- |
| `make install` | Install the package and development tools |
| `make check` | Run pytest and Ruff |
| `make demo` | Regenerate tracked deterministic reports |
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

## Current limitations

- All evaluation, promotion, and drift evidence is synthetic or local staging data.
- The API has no production authentication, durable database, or distributed coordination.
- Monitoring is report-based rather than a live alerting system.
- The Docker image is a local demonstration target, not a cloud deployment.
- CI covers install, tests, lint, and CLI smoke behavior; it does not publish releases.
