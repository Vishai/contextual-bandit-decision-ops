# Local and Staging Operations

## Supported environment

Release 0.1.0 is a CPU-only Python project intended for local development,
deterministic demos, and a staging-style FastAPI process. Python 3.11 is used in the
container and CI reference paths.

## Install and verify

```bash
make install
make release-check
```

`make release-check` runs unit/API tests, Ruff, the in-process service smoke test,
shell syntax checks, the metadata/version command, and release-file presence
checks. It does not require Docker.

## Reproduce the portfolio reports

```bash
make demo
```

This regenerates:

- `reports/synthetic_bandit_log_summary.md`
- `reports/baseline_policy_comparison.md`
- `reports/contextual_learning_policy_comparison.md`
- `reports/off_policy_evaluation.md`
- `reports/policy_promotion_gate.md`
- `reports/staging_observability_report.md`

The commands use fixed seeds and overwrite deterministic outputs. Runtime CSV,
JSON, and JSONL artifacts are ignored unless intentionally tracked.

## Run the staging API

```bash
contextual-bandit-service --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/health
```

Endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness and staging status |
| `GET /policy` | Policy name, actions, constraints, and environment |
| `POST /decide` | Select an action and append a decision record |
| `POST /feedback` | Validate reward and append feedback |
| `GET /metrics` | Process-local decision, action, feedback, and reward counters |

The service writes local decision and feedback JSONL logs. It has no production
authentication, durable queue, shared state, request rate limits, or privacy
controls. Delete or rotate local logs manually during development.

An external-process-free smoke path is available:

```bash
make smoke
```

## Monitoring workflow

```bash
contextual-bandit-observe \
  --output reports/staging_observability_report.md \
  --artifact artifacts/staging_observability_report.json
```

The command compares reference and current deterministic windows. Review action
total variation, reward-rate difference, standardized feature shifts, propensity
bounds, missing feedback, exploration, and API counters together. Warnings are
signals for investigation, not automated proof of an incident.

The default report deliberately contains several warnings, including action and
reward shifts, low propensities, and `25%` missing feedback. That fixture exercises
the detection path; it is not a live service health statement.

## Docker

```bash
make docker-build
make docker-smoke
```

The image installs the package in a CPU-only Python base, runs as a non-root user,
exposes port 8000, and starts the API. The smoke script builds the image, starts a
temporary container, checks `/health`, and removes the container through a shell
trap. Docker availability is optional for the normal local check.

## CI and release metadata

GitHub Actions installs the development extras, runs `pytest -q`, runs
`ruff check .`, and executes a lightweight CLI/API smoke command. Version metadata
is available through:

```bash
contextual-bandit-info
contextual-bandit-info --version
```

Before changing the version, update `pyproject.toml`, package metadata,
`CITATION.cff`, `CHANGELOG.md`, the portfolio summary, and any release references
together.

## Failure triage

1. Re-run the failing command with its fixed seed and preserve the output.
2. Check whether generated files changed or a runtime log was accidentally tracked.
3. For decision anomalies, compare policy metadata, context encoding, action
   constraints, and seed.
4. For OPE anomalies, inspect logged propensities, target support, ESS, and replay
   matches before interpreting the estimate.
5. For monitoring warnings, verify log completeness before attributing a shift to
   policy behavior.
