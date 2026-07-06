# Contextual Bandit Decision Ops 0.1.0

## Project overview

Contextual Bandit Decision Ops is a CPU-only, deterministic reference project for
the lifecycle of contextual decisions: simulate, learn, evaluate, constrain, serve,
and monitor. It is designed for portfolio review and local/staging experimentation,
not as a claim of production policy quality.

## Main capabilities

- Typed synthetic contexts and logged bandit events with propensities
- Random, fixed, epsilon-greedy, and oracle-assisted baselines
- LinUCB and Linear Thompson Sampling with sequential updates
- Replay, IPS, SNIPS, and doubly robust off-policy evaluation
- Blocked-action, exploration, capacity, support, improvement, and regret gates
- A staging-only FastAPI decision and feedback service
- Local drift and service-log checks
- Reproducible CLI reports, Docker packaging, CI, and smoke paths

## Verification results

Release verification completed with:

- `57 passed` from `pytest -q`
- no findings from `ruff check .`
- successful `make check`
- successful in-process API smoke test
- successful deterministic demo regeneration
- shell-valid Docker build/smoke workflow

Docker engine availability is not required for the local release check.

## Evidence snapshot

| Evaluation | Selected result |
|---|---|
| Synthetic log | 100 rows, `28%` observed reward |
| Baseline comparison | Random `0.282`; epsilon-greedy `0.337`; oracle `0.341` |
| Sequential learning | LinUCB `0.325`; Thompson Sampling `0.313`; random `0.278` |
| LinUCB regret | `68.42` cumulative expected regret over 5,000 events |
| Oracle OPE | DR estimate `0.3394`, simulator value `0.3438`, ESS `1,660` |
| Promotion gate | **HOLD** |
| Observability fixture | 6 warnings, 0 failed checks |

All evidence is deterministic and synthetic. Values are useful for implementation
verification and method comparison, not real-world outcome prediction.

## Key reports

- [Synthetic log summary](../synthetic_bandit_log_summary.md)
- [Baseline policy comparison](../baseline_policy_comparison.md)
- [Contextual learning comparison](../contextual_learning_policy_comparison.md)
- [Off-policy evaluation](../off_policy_evaluation.md)
- [Policy promotion gate](../policy_promotion_gate.md)
- [Staging observability](../staging_observability_report.md)

Supporting explanations are in the
[architecture](../../docs/architecture.md),
[policy card](../../docs/policy_card.md), and
[evaluation methodology](../../docs/evaluation_methodology.md).

## Final status

**Portfolio release: ready for review. Real-policy promotion: HOLD.**

The implementation and reproducibility paths are complete for release 0.1.0. A real
launch is not justified because the data and reward mechanism are synthetic, the
default high-performing epsilon-greedy comparator is oracle-assisted, and the
service lacks production security and durability.

## Reviewer quickstart

```bash
python -m pip install -e ".[dev]"
make release-check
make demo
contextual-bandit-info
```

For the shortest review, read this summary, the policy card, and the promotion gate
report; then run `make smoke` to exercise the API in process.
