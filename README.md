# Contextual Bandit Decision Ops

A small, production-style foundation for deterministic contextual bandit simulation.

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

## Development

```bash
make check
```
