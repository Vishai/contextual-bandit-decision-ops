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

## Development

```bash
make check
```
