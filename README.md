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

## Development

```bash
make check
```
