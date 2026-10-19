from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from .offer_features import DEVICES, FEATURE_NAMES, GEOGRAPHIES, encode_contexts
from .synthetic_activity import (
    ActivityConfig,
    DecisionPoint,
    build_feature_store,
    decision_points,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the synthetic SQLite feature store and daily feature snapshots"
    )
    defaults = ActivityConfig()
    parser.add_argument("--db", type=Path, default=Path("data/processed/feature_store.sqlite"))
    parser.add_argument("--report", type=Path, default=Path("reports/feature_pipeline_summary.md"))
    parser.add_argument("--users", type=int, default=defaults.n_users)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--history-days", type=int, default=defaults.history_days)
    parser.add_argument("--simulation-days", type=int, default=defaults.simulation_days)
    return parser


def _feature_table(points: list[DecisionPoint]) -> list[str]:
    matrix = encode_contexts([point.context for point in points])
    lines = ["| Feature | Mean | Min | Max |", "| --- | ---: | ---: | ---: |"]
    for name, column in zip(FEATURE_NAMES, matrix.T, strict=True):
        lines.append(f"| `{name}` | {column.mean():.3f} | {column.min():.3f} | {column.max():.3f} |")
    return lines


def write_report(
    path: Path,
    config: ActivityConfig,
    counts: dict[str, int],
    points: list[DecisionPoint],
) -> Path:
    contexts = [point.context for point in points]
    devices = Counter(context.device for context in contexts)
    regions = Counter(context.region for context in contexts)
    cold_start = sum(point.is_new_user for point in points)
    never_purchased = sum(not context.has_purchased for context in contexts)
    lines = [
        "# Feature Pipeline Summary",
        "",
        "Synthetic raw activity is loaded into SQLite, and one point-in-time feature snapshot",
        "is built each simulated morning with the SQL in `feature_sql.py`. Every visit in the",
        "simulation window becomes a decision point that reads that morning's snapshot.",
        "",
        "## Run",
        "",
        "| Setting | Value |",
        "| --- | ---: |",
        f"| Users | {config.n_users} |",
        f"| Seed | {config.seed} |",
        f"| History window (days) | {config.history_days} |",
        f"| Simulation window (days) | {config.simulation_days} |",
        f"| Simulation start | {config.start.date().isoformat()} |",
        "",
        "## Tables",
        "",
        "| Table | Rows |",
        "| --- | ---: |",
        *(f"| `{table}` | {rows} |" for table, rows in counts.items()),
        "",
        "## Decision points",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Decision points | {len(points)} |",
        f"| Unique users | {len({context.user_id for context in contexts})} |",
        f"| Cold-start decisions (no snapshot row yet) | {cold_start} |",
        f"| Decisions for users who never purchased | {never_purchased} |",
        f"| Mean sessions in last 30 days | {np.mean([c.sessions_30d for c in contexts]):.2f} |",
        f"| Mean purchases in last 90 days | {np.mean([c.purchases_90d for c in contexts]):.2f} |",
        "",
        "| Device | Share |",
        "| --- | ---: |",
        *(f"| {device} | {devices[device] / len(points):.1%} |" for device in DEVICES),
        "",
        "| Region | Share |",
        "| --- | ---: |",
        *(f"| {region} | {regions[region] / len(points):.1%} |" for region in GEOGRAPHIES),
        "",
        "## Encoded features",
        "",
        *_feature_table(points),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ActivityConfig(
        n_users=args.users,
        seed=args.seed,
        history_days=args.history_days,
        simulation_days=args.simulation_days,
    )
    if args.db.exists():
        print(f"Rebuilding {args.db} from scratch")
    for suffix in ("", "-wal", "-shm"):
        Path(f"{args.db}{suffix}").unlink(missing_ok=True)
    store, population = build_feature_store(config, args.db)
    with store:
        points = decision_points(store, population)
        counts = store.table_counts()
    report = write_report(args.report, config, counts, points)
    print(f"Wrote feature store {args.db}")
    print(f"Wrote report {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
