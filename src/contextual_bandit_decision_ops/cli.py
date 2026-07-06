from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import SimulationConfig
from .simulation import generate_synthetic_bandit_log


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a synthetic contextual bandit log")
    parser.add_argument("--rows", type=int, default=100, help="Number of events to simulate")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/synthetic_bandit_log.csv"),
        help="Destination CSV path",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("reports/synthetic_bandit_log_summary.md"),
        help="Destination Markdown report path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = SimulationConfig(
        n_events=args.rows,
        seed=args.seed,
        output_csv=args.output_csv,
        report_md=args.report_md,
    )
    generate_synthetic_bandit_log(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
