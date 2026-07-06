from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .comparison_report import write_comparison_outputs
from .config import PolicyComparisonConfig
from .evaluation import compare_policies


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare baseline policies on a deterministic simulator replay"
    )
    parser.add_argument("--events", type=int, default=1_000, help="Number of replay events")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument("--actions", type=int, default=3, help="Number of available actions")
    parser.add_argument("--fixed-action", type=int, default=0, help="Action for the fixed policy")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Exploration rate for the epsilon-greedy policy",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("reports/baseline_policy_comparison.md"),
        help="Destination Markdown report path",
    )
    parser.add_argument(
        "--artifact-json",
        type=Path,
        default=Path("artifacts/baseline_policy_comparison.json"),
        help="Destination JSON artifact path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PolicyComparisonConfig(
        n_events=args.events,
        seed=args.seed,
        n_actions=args.actions,
        fixed_action=args.fixed_action,
        epsilon=args.epsilon,
        report_md=args.report_md,
        artifact_json=args.artifact_json,
    )
    results = compare_policies(config)
    write_comparison_outputs(config, results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
