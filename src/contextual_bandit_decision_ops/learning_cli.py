from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import LearningComparisonConfig
from .learning_evaluation import compare_learning_policies
from .learning_report import write_learning_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare contextual policies in a deterministic sequential simulation"
    )
    parser.add_argument("--events", type=int, default=5_000, help="Number of learning events")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument("--actions", type=int, default=3, help="Number of available actions")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Exploration rate for online epsilon-greedy",
    )
    parser.add_argument(
        "--linucb-alpha",
        type=float,
        default=0.5,
        help="LinUCB confidence bonus multiplier",
    )
    parser.add_argument(
        "--thompson-scale",
        type=float,
        default=0.25,
        help="Thompson posterior sampling scale",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1.0,
        help="Ridge precision used to initialize linear models",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("reports/contextual_learning_policy_comparison.md"),
        help="Destination Markdown report path",
    )
    parser.add_argument(
        "--artifact-json",
        type=Path,
        default=Path("artifacts/contextual_learning_policy_comparison.json"),
        help="Destination JSON artifact path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = LearningComparisonConfig(
        n_events=args.events,
        seed=args.seed,
        n_actions=args.actions,
        epsilon=args.epsilon,
        linucb_alpha=args.linucb_alpha,
        thompson_scale=args.thompson_scale,
        regularization=args.regularization,
        report_md=args.report_md,
        artifact_json=args.artifact_json,
    )
    results = compare_learning_policies(config)
    write_learning_outputs(config, results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
