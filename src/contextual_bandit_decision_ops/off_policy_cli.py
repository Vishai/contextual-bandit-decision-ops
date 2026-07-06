from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import OffPolicyEvaluationConfig
from .off_policy import run_off_policy_evaluation
from .off_policy_report import write_off_policy_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate target policies from deterministic synthetic logged data"
    )
    parser.add_argument("--events", type=int, default=5_000, help="Number of logged events")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument("--actions", type=int, default=3, help="Number of available actions")
    parser.add_argument("--fixed-action", type=int, default=0, help="Fixed target action")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Exploration rate for the epsilon-greedy target",
    )
    parser.add_argument(
        "--reward-model-regularization",
        type=float,
        default=1.0,
        help="Ridge regularization for the doubly robust reward model",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("reports/off_policy_evaluation.md"),
        help="Destination Markdown report path",
    )
    parser.add_argument(
        "--artifact-json",
        type=Path,
        default=Path("artifacts/off_policy_evaluation.json"),
        help="Destination JSON artifact path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = OffPolicyEvaluationConfig(
        n_events=args.events,
        seed=args.seed,
        n_actions=args.actions,
        fixed_action=args.fixed_action,
        epsilon=args.epsilon,
        reward_model_regularization=args.reward_model_regularization,
        report_md=args.report_md,
        artifact_json=args.artifact_json,
    )
    run = run_off_policy_evaluation(config)
    write_off_policy_outputs(config, run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
