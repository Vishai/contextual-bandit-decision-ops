from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .promotion_gate import PromotionGateConfig, run_promotion_gate
from .promotion_report import write_promotion_gate_outputs
from .safety import PolicyConstraints


def build_parser() -> argparse.ArgumentParser:
    defaults = PolicyConstraints()
    parser = argparse.ArgumentParser(
        description="Run deterministic safety and promotion checks for candidate policies"
    )
    parser.add_argument(
        "--simulation-events",
        type=int,
        default=5_000,
        help="Number of simulation events",
    )
    parser.add_argument(
        "--ope-events",
        type=int,
        default=5_000,
        help="Number of logged OPE events",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument("--actions", type=int, default=3, help="Number of available actions")
    parser.add_argument("--fixed-action", type=int, default=0, help="Fixed candidate action")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Exploration rate for the epsilon-greedy candidate",
    )
    parser.add_argument(
        "--blocked-action",
        type=int,
        action="append",
        default=[],
        help="Action to block; may be supplied more than once",
    )
    parser.add_argument(
        "--max-action-share",
        type=float,
        default=defaults.max_action_share,
    )
    parser.add_argument(
        "--min-exploration-rate",
        type=float,
        default=defaults.min_exploration_rate,
    )
    parser.add_argument(
        "--min-action-count",
        type=int,
        default=defaults.min_action_count,
    )
    parser.add_argument(
        "--min-ope-ess",
        type=float,
        default=defaults.min_effective_sample_size,
    )
    parser.add_argument(
        "--min-replay-matches",
        type=int,
        default=defaults.min_matched_replay_count,
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=defaults.min_estimated_improvement,
    )
    parser.add_argument(
        "--max-average-regret",
        type=float,
        default=defaults.max_average_regret,
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("reports/policy_promotion_gate.md"),
    )
    parser.add_argument(
        "--artifact-json",
        type=Path,
        default=Path("artifacts/policy_promotion_gate.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    constraints = PolicyConstraints(
        blocked_actions=frozenset(args.blocked_action),
        max_action_share=args.max_action_share,
        min_exploration_rate=args.min_exploration_rate,
        min_action_count=args.min_action_count,
        min_effective_sample_size=args.min_ope_ess,
        min_matched_replay_count=args.min_replay_matches,
        min_estimated_improvement=args.min_improvement,
        max_average_regret=args.max_average_regret,
        require_non_synthetic_evidence=True,
    )
    config = PromotionGateConfig(
        simulation_events=args.simulation_events,
        ope_events=args.ope_events,
        seed=args.seed,
        n_actions=args.actions,
        fixed_action=args.fixed_action,
        epsilon=args.epsilon,
        constraints=constraints,
        report_md=args.report_md,
        artifact_json=args.artifact_json,
    )
    run = run_promotion_gate(config)
    write_promotion_gate_outputs(config, run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
