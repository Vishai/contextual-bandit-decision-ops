from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .drift_monitoring import (
    ObservabilityConfig,
    analyze_observability,
    run_observability,
)
from .observability_report import write_observability_outputs
from .service_log_analysis import load_service_log_window


def build_parser() -> argparse.ArgumentParser:
    defaults = ObservabilityConfig()
    parser = argparse.ArgumentParser(
        description="Analyze local/staging decision and feedback drift"
    )
    parser.add_argument("--reference-events", type=int, default=1_000)
    parser.add_argument("--current-events", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--actions", type=int, default=3)
    parser.add_argument(
        "--max-action-tv",
        type=float,
        default=defaults.max_action_tv_distance,
    )
    parser.add_argument(
        "--max-reward-shift",
        type=float,
        default=defaults.max_reward_rate_difference,
    )
    parser.add_argument(
        "--min-propensity",
        type=float,
        default=defaults.minimum_healthy_propensity,
    )
    parser.add_argument(
        "--max-missing-feedback",
        type=float,
        default=defaults.max_missing_feedback_rate,
    )
    parser.add_argument(
        "--min-exploration",
        type=float,
        default=defaults.min_exploration_rate,
    )
    parser.add_argument("--reference-decisions", type=Path)
    parser.add_argument("--reference-feedback", type=Path)
    parser.add_argument("--current-decisions", type=Path)
    parser.add_argument("--current-feedback", type=Path)
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("reports/staging_observability_report.md"),
    )
    parser.add_argument(
        "--artifact-json",
        type=Path,
        default=Path("artifacts/staging_observability_report.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = ObservabilityConfig(
        reference_events=args.reference_events,
        current_events=args.current_events,
        seed=args.seed,
        n_actions=args.actions,
        max_action_tv_distance=args.max_action_tv,
        max_reward_rate_difference=args.max_reward_shift,
        minimum_healthy_propensity=args.min_propensity,
        max_missing_feedback_rate=args.max_missing_feedback,
        min_exploration_rate=args.min_exploration,
        report_md=args.report_md,
        artifact_json=args.artifact_json,
    )

    log_paths = (
        args.reference_decisions,
        args.reference_feedback,
        args.current_decisions,
        args.current_feedback,
    )
    if any(log_paths) and not all(log_paths):
        parser.error("all four reference/current decision/feedback log paths are required")
    if all(log_paths):
        reference_window = load_service_log_window(
            args.reference_decisions,
            args.reference_feedback,
        )
        current_window = load_service_log_window(
            args.current_decisions,
            args.current_feedback,
        )
        run = analyze_observability(
            config,
            reference_window,
            current_window,
            source="provided local JSONL service logs",
        )
    else:
        run = run_observability(config)

    write_observability_outputs(config, run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
