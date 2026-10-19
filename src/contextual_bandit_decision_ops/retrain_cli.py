from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

from .feature_store import FeatureStore
from .model_registry import ModelRegistry
from .offer_history import log_historical_decisions
from .offer_policies import EpsilonExplorationPolicy, RuleBasedOfferPolicy
from .offer_simulation import default_conversion_model
from .retraining import RetrainingConfig, RetrainingRun, retrain
from .synthetic_activity import ActivityConfig, build_feature_store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrain the offer posterior from logged feedback and promote it if it's safe"
    )
    defaults = RetrainingConfig(store_path="unused")
    parser.add_argument("--store", type=Path, default=Path("data/processed/feature_store.sqlite"))
    parser.add_argument("--model-dir", type=Path, default=Path(defaults.model_dir))
    parser.add_argument("--decision-log", type=Path, default=None)
    parser.add_argument("--feedback-log", type=Path, default=None)
    parser.add_argument("--lookback-days", type=int, default=defaults.lookback_days)
    parser.add_argument("--holdout-fraction", type=float, default=defaults.holdout_fraction)
    parser.add_argument("--min-training-events", type=int, default=defaults.min_training_events)
    parser.add_argument(
        "--max-revenue-regression", type=float, default=defaults.max_revenue_regression
    )
    parser.add_argument("--notify-url", default=None, help="Service URL to tell to reload")
    parser.add_argument(
        "--interval-minutes",
        type=float,
        default=0.0,
        help="Keep running on this schedule; 0 runs once",
    )
    parser.add_argument("--max-runs", type=int, default=0, help="Stop after this many runs")
    parser.add_argument(
        "--bootstrap-demo-store",
        action="store_true",
        help="If the store has no history yet, fill it with synthetic rule-based history",
    )
    parser.add_argument("--demo-days", type=int, default=28, help="History length for the demo")
    parser.add_argument("--report-md", type=Path, default=Path("reports/retraining_run.md"))
    return parser


def bootstrap_demo_store(path: Path, days: int = 28, seed: int = 5) -> int:
    """Build the demo history in memory, then copy it to disk in one go, which is
    much faster than writing it row by row to a file."""
    truth = default_conversion_model()
    catalog = truth.catalog
    store, population = build_feature_store(ActivityConfig(simulation_days=days))
    with store:
        logging_policy = EpsilonExplorationPolicy(RuleBasedOfferPolicy(catalog), catalog, 0.2)
        logged = log_historical_decisions(store, population, truth, logging_policy, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        target = sqlite3.connect(path)
        try:
            store.connection.backup(target)
        finally:
            target.close()
    return logged


def store_has_history(path: Path) -> bool:
    if not path.exists():
        return False
    with FeatureStore(path) as store:
        return store.table_counts()["offer_events"] > 0


def render_report(run: RetrainingRun, registry: ModelRegistry) -> str:
    lines = [
        "# Retraining Run",
        "",
        f"**Status: {run.status}.** {run.reason[:1].upper() + run.reason[1:]}.",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| New model version | {run.model_version or 'none'} |",
        f"| Previous model version | {run.previous_version or 'none'} |",
        f"| Data window | {run.window_start or 'n/a'} to {run.window_end or 'n/a'} |",
        f"| Training events | {run.train_events:,} |",
        f"| Holdout events | {run.holdout_events:,} |",
        f"| Service told to reload | {run.service_reloaded if run.service_reloaded is not None else 'no URL given'} |",
        "",
        "## Log ingestion",
        "",
        "| Decisions read | Decisions added | Feedback read | Feedback applied | Feedback without a decision |",
        "| ---: | ---: | ---: | ---: | ---: |",
        (f"| {run.ingest.decisions_read:,} | {run.ingest.decisions_inserted:,} | "
        f"{run.ingest.feedback_read:,} | {run.ingest.feedback_applied:,} | "
        f"{run.ingest.feedback_unmatched:,} |"),
    ]
    if run.estimates:
        lines.extend(
            [
                "",
                "## Holdout evaluation (doubly robust)",
                "",
                "| Policy | Conversion | Revenue / decision | Effective sample size |",
                "| --- | ---: | ---: | ---: |",
                *(
                    f"| {name} | {estimate.conversion_dr:.2%} | {estimate.revenue_dr:.3f} | "
                    f"{estimate.effective_sample_size:,.0f} |"
                    for name, estimate in run.estimates.items()
                ),
            ]
        )
    if run.checks:
        lines.extend(
            [
                "",
                "## Promotion gate",
                "",
                "A candidate only fails a comparison when the 95% interval shows it is worse by",
                "more than the allowed margin.",
                "",
                "| Check | Result | Detail |",
                "| --- | --- | --- |",
                *(
                    f"| {check.name} | {'pass' if check.passed else 'FAIL'} | {check.detail} |"
                    for check in run.checks
                ),
            ]
        )
    history = registry.history()[-5:]
    if history:
        lines.extend(
            [
                "",
                "## Recent registry entries",
                "",
                "| Recorded at | Status | Version | Reason |",
                "| --- | --- | --- | --- |",
                *(
                    f"| {entry.get('recorded_at', '')} | {entry.get('status', '')} | "
                    f"{entry.get('model_version', '')} | {entry.get('reason', '')} |"
                    for entry in history
                ),
            ]
        )
    lines.append("")
    return "\n".join(lines)


def run_once(args: argparse.Namespace) -> RetrainingRun:
    config = RetrainingConfig(
        store_path=args.store,
        model_dir=args.model_dir,
        decision_log=args.decision_log,
        feedback_log=args.feedback_log,
        lookback_days=args.lookback_days,
        holdout_fraction=args.holdout_fraction,
        min_training_events=args.min_training_events,
        max_revenue_regression=args.max_revenue_regression,
        notify_url=args.notify_url,
    )
    run = retrain(config)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(render_report(run, ModelRegistry(args.model_dir)), "utf-8")
    print(f"{run.status}: {run.reason} (model {run.model_version or run.previous_version})")
    return run


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_demo_store and not store_has_history(args.store):
        logged = bootstrap_demo_store(args.store, days=args.demo_days)
        print(f"Created {args.store} with {logged:,} logged decisions from the rule-based system")
    runs = 0
    while True:
        run_once(args)
        runs += 1
        if args.interval_minutes <= 0 or (args.max_runs and runs >= args.max_runs):
            return 0
        time.sleep(args.interval_minutes * 60.0)


if __name__ == "__main__":
    sys.exit(main())
