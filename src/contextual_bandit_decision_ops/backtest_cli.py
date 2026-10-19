from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .offer_backtest import BASELINE, BacktestConfig, BacktestRun, run_backtest
from .synthetic_activity import ActivityConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backtest offer policies on historical logs with offline policy evaluation"
    )
    defaults = BacktestConfig()
    parser.add_argument("--users", type=int, default=defaults.activity.n_users)
    parser.add_argument("--days", type=int, default=defaults.activity.simulation_days)
    parser.add_argument("--train-days", type=int, default=defaults.train_days)
    parser.add_argument("--logging-epsilon", type=float, default=defaults.logging_epsilon)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--bootstrap-samples", type=int, default=defaults.bootstrap_samples)
    parser.add_argument("--report-md", type=Path, default=Path("reports/offer_backtest.md"))
    parser.add_argument(
        "--artifact-json", type=Path, default=Path("artifacts/offer_backtest.json")
    )
    return parser


def _interval(bounds: tuple[float, float] | None, percent: bool) -> str:
    if bounds is None:
        return "baseline"
    low, high = bounds
    if percent:
        return f"{low:+.1%} to {high:+.1%}"
    return f"{low:+.2f} to {high:+.2f}"


def render_report(run: BacktestRun) -> str:
    config = run.config
    baseline = run.offline[BASELINE]
    forward_baseline = run.forward[BASELINE]
    lines = [
        "# Offer Backtest",
        "",
        f"The old production system (`{run.logging_policy}`: rule-based targeting with",
        (f"{config.logging_epsilon:.0%} uniform exploration) is replayed over "
        f"{config.activity.simulation_days} days of"),
        "synthetic traffic, and every decision goes into the SQLite `offer_events` table with",
        "its propensity. Policies are trained on the first",
        f"{config.train_days} days of that history and evaluated on the rest.",
        "",
        "Outcomes come from a synthetic ground truth model. That is what lets this report",
        "check each offline estimate against the true value, but it also means the numbers",
        "describe the simulator, not a live business.",
        "",
        "## Historical data",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Training events | {run.train_events:,} |",
        f"| Held-out events | {run.test_events:,} |",
        f"| Held-out users | {run.test_users:,} |",
        f"| Logged conversion rate (all history) | {run.logged_conversion_rate:.2%} |",
        f"| Logged revenue per decision (all history) | {run.logged_revenue_per_decision:.3f} |",
        "",
        "## Offline policy evaluation on the held-out window",
        "",
        "Each trained policy is frozen and scored on held-out logs it never saw.",
        "",
        ("| Policy | Conversion IPS | Conversion SNIPS | Conversion DR | True conversion |"
        " Revenue/decision SNIPS | Revenue/decision DR | True revenue/decision | ESS |"),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, estimate in run.offline.items():
        lines.append(
            f"| {name} | {estimate.conversion_ips:.2%} | {estimate.conversion_snips:.2%} | "
            f"{estimate.conversion_dr:.2%} | {estimate.true_conversion:.2%} | "
            f"{estimate.revenue_snips:.3f} | {estimate.revenue_dr:.3f} | "
            f"{estimate.true_revenue_per_decision:.3f} | {estimate.effective_sample_size:,.0f} |"
        )
    lines.extend(
        [
            "",
            "### Lift over the rule-based strategy (doubly robust)",
            "",
            "Intervals are 95% paired bootstrap intervals over held-out events.",
            "",
            "| Policy | Conversion change (pts) | 95% interval | Revenue / user | Lift | 95% interval |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, estimate in run.offline.items():
        conversion_change = (estimate.conversion_dr - baseline.conversion_dr) * 100.0
        revenue_lift = estimate.revenue_dr / baseline.revenue_dr - 1.0
        lines.append(
            f"| {name} | {conversion_change:+.2f} | "
            f"{_interval(estimate.conversion_lift_pts_ci, percent=False)} | "
            f"{estimate.revenue_per_user_dr:.2f} | {revenue_lift:+.1%} | "
            f"{_interval(estimate.revenue_lift_ci, percent=True)} |"
        )
    lines.extend(
        [
            "",
            "## Forward simulation after warm start",
            "",
            "Each policy is trained on the same history, then run on the held-out visits while",
            "it keeps learning from every outcome. Outcomes come from the true model.",
            "",
            ("| Policy | Conversion | Change vs rules (pts) | Revenue / user | Lift vs rules |"
            " Cumulative regret |"),
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, result in run.forward.items():
        conversion_change = (result.conversion_rate - forward_baseline.conversion_rate) * 100.0
        lift = result.revenue_per_user / forward_baseline.revenue_per_user - 1.0
        lines.append(
            f"| {name} | {result.conversion_rate:.2%} | {conversion_change:+.2f} | "
            f"{result.revenue_per_user:.2f} | {lift:+.1%} | {result.cumulative_regret:,.0f} |"
        )
    lines.extend(
        [
            "",
            "## Replay of learning from scratch",
            "",
            "Every learner starts empty and walks the full history in order. An event counts",
            "only when the learner picks the logged offer and passes a propensity-based",
            "acceptance draw, so the kept events behave like uniformly logged data. It's",
            "unbiased but uses only a few percent of the logs.",
            "",
            "| Policy | Accepted events | Conversion | Revenue / decision |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, replay in run.replay.items():
        lines.append(
            f"| {name} | {replay.accepted_events:,} | {replay.conversion_rate:.2%} | "
            f"{replay.revenue_per_decision:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Reading the results",
            "",
            "- IPS and SNIPS only use logged outcomes and propensities. Doubly robust adds a",
            "  logistic reward model fit on the training window, which cuts variance.",
            "- ESS is the effective sample size of the importance weights. Policies that",
            "  disagree with the rules a lot only match a small slice of the logs, so their",
            "  intervals are wide.",
            "- Revenue per user is revenue per decision times the held-out decisions per user.",
            "- The replay starts every learner cold on about 4% of the logs, which is why",
            "  model-based learners trail the rules there. The warm-started runs above show",
            "  how they do with the full history.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BacktestConfig(
        activity=ActivityConfig(n_users=args.users, simulation_days=args.days),
        train_days=args.train_days,
        logging_epsilon=args.logging_epsilon,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    run = run_backtest(config)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(render_report(run), encoding="utf-8")
    args.artifact_json.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(run)
    payload["config"]["activity"]["start"] = config.activity.start.isoformat()
    args.artifact_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {args.report_md}")
    print(f"Wrote {args.artifact_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
