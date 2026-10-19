from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .offer_scenarios import (
    PolicyFactory,
    Scenario,
    ScenarioPolicyResult,
    cold_start_scenario,
    default_factories,
    delayed_rewards_scenario,
    run_scenario,
    shifting_rates_scenario,
    thompson_factory,
)
from .synthetic_activity import ActivityConfig, DecisionPoint, build_feature_store, decision_points

DELAY_SWEEP_HOURS = (0.0, 6.0, 24.0, 72.0)
FORGETTING_SWEEP = (1.0, 0.9998, 0.9995, 0.999)


@dataclass(frozen=True)
class SweepRow:
    label: str
    runs: int
    total_regret: float
    total_regret_std: float
    regret_after_change: float
    regret_last_quarter: float
    revenue_per_user: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run cold-start, delayed-reward and drift simulations for the offer bandit"
    )
    defaults = ActivityConfig()
    parser.add_argument("--users", type=int, default=defaults.n_users)
    parser.add_argument("--simulation-days", type=int, default=defaults.simulation_days)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--delay-hours", type=float, default=24.0)
    parser.add_argument("--sweep-seeds", type=int, default=3, help="Runs averaged per sweep row")
    parser.add_argument("--skip-sweeps", action="store_true", help="Skip the delay/forgetting sweeps")
    parser.add_argument("--report-md", type=Path, default=Path("reports/offer_scenarios.md"))
    parser.add_argument(
        "--artifact-json", type=Path, default=Path("artifacts/offer_scenarios.json")
    )
    return parser


def sweep_row(
    label: str,
    scenario: Scenario,
    factory: PolicyFactory,
    points: Sequence[DecisionPoint],
    seeds: Sequence[int],
) -> SweepRow:
    runs = [next(iter(run_scenario(scenario, [factory], points, seed=s).values())) for s in seeds]
    totals = np.array([run.cumulative_regret for run in runs])
    return SweepRow(
        label=label,
        runs=len(runs),
        total_regret=float(totals.mean()),
        total_regret_std=float(totals.std(ddof=1)) if len(runs) > 1 else 0.0,
        regret_after_change=float(np.mean([run.regret_after_change for run in runs])),
        regret_last_quarter=float(np.mean([run.regret_last_quarter for run in runs])),
        revenue_per_user=float(np.mean([run.revenue_per_user for run in runs])),
    )


def _policy_table(results: Mapping[str, ScenarioPolicyResult]) -> list[str]:
    lines = [
        (
            "| Policy | Conversion | Revenue / user | Total regret | Regret before change |"
            " Regret after change | Last-quarter regret |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result.conversion_rate:.2%} | {result.revenue_per_user:.2f} | "
            f"{result.cumulative_regret:,.0f} | {result.regret_before_change:,.0f} | "
            f"{result.regret_after_change:,.0f} | {result.regret_last_quarter:,.0f} |"
        )
    return lines


def _cold_start_section(results: Mapping[str, ScenarioPolicyResult]) -> list[str]:
    oracle_share = next(iter(results.values())).oracle_launch_offer_share_last_quarter or 0.0
    lines = [
        "",
        f"The oracle would send {oracle_share:.1%} of last-quarter traffic to the new offer.",
        "",
        (
            "| Policy | New-offer share (last quarter) | Regret per decision, users under 7 days |"
            " Regret per decision, returning users |"
        ),
        "| --- | ---: | ---: | ---: |",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result.launch_offer_share_last_quarter or 0.0:.1%} | "
            f"{result.new_user_regret_per_decision:.3f} | "
            f"{result.returning_user_regret_per_decision:.3f} |"
        )
    return lines


def _sweep_table(title: str, header: str, rows: Sequence[SweepRow]) -> list[str]:
    runs = rows[0].runs if rows else 0
    lines = [
        "",
        f"### {title}",
        "",
        f"Thompson Sampling only, averaged over {runs} seeds. The spread column is the standard",
        "deviation of total regret between seeds.",
        "",
        (f"| {header} | Total regret | Spread | Regret after change | Last-quarter regret |"
        " Revenue / user |"),
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.label} | {row.total_regret:,.0f} | {row.total_regret_std:,.0f} | "
            f"{row.regret_after_change:,.0f} | {row.regret_last_quarter:,.0f} | "
            f"{row.revenue_per_user:.2f} |"
        )
    return lines


def _takeaways(
    sections: Sequence[tuple[Scenario, Mapping[str, ScenarioPolicyResult]]],
    delay_sweep: Sequence[SweepRow],
    forgetting_sweep: Sequence[SweepRow],
) -> list[str]:
    lines = ["", "## Takeaways", ""]
    by_name = {scenario.name: results for scenario, results in sections}

    cold = by_name.get("cold_start_offer_launch")
    if cold and "thompson_sampling" in cold:
        ts = cold["thompson_sampling"]
        others = ", ".join(
            f"{name} {result.launch_offer_share_last_quarter or 0.0:.0%}"
            for name, result in cold.items()
            if name != "thompson_sampling"
        )
        lines.append(
            f"- Cold start: Thompson Sampling sent {ts.launch_offer_share_last_quarter or 0.0:.0%}"
            f" of last-quarter traffic to the new offer against an oracle share of "
            f"{ts.oracle_launch_offer_share_last_quarter or 0.0:.0%} ({others}). A wide "
            "posterior on an unseen offer is what gets it tried; greedy has no reason to, "
            "and the rules don't know the offer exists."
        )

    delayed = by_name.get("delayed_rewards")
    if delayed and "rule_based" in delayed and "thompson_sampling" in delayed:
        ts, rules = delayed["thompson_sampling"], delayed["rule_based"]
        line = (
            f"- Delayed rewards: with outcomes arriving late, Thompson Sampling finished with "
            f"{ts.cumulative_regret:,.0f} regret against {rules.cumulative_regret:,.0f} for "
            "the rules."
        )
        if len(delay_sweep) >= 2:
            first, last = delay_sweep[0], delay_sweep[-1]
            change = last.total_regret / first.total_regret - 1.0
            line += (
                f" Going from {first.label} to {last.label} of average delay changed its "
                f"regret by {change:+.0%} in the sweep."
            )
        lines.append(line)

    drift = by_name.get("shifting_conversion_rates")
    if drift and "rule_based" in drift:
        rules = drift["rule_based"]
        best = min(
            (r for n, r in drift.items() if n != "rule_based"),
            key=lambda r: r.regret_after_change,
        )
        line = (
            f"- Shifting rates: the rules can't adapt and lost {rules.regret_after_change:,.0f}"
            f" after the shift. The best learner here, {best.policy_name}, lost "
            f"{best.regret_after_change:,.0f}."
        )
        if forgetting_sweep:
            winner = min(forgetting_sweep, key=lambda row: row.regret_after_change)
            noise = float(np.mean([row.total_regret_std for row in forgetting_sweep]))
            line += (
                f" In the forgetting sweep the lowest after-change regret came from "
                f"`{winner.label}`. Seed-to-seed spread is about {noise:,.0f}, so treat small "
                "gaps between settings as noise."
            )
        lines.append(line)
    lines.append("")
    return lines


def render_report(
    sections: Sequence[tuple[Scenario, Mapping[str, ScenarioPolicyResult]]],
    delay_sweep: Sequence[SweepRow],
    forgetting_sweep: Sequence[SweepRow],
    n_points: int,
    config: ActivityConfig,
) -> str:
    lines = [
        "# Offer Bandit Scenario Simulations",
        "",
        f"Each scenario replays the same {n_points:,} visits from the SQL feature store",
        f"({config.simulation_days} days, {config.n_users:,} users). All learners start with no",
        "data. Regret is expected revenue lost against the best available offer under the",
        "true model at that moment. The change point is halfway through. In the delay",
        "scenario nothing changes there, so the two halves just show learning progress.",
        "Ground truth is synthetic, and the main tables are single runs, so differences of",
        "10 to 15% between similar learners can come down to the seed.",
    ]
    for scenario, results in sections:
        lines.extend(["", f"## {scenario.name.replace('_', ' ').capitalize()}", ""])
        lines.append(scenario.description)
        lines.append("")
        lines.extend(_policy_table(results))
        if scenario.launch_offer is not None:
            lines.extend(_cold_start_section(results))
        if scenario.mean_feedback_delay_hours > 0:
            applied = [r.feedback_applied for r in results.values() if r.feedback_applied]
            if applied:
                lines.extend(
                    [
                        "",
                        (f"By the end, {applied[0]:,} of {n_points:,} outcomes had reached the "
                        "learners. The rest were still in flight."),
                    ]
                )
            if delay_sweep:
                lines.extend(
                    _sweep_table("Thompson Sampling vs feedback delay", "Mean delay", delay_sweep)
                )
        if scenario.model_after is not None and forgetting_sweep:
            lines.extend(
                _sweep_table(
                    "Thompson Sampling vs forgetting factor", "Forgetting", forgetting_sweep
                )
            )
    lines.extend(_takeaways(sections, delay_sweep, forgetting_sweep))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ActivityConfig(n_users=args.users, simulation_days=args.simulation_days)
    store, population = build_feature_store(config)
    with store:
        points = decision_points(store, population)

    scenarios = [
        cold_start_scenario(),
        delayed_rewards_scenario(args.delay_hours),
        shifting_rates_scenario(),
    ]
    sections = [
        (scenario, run_scenario(scenario, default_factories(), points, seed=args.seed))
        for scenario in scenarios
    ]

    delay_sweep: list[SweepRow] = []
    forgetting_sweep: list[SweepRow] = []
    if not args.skip_sweeps:
        seeds = [args.seed + offset for offset in range(args.sweep_seeds)]
        delay_sweep = [
            sweep_row(f"{hours:g} h", delayed_rewards_scenario(hours), thompson_factory(),
                      points, seeds)
            for hours in DELAY_SWEEP_HOURS
        ]
        drift = shifting_rates_scenario()
        forgetting_sweep = [
            sweep_row("none" if factor == 1.0 else f"{factor:g}", drift,
                      thompson_factory(factor), points, seeds)
            for factor in FORGETTING_SWEEP
        ]

    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(
        render_report(sections, delay_sweep, forgetting_sweep, len(points), config),
        encoding="utf-8",
    )
    args.artifact_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenarios": {
            scenario.name: {name: asdict(result) for name, result in results.items()}
            for scenario, results in sections
        },
        "delay_sweep": [asdict(row) for row in delay_sweep],
        "forgetting_sweep": [asdict(row) for row in forgetting_sweep],
    }
    args.artifact_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.report_md}")
    print(f"Wrote {args.artifact_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
