from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from .bayesian_policies import BayesianLogisticThompsonSampling, BetaBernoulliThompsonSampling
from .offer_evaluation import OfferPolicyResult, compare_offer_policies
from .offer_policies import (
    GreedyOfferPolicy,
    OracleOfferPolicy,
    RandomOfferPolicy,
    RuleBasedOfferPolicy,
    UCBOfferPolicy,
)
from .offer_simulation import default_conversion_model
from .synthetic_activity import ActivityConfig, build_feature_store, decision_points

BASELINE = "rule_based"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare offer policies online on the synthetic feature-store traffic"
    )
    defaults = ActivityConfig()
    parser.add_argument("--users", type=int, default=defaults.n_users)
    parser.add_argument("--simulation-days", type=int, default=defaults.simulation_days)
    parser.add_argument("--activity-seed", type=int, default=defaults.seed)
    parser.add_argument("--seed", type=int, default=21, help="Seed for policies and conversions")
    parser.add_argument("--ucb-width", type=float, default=1.0)
    parser.add_argument("--exploration-scale", type=float, default=1.0)
    parser.add_argument(
        "--report-md", type=Path, default=Path("reports/offer_policy_comparison.md")
    )
    parser.add_argument(
        "--artifact-json", type=Path, default=Path("artifacts/offer_policy_comparison.json")
    )
    return parser


def _lift(value: float, baseline: float) -> str:
    if baseline == 0:
        return "n/a"
    return f"{(value - baseline) / baseline:+.1%}"


def _second_half_regret(result: OfferPolicyResult) -> float:
    curve = result.regret_curve
    if not curve:
        return 0.0
    return curve[-1] - curve[len(curve) // 2]


def render_report(
    results: Mapping[str, OfferPolicyResult],
    config: ActivityConfig,
    seed: int,
) -> str:
    any_result = next(iter(results.values()))
    offers = list(any_result.allocation)
    baseline = results.get(BASELINE)
    lines = [
        "# Offer Policy Comparison",
        "",
        "Every policy sees the same stream of visits built from the SQL feature store and the",
        "same random draws for conversion. Learning policies start with no data and update",
        "after each outcome. Conversions come from a synthetic ground truth model, so these",
        "numbers show how the policies behave relative to each other, not real business results.",
        "",
        "## Setup",
        "",
        "| Setting | Value |",
        "| --- | ---: |",
        f"| Decisions | {any_result.decisions} |",
        f"| Unique users | {any_result.unique_users} |",
        f"| Simulated days | {config.simulation_days} |",
        f"| Activity seed | {config.seed} |",
        f"| Policy seed | {seed} |",
        "",
        "## Results",
        "",
        (
            "| Policy | Conversion | Revenue / user | Revenue / decision | Expected revenue / decision"
            " | Cumulative regret | Second-half regret | p50 ms | p99 ms |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result.conversion_rate:.2%} | {result.revenue_per_user:.2f} | "
            f"{result.revenue_per_decision:.3f} | {result.expected_revenue_per_decision:.3f} | "
            f"{result.cumulative_regret:,.0f} | {_second_half_regret(result):,.0f} | "
            f"{result.latency_p50_ms:.3f} | {result.latency_p99_ms:.3f} |"
        )
    if baseline is not None:
        lines.extend(
            [
                "",
                "## Against the rule-based strategy",
                "",
                "| Policy | Conversion change (pts) | Revenue / user lift | Regret change |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for name, result in results.items():
            if name == BASELINE:
                continue
            lines.append(
                f"| {name} | {(result.conversion_rate - baseline.conversion_rate) * 100:+.2f} | "
                f"{_lift(result.revenue_per_user, baseline.revenue_per_user)} | "
                f"{_lift(result.cumulative_regret, baseline.cumulative_regret)} |"
            )
    lines.extend(
        [
            "",
            "## Allocation",
            "",
            "| Policy | " + " | ".join(offers) + " |",
            "| --- | " + " | ".join("---:" for _ in offers) + " |",
        ]
    )
    for name, result in results.items():
        shares = result.allocation_share
        lines.append(f"| {name} | " + " | ".join(f"{shares[o]:.1%}" for o in offers) + " |")

    lines.extend(
        [
            "",
            "## Cumulative regret over time",
            "",
            "| Checkpoint | " + " | ".join(results) + " |",
            "| ---: | " + " | ".join("---:" for _ in results) + " |",
        ]
    )
    curve_length = len(any_result.regret_curve)
    for index in range(4, curve_length, 5):
        values = " | ".join(f"{r.regret_curve[index]:,.0f}" for r in results.values())
        lines.append(f"| {(index + 1) / curve_length:.0%} | {values} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Regret is the expected revenue lost against the best offer for each visit under",
            "  the true model. It is the least noisy way to compare the learners.",
            "- `greedy`, `ucb` and `thompson_sampling` share one Bayesian logistic model, so",
            "  they differ only in how they explore.",
            "- `beta_thompson_sampling` only splits by device, which shows how much the full",
            "  user context is worth.",
            "- `oracle` knows the true conversion rates and is only a reference point.",
            "- Latency is the in-process time for one choose plus one update.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ActivityConfig(
        n_users=args.users,
        seed=args.activity_seed,
        simulation_days=args.simulation_days,
    )
    model = default_conversion_model()
    catalog = model.catalog
    policies = [
        RandomOfferPolicy(catalog),
        RuleBasedOfferPolicy(catalog),
        GreedyOfferPolicy(catalog),
        UCBOfferPolicy(catalog, ucb_width=args.ucb_width),
        BayesianLogisticThompsonSampling(catalog, exploration_scale=args.exploration_scale),
        BetaBernoulliThompsonSampling(catalog),
        OracleOfferPolicy(model),
    ]
    store, population = build_feature_store(config)
    with store:
        points = decision_points(store, population)
    results = compare_offer_policies(policies, points, model, seed=args.seed)

    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(render_report(results, config, args.seed), encoding="utf-8")
    args.artifact_json.parent.mkdir(parents=True, exist_ok=True)
    args.artifact_json.write_text(
        json.dumps({name: asdict(result) for name, result in results.items()}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {args.report_md}")
    print(f"Wrote {args.artifact_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
