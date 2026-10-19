from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import numpy as np

from .bayesian_policies import BayesianLogisticThompsonSampling
from .model_store import load_policy_artifact
from .offer_api import create_offer_app
from .offer_history import load_logged_events, log_historical_decisions
from .offer_logs import read_service_logs
from .offer_monitoring import (
    MonitoringResult,
    MonitoringThresholds,
    analyze_offer_logs,
    split_by_time,
)
from .offer_policies import EpsilonExplorationPolicy, RuleBasedOfferPolicy
from .offer_scenarios import DRIFT_SHIFTS
from .offer_service import OfferServiceConfig
from .offer_simulation import ConversionModel, default_conversion_model
from .offers import default_offer_catalog
from .synthetic_activity import ActivityConfig, DecisionPoint, build_feature_store, decision_points


class SimulatedClock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now


@dataclass(frozen=True)
class DemoTraffic:
    decision_log: Path
    feedback_log: Path
    model: BayesianLogisticThompsonSampling
    reference_truth: ConversionModel
    current_truth: ConversionModel
    split_at: datetime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monitor regret, offer allocation, latency and data drift from service logs"
    )
    parser.add_argument("--decision-log", type=Path, default=None, help="File or replica folder")
    parser.add_argument("--feedback-log", type=Path, default=None, help="File or replica folder")
    parser.add_argument("--split-at", default=None, help="ISO time where the current window starts")
    parser.add_argument("--reference-fraction", type=float, default=0.5)
    parser.add_argument("--model-artifact", type=Path, default=None)
    parser.add_argument("--latency-budget-ms", type=float, default=100.0)
    parser.add_argument("--users", type=int, default=1_500, help="Demo population size")
    parser.add_argument("--decisions", type=int, default=3_000, help="Demo decisions per window")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--fail-on-alert", action="store_true")
    parser.add_argument(
        "--report-md", type=Path, default=Path("reports/offer_monitoring_report.md")
    )
    parser.add_argument(
        "--artifact-json", type=Path, default=Path("artifacts/offer_monitoring_report.json")
    )
    return parser


def _points(config: ActivityConfig, limit: int) -> list[DecisionPoint]:
    store, population = build_feature_store(config)
    with store:
        return decision_points(store, population)[:limit]


def simulate_demo_traffic(directory: Path, users: int, decisions: int, seed: int) -> DemoTraffic:
    """A normal week, then a week where a mobile campaign brings in lots of new users
    and conversion rates move, all served through the real API.

    The model starts warm, fit on two weeks of logs from the old rule-based system,
    and keeps learning online from every outcome, the way it would in production.
    """
    truth = default_conversion_model()
    catalog = truth.catalog
    history = ActivityConfig(n_users=users, seed=seed, simulation_days=14)
    store, population = build_feature_store(history)
    with store:
        logging_policy = EpsilonExplorationPolicy(RuleBasedOfferPolicy(catalog), catalog, 0.2)
        log_historical_decisions(store, population, truth, logging_policy, seed)
        events = load_logged_events(store)
    policy = BayesianLogisticThompsonSampling(catalog)
    policy.fit([e.context for e in events], [e.offer_id for e in events],
               [e.converted for e in events])

    days = max(1, round(decisions / max(users * 0.3, 1)))
    reference_config = ActivityConfig(n_users=users, seed=seed + 1, simulation_days=days,
                                      start=history.end)
    current_config = ActivityConfig(
        n_users=users,
        seed=seed + 2,
        simulation_days=days,
        start=reference_config.end,
        new_user_share=0.35,
        device_shares=(0.78, 0.17, 0.05),
    )
    shifted = truth.with_logit_shift(DRIFT_SHIFTS)
    windows = [
        (_points(reference_config, decisions), truth),
        (_points(current_config, decisions), shifted),
    ]

    decision_log = directory / "decisions.jsonl"
    feedback_log = directory / "feedback.jsonl"
    clock = SimulatedClock(reference_config.start)
    app = create_offer_app(
        OfferServiceConfig(seed=seed, decision_log_path=decision_log,
                           feedback_log_path=feedback_log),
        policy=policy,
        clock=clock,
    )

    async def drive() -> None:
        rng = np.random.default_rng(seed)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://monitor") as client:
            for points, model in windows:
                for point in points:
                    clock.now = point.timestamp
                    context = point.context.to_dict()
                    user_id = context.pop("user_id")
                    response = await client.post(
                        "/decide", json={"user_id": user_id, "context": context}
                    )
                    response.raise_for_status()
                    body = response.json()
                    rate = model.conversion_probability(point.context, body["offer_id"])
                    clock.now = point.timestamp + timedelta(minutes=30)
                    await client.post(
                        "/feedback",
                        json={"event_id": body["event_id"], "converted": bool(rng.random() < rate)},
                    )

    asyncio.run(drive())
    return DemoTraffic(
        decision_log, feedback_log, app.state.offer_service.policy, truth, shifted,
        current_config.start,
    )


def _fmt(value: float | None, pattern: str) -> str:
    return "n/a" if value is None else format(value, pattern)


def render_report(result: MonitoringResult, source: str) -> str:
    reference, current = result.reference, result.current
    offers = list(current.allocation)
    icon = {"ok": "ok", "warn": "WARN", "alert": "ALERT"}
    lines = [
        "# Offer Service Monitoring",
        "",
        f"**Overall status: {icon[result.status]}.** Source: {source}.",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
        *(f"| {c.name} | {icon[c.status]} | {c.detail} |" for c in result.checks),
        "",
        "## Windows",
        "",
        "| Metric | Reference | Current |",
        "| --- | ---: | ---: |",
        f"| From | {reference.start} | {current.start} |",
        f"| To | {reference.end} | {current.end} |",
        f"| Decisions | {reference.decisions:,} | {current.decisions:,} |",
        f"| Unique users | {reference.unique_users:,} | {current.unique_users:,} |",
        (f"| Conversion | {_fmt(reference.conversion_rate, '.2%')} | "
        f"{_fmt(current.conversion_rate, '.2%')} |"),
        (f"| Revenue / decision | {_fmt(reference.revenue_per_decision, '.3f')} | "
        f"{_fmt(current.revenue_per_decision, '.3f')} |"),
        f"| Feedback coverage | {reference.feedback_coverage:.1%} | {current.feedback_coverage:.1%} |",
        f"| Exploration rate | {reference.exploration_rate:.1%} | {current.exploration_rate:.1%} |",
        (f"| Decision latency p50 ms | {_fmt(reference.latency.p50_ms, '.2f')} | "
        f"{_fmt(current.latency.p50_ms, '.2f')} |"),
        (f"| Decision latency p95 ms | {_fmt(reference.latency.p95_ms, '.2f')} | "
        f"{_fmt(current.latency.p95_ms, '.2f')} |"),
        (f"| Decision latency p99 ms | {_fmt(reference.latency.p99_ms, '.2f')} | "
        f"{_fmt(current.latency.p99_ms, '.2f')} |"),
        (f"| Minimum propensity | {_fmt(reference.min_propensity, '.4f')} | "
        f"{_fmt(current.min_propensity, '.4f')} |"),
        (f"| Regret / decision, model estimate | "
        f"{_fmt(reference.estimated_regret_per_decision, '.3f')} | "
        f"{_fmt(current.estimated_regret_per_decision, '.3f')} |"),
        (f"| Regret / decision, true (simulation only) | "
        f"{_fmt(reference.true_regret_per_decision, '.3f')} | "
        f"{_fmt(current.true_regret_per_decision, '.3f')} |"),
        "",
        "The model estimate scores each logged decision against the best offer under the",
        "served model's own posterior. It shows how much the policy is still exploring, but it",
        "can't see where the model itself is wrong. True regret needs the simulator.",
        "",
        "## Offer allocation",
        "",
        "| Offer | Reference | Current | Conversion (reference) | Conversion (current) |",
        "| --- | ---: | ---: | ---: | ---: |",
        *(
            f"| {offer} | {reference.allocation_share[offer]:.1%} | "
            f"{current.allocation_share[offer]:.1%} | "
            f"{_fmt(reference.conversion_by_offer.get(offer), '.2%')} | "
            f"{_fmt(current.conversion_by_offer.get(offer), '.2%')} |"
            for offer in offers
        ),
        "",
        "### Allocation over time",
        "",
        "Both windows in order, split into equal slices of decisions.",
        "",
        "| Slice | " + " | ".join(offers) + " |",
        "| ---: | " + " | ".join("---:" for _ in offers) + " |",
        *(
            f"| {index + 1} | " + " | ".join(f"{row[o]:.1%}" for o in offers) + " |"
            for index, row in enumerate(result.allocation_over_time)
        ),
        "",
        "## Data drift",
        "",
        "PSI for numeric features (under 0.1 stable, 0.1 to 0.25 moderate, over 0.25 large),",
        "total variation distance for categorical ones.",
        "",
        "| Feature | Statistic | Value | Status |",
        "| --- | --- | ---: | --- |",
        *(
            f"| {d.feature} | {d.statistic.upper()} | {d.value:.3f} | {icon[d.status]} |"
            for d in sorted(result.drift, key=lambda d: -d.value)
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    catalog = default_offer_catalog()
    thresholds = MonitoringThresholds(latency_budget_ms=args.latency_budget_ms)
    model = reference_truth = current_truth = None
    split_at = datetime.fromisoformat(args.split_at) if args.split_at else None

    with tempfile.TemporaryDirectory() as directory:
        if args.decision_log is None:
            demo = simulate_demo_traffic(Path(directory), args.users, args.decisions, args.seed)
            decision_log, feedback_log = demo.decision_log, demo.feedback_log
            model, split_at = demo.model, demo.split_at
            reference_truth, current_truth = demo.reference_truth, demo.current_truth
            source = (
                "simulated traffic through the API: a normal week, then a mobile campaign "
                "with new users and shifted conversion rates"
            )
        else:
            decision_log, feedback_log = args.decision_log, args.feedback_log
            source = f"service logs in {args.decision_log}"
        if args.model_artifact is not None:
            model = load_policy_artifact(args.model_artifact, catalog).policy
        decisions, feedback = read_service_logs(decision_log, feedback_log)

    if not decisions:
        print("No decisions found in the logs.", file=sys.stderr)
        return 2
    reference, current = split_by_time(decisions, split_at, args.reference_fraction)
    result = analyze_offer_logs(
        reference, current, feedback, catalog, thresholds, model, reference_truth, current_truth
    )
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(render_report(result, source), encoding="utf-8")
    args.artifact_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"status": result.status, **asdict(result)}
    args.artifact_json.write_text(json.dumps(payload, indent=2, default=str), "utf-8")
    print(f"Monitoring status: {result.status}")
    print(f"Wrote {args.report_md}")
    return 1 if args.fail_on_alert and result.status == "alert" else 0


if __name__ == "__main__":
    sys.exit(main())
