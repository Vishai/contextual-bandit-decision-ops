from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import numpy as np

from .offer_api import LATENCY_HEADER, create_offer_app
from .offer_api_schemas import LatencySummary
from .offer_service import OfferServiceConfig, summarize_latencies
from .offer_simulation import ConversionModel, default_conversion_model
from .synthetic_activity import ActivityConfig, DecisionPoint, build_feature_store, decision_points


@dataclass(frozen=True)
class LatencyBenchmarkResult:
    target: str
    requests: int
    budget_ms: float
    decide: LatencySummary
    feedback: LatencySummary
    server_decide: LatencySummary
    passed: bool
    conversion_rate_first_half: float
    conversion_rate_second_half: float
    revenue_per_decision_first_half: float
    revenue_per_decision_second_half: float
    allocation_first_half: dict[str, int]
    allocation_second_half: dict[str, int]
    posterior_updates: int


def _payload(point: DecisionPoint) -> dict[str, object]:
    context = point.context.to_dict()
    user_id = context.pop("user_id")
    return {"user_id": user_id, "context": context}


async def drive_service(
    client: httpx.AsyncClient,
    points: Sequence[DecisionPoint],
    truth: ConversionModel,
    n_requests: int,
    budget_ms: float,
    seed: int = 7,
    warmup: int = 20,
    target: str = "in-process",
) -> LatencyBenchmarkResult:
    """Send decide and feedback calls the way a real client would and time them.

    Outcomes come from the simulator's true conversion model, so the service learns
    from realistic feedback while it's being timed. Warm-up calls aren't counted.
    """
    if not points:
        raise ValueError("need at least one decision point")
    rng = np.random.default_rng(seed)
    decide_ms: list[float] = []
    feedback_ms: list[float] = []
    server_ms: list[float] = []
    outcomes: list[tuple[str, bool, float]] = []

    for index in range(warmup + n_requests):
        point = points[index % len(points)]
        started = time.perf_counter()
        decision = await client.post("/decide", json=_payload(point))
        decide_elapsed = (time.perf_counter() - started) * 1_000.0
        decision.raise_for_status()
        body = decision.json()
        offer_id = body["offer_id"]
        converted = bool(rng.random() < truth.conversion_probability(point.context, offer_id))
        started = time.perf_counter()
        feedback = await client.post(
            "/feedback", json={"event_id": body["event_id"], "converted": converted}
        )
        feedback_elapsed = (time.perf_counter() - started) * 1_000.0
        feedback.raise_for_status()
        if index < warmup:
            continue
        decide_ms.append(decide_elapsed)
        feedback_ms.append(feedback_elapsed)
        server_ms.append(float(decision.headers.get(LATENCY_HEADER, "nan")))
        revenue = truth.catalog.revenue(offer_id) if converted else 0.0
        outcomes.append((offer_id, converted, revenue))

    metrics = (await client.get("/metrics")).json()
    half = len(outcomes) // 2
    first, second = outcomes[:half], outcomes[half:]
    decide_summary = summarize_latencies(decide_ms, budget_ms)

    def rate(rows: list[tuple[str, bool, float]]) -> float:
        return float(np.mean([converted for _, converted, _ in rows])) if rows else 0.0

    def revenue(rows: list[tuple[str, bool, float]]) -> float:
        return float(np.mean([value for _, _, value in rows])) if rows else 0.0

    return LatencyBenchmarkResult(
        target=target,
        requests=n_requests,
        budget_ms=budget_ms,
        decide=decide_summary,
        feedback=summarize_latencies(feedback_ms, budget_ms),
        server_decide=summarize_latencies([v for v in server_ms if math.isfinite(v)], budget_ms),
        passed=decide_summary.p99_ms is not None and decide_summary.p99_ms <= budget_ms,
        conversion_rate_first_half=rate(first),
        conversion_rate_second_half=rate(second),
        revenue_per_decision_first_half=revenue(first),
        revenue_per_decision_second_half=revenue(second),
        allocation_first_half=dict(Counter(offer for offer, _, _ in first)),
        allocation_second_half=dict(Counter(offer for offer, _, _ in second)),
        posterior_updates=int(metrics["posterior_updates"]),
    )


def render_report(result: LatencyBenchmarkResult) -> str:
    def row(name: str, summary: LatencySummary) -> str:
        def fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.2f}"

        within = "n/a" if summary.within_budget_rate is None else f"{summary.within_budget_rate:.2%}"
        return (
            f"| {name} | {summary.count:,} | {fmt(summary.p50_ms)} | {fmt(summary.p95_ms)} | "
            f"{fmt(summary.p99_ms)} | {fmt(summary.max_ms)} | {within} |"
        )

    offers = sorted(set(result.allocation_first_half) | set(result.allocation_second_half))
    verdict = "PASS" if result.passed else "FAIL"
    return "\n".join(
        [
            "# Decision Latency Benchmark",
            "",
            f"Target: `{result.target}`. {result.requests:,} decide calls, each followed by a",
            "feedback call that updates the posterior online. Contexts come from the SQL",
            "feature store and outcomes from the simulator's true conversion model.",
            "",
            (f"**Result: {verdict}.** The p99 decide latency is compared with the "
            f"{result.budget_ms:g} ms budget."),
            "",
            "| Call | Count | p50 ms | p95 ms | p99 ms | Max ms | Within budget |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            row("POST /decide (client)", result.decide),
            row("POST /decide (server)", result.server_decide),
            row("POST /feedback (client)", result.feedback),
            "",
            "Client timing covers the full request through the HTTP stack. Server timing comes",
            f"from the `{LATENCY_HEADER}` header that the service middleware sets.",
            "",
            "## Online learning during the run",
            "",
            "| Half | Conversion | Revenue / decision |",
            "| --- | ---: | ---: |",
            (f"| First | {result.conversion_rate_first_half:.2%} | "
            f"{result.revenue_per_decision_first_half:.3f} |"),
            (f"| Second | {result.conversion_rate_second_half:.2%} | "
            f"{result.revenue_per_decision_second_half:.3f} |"),
            "",
            "| Offer | First half | Second half |",
            "| --- | ---: | ---: |",
            *(
                f"| {offer} | {result.allocation_first_half.get(offer, 0):,} | "
                f"{result.allocation_second_half.get(offer, 0):,} |"
                for offer in offers
            ),
            "",
            f"The service applied {result.posterior_updates:,} posterior updates during the run.",
            "",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark offer decision latency")
    parser.add_argument("--requests", type=int, default=2_000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--budget-ms", type=float, default=100.0)
    parser.add_argument("--users", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--url", default=None, help="Benchmark a running service instead of an in-process app"
    )
    parser.add_argument("--report-md", type=Path, default=Path("reports/latency_benchmark.md"))
    parser.add_argument(
        "--artifact-json", type=Path, default=Path("artifacts/latency_benchmark.json")
    )
    return parser


async def _run(args: argparse.Namespace, points: list[DecisionPoint]) -> LatencyBenchmarkResult:
    truth = default_conversion_model()
    if args.url:
        async with httpx.AsyncClient(base_url=args.url, timeout=10.0) as client:
            return await drive_service(
                client, points, truth, args.requests, args.budget_ms, args.seed, args.warmup,
                target=args.url,
            )
    with tempfile.TemporaryDirectory() as directory:
        app = create_offer_app(
            OfferServiceConfig(
                decision_log_path=Path(directory) / "decisions.jsonl",
                feedback_log_path=Path(directory) / "feedback.jsonl",
                latency_budget_ms=args.budget_ms,
            )
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://benchmark") as client:
            return await drive_service(
                client, points, truth, args.requests, args.budget_ms, args.seed, args.warmup
            )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store, population = build_feature_store(ActivityConfig(n_users=args.users, simulation_days=7))
    with store:
        points = decision_points(store, population)
    result = asyncio.run(_run(args, points))
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(render_report(result), encoding="utf-8")
    args.artifact_json.parent.mkdir(parents=True, exist_ok=True)
    args.artifact_json.write_text(json.dumps(asdict(result), indent=2, default=str), "utf-8")
    p99 = result.decide.p99_ms or 0.0
    print(f"decide p99 {p99:.2f} ms against a {args.budget_ms:g} ms budget: "
          f"{'PASS' if result.passed else 'FAIL'}")
    print(f"Wrote {args.report_md}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
