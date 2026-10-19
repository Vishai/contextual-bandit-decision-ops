from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

import httpx
import uvicorn

from .offer_api import create_offer_app
from .offer_service import OfferServiceConfig

SMOKE_CONTEXT = {
    "device": "mobile",
    "region": "east",
    "engagement_score": 0.6,
    "sessions_7d": 2,
    "sessions_30d": 9,
    "days_since_last_visit": 1.0,
    "tenure_days": 210.0,
    "purchases_90d": 3,
    "revenue_90d": 190.0,
    "avg_order_value": 63.0,
    "days_since_last_purchase": 8.0,
    "offer_impressions_30d": 2,
    "offer_conversions_30d": 0,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or smoke-test the offer decision service")
    defaults = OfferServiceConfig()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--environment", default=defaults.environment)
    parser.add_argument("--feature-store", type=Path, default=None)
    parser.add_argument("--model-artifact", type=Path, default=None)
    parser.add_argument("--decision-log", type=Path, default=Path(defaults.decision_log_path))
    parser.add_argument("--feedback-log", type=Path, default=Path(defaults.feedback_log_path))
    parser.add_argument("--exploration-floor", type=float, default=defaults.exploration_floor)
    parser.add_argument("--latency-budget-ms", type=float, default=defaults.latency_budget_ms)
    parser.add_argument(
        "--no-online-updates",
        action="store_true",
        help="Serve a frozen posterior and only learn through retraining",
    )
    parser.add_argument(
        "--model-poll-seconds",
        type=float,
        default=defaults.model_poll_seconds,
        help="Check the model artifact for a new version this often; 0 turns it off",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Exercise the API and exit")
    return parser


def config_from_args(args: argparse.Namespace) -> OfferServiceConfig:
    model_artifact = args.model_artifact
    if model_artifact is not None and not model_artifact.exists():
        note = "will load it once it appears" if args.model_poll_seconds > 0 else "serving untrained"
        print(f"Model artifact {model_artifact} not found yet, {note}", file=sys.stderr)
    return OfferServiceConfig(
        seed=args.seed,
        environment=args.environment,
        decision_log_path=args.decision_log,
        feedback_log_path=args.feedback_log,
        feature_store_path=args.feature_store,
        model_artifact_path=model_artifact,
        exploration_floor=args.exploration_floor,
        latency_budget_ms=args.latency_budget_ms,
        online_updates=not args.no_online_updates,
        model_poll_seconds=args.model_poll_seconds,
    )


async def _smoke_requests(config: OfferServiceConfig) -> dict[str, object]:
    app = create_offer_app(config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://smoke-test") as client:
        health = await client.get("/health")
        decision = await client.post(
            "/decide", json={"user_id": "smoke-user", "context": SMOKE_CONTEXT}
        )
        cold_start = await client.post(
            "/decide", json={"user_id": "smoke-new-user", "device": "tablet", "region": "west"}
        )
        feedback = await client.post(
            "/feedback", json={"event_id": decision.json()["event_id"], "converted": True}
        )
        posterior = await client.post(
            "/posterior", json={"user_id": "smoke-user", "context": SMOKE_CONTEXT}
        )
        metrics = await client.get("/metrics")
    responses = (health, decision, cold_start, feedback, posterior, metrics)
    passed = (
        all(response.status_code == 200 for response in responses)
        and cold_start.json()["feature_source"] == "cold_start"
        and feedback.json()["posterior_updated"] is True
        and metrics.json()["posterior_updates"] == 1
    )
    return {
        "passed": passed,
        "decision": decision.json(),
        "metrics": metrics.json(),
    }


def run_smoke_test(config: OfferServiceConfig) -> int:
    result = asyncio.run(_smoke_requests(config))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke_test:
        with tempfile.TemporaryDirectory() as directory:
            args.decision_log = Path(directory) / "decisions.jsonl"
            args.feedback_log = Path(directory) / "feedback.jsonl"
            return run_smoke_test(config_from_args(args))
    uvicorn.run(create_offer_app(config_from_args(args)), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
