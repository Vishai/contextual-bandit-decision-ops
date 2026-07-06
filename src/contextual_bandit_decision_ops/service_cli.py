from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
import uvicorn

from .api import create_app
from .service import ServiceConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or smoke-test the local/staging decision service"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Local bind host")
    parser.add_argument("--port", type=int, default=8000, help="Local bind port")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic policy seed")
    parser.add_argument(
        "--decision-log",
        type=Path,
        default=Path("logs/decisions.jsonl"),
    )
    parser.add_argument(
        "--feedback-log",
        type=Path,
        default=Path("logs/feedback.jsonl"),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Exercise the API in process and exit",
    )
    return parser


async def _smoke_test_requests(config: ServiceConfig) -> dict[str, object]:
    fixed_time = datetime(2024, 1, 1, tzinfo=UTC)
    app = create_app(config=config, clock=lambda: fixed_time)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://staging-test",
    ) as client:
        health = await client.get("/health")
        decision = await client.post(
            "/decide",
            json={
                "user_id": "smoke-user",
                "context": {
                    "age": 35,
                    "engagement": 0.6,
                    "region": "north",
                },
            },
        )
        feedback = await client.post(
            "/feedback",
            json={
                "event_id": decision.json()["event_id"],
                "reward": 1.0,
                "metadata": {"source": "in-process-smoke"},
            },
        )
        metrics = await client.get("/metrics")

    passed = all(response.status_code == 200 for response in (health, decision, feedback, metrics))
    return {
        "passed": passed,
        "health": health.json(),
        "metrics": metrics.json(),
    }


def _run_smoke_test(config: ServiceConfig) -> int:
    result = asyncio.run(_smoke_test_requests(config))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ServiceConfig(
        seed=args.seed,
        decision_log_path=args.decision_log,
        feedback_log_path=args.feedback_log,
    )
    if args.smoke_test:
        return _run_smoke_test(config)

    uvicorn.run(
        create_app(config=config),
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
