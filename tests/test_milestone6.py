import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from contextual_bandit_decision_ops.api import create_app
from contextual_bandit_decision_ops.safety import PolicyConstraints
from contextual_bandit_decision_ops.service import ServiceConfig
from contextual_bandit_decision_ops.service_cli import main


@pytest.fixture
def service_app(tmp_path: Path) -> FastAPI:
    config = ServiceConfig(
        seed=7,
        decision_log_path=tmp_path / "decisions.jsonl",
        feedback_log_path=tmp_path / "feedback.jsonl",
        safety_constraints=PolicyConstraints(blocked_actions=frozenset({2})),
    )
    fixed_time = datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC)
    return create_app(config=config, clock=lambda: fixed_time)


async def _request(
    app: FastAPI,
    method: str,
    path: str,
    **kwargs: object,
) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://staging-test",
    ) as client:
        return await client.request(method, path, **kwargs)


def request(
    app: FastAPI,
    method: str,
    path: str,
    **kwargs: object,
) -> httpx.Response:
    return asyncio.run(_request(app, method, path, **kwargs))


def valid_decision_payload() -> dict[str, object]:
    return {
        "user_id": "user-123",
        "context": {
            "age": 34,
            "engagement": 0.75,
            "region": "north",
        },
        "metadata": {"request_source": "pytest"},
    }


def test_health_endpoint(service_app: FastAPI) -> None:
    response = request(service_app, "GET", "/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "environment": "local-staging",
        "staging_only": True,
    }


def test_policy_metadata_endpoint(service_app: FastAPI) -> None:
    response = request(service_app, "GET", "/policy")

    assert response.status_code == 200
    payload = response.json()
    assert payload["policy_name"] == "random_uniform"
    assert payload["available_actions"] == [0, 1]
    assert payload["safety_constraints"]["blocked_actions"] == [2]
    assert payload["staging_only"] is True


def test_valid_decide_request(service_app: FastAPI) -> None:
    response = request(
        service_app,
        "POST",
        "/decide",
        json=valid_decision_payload(),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["event_id"] == "service-event-7-00000000"
    assert payload["action"] in {0, 1}
    assert payload["propensity"] == pytest.approx(0.5)
    assert payload["policy_name"] == "random_uniform"
    assert payload["timestamp"] == "2025-01-02T03:04:05Z"


def test_invalid_decide_request_is_rejected(service_app: FastAPI) -> None:
    payload = valid_decision_payload()
    payload["context"] = {
        "age": 12,
        "engagement": 0.75,
        "region": "north",
    }

    response = request(service_app, "POST", "/decide", json=payload)

    assert response.status_code == 422


def test_feedback_is_logged(service_app: FastAPI) -> None:
    decision = request(
        service_app,
        "POST",
        "/decide",
        json=valid_decision_payload(),
    ).json()
    response = request(
        service_app,
        "POST",
        "/feedback",
        json={
            "event_id": decision["event_id"],
            "reward": 0.8,
            "metadata": {"outcome": "converted"},
        },
    )

    assert response.status_code == 200
    service = service_app.state.decision_service
    feedback_record = json.loads(
        service.config.feedback_log_path.read_text(encoding="utf-8").strip()
    )
    assert feedback_record["event_id"] == decision["event_id"]
    assert feedback_record["reward"] == pytest.approx(0.8)
    assert feedback_record["metadata"] == {"outcome": "converted"}


def test_invalid_reward_is_rejected(service_app: FastAPI) -> None:
    decision = request(
        service_app,
        "POST",
        "/decide",
        json=valid_decision_payload(),
    ).json()

    response = request(
        service_app,
        "POST",
        "/feedback",
        json={"event_id": decision["event_id"], "reward": 1.5},
    )

    assert response.status_code == 422


def test_metrics_counters(service_app: FastAPI) -> None:
    decisions = [
        request(
            service_app,
            "POST",
            "/decide",
            json=valid_decision_payload(),
        ).json()
        for _ in range(3)
    ]
    request(
        service_app,
        "POST",
        "/feedback",
        json={"event_id": decisions[0]["event_id"], "reward": 0.25},
    )

    response = request(service_app, "GET", "/metrics")

    assert response.status_code == 200
    metrics = response.json()
    assert metrics["decisions"] == 3
    assert metrics["feedback_records"] == 1
    assert sum(metrics["action_counts"].values()) == 3
    assert metrics["average_observed_reward"] == pytest.approx(0.25)


def test_decision_log_is_created(service_app: FastAPI) -> None:
    response = request(
        service_app,
        "POST",
        "/decide",
        json=valid_decision_payload(),
    )
    service = service_app.state.decision_service

    assert service.config.decision_log_path.exists()
    decision_record = json.loads(
        service.config.decision_log_path.read_text(encoding="utf-8").strip()
    )
    assert decision_record["event_id"] == response.json()["event_id"]
    assert decision_record["user_id"] == "user-123"
    assert decision_record["context"]["region"] == "north"
    assert decision_record["propensity"] == pytest.approx(0.5)


def test_service_cli_in_process_smoke_test(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    decision_log = tmp_path / "smoke-decisions.jsonl"
    feedback_log = tmp_path / "smoke-feedback.jsonl"

    exit_code = main(
        [
            "--smoke-test",
            "--decision-log",
            str(decision_log),
            "--feedback-log",
            str(feedback_log),
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["passed"] is True
    assert output["metrics"]["decisions"] == 1
    assert decision_log.exists()
    assert feedback_log.exists()
