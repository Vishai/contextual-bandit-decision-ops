import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi import FastAPI

from contextual_bandit_decision_ops.bayesian_policies import BayesianLogisticThompsonSampling
from contextual_bandit_decision_ops.latency_benchmark import main as latency_main
from contextual_bandit_decision_ops.model_store import load_policy_artifact, save_policy_artifact
from contextual_bandit_decision_ops.offer_api import LATENCY_HEADER, create_offer_app
from contextual_bandit_decision_ops.offer_features import new_user_context
from contextual_bandit_decision_ops.offer_service import OfferServiceConfig
from contextual_bandit_decision_ops.offer_service_cli import SMOKE_CONTEXT
from contextual_bandit_decision_ops.offer_service_cli import main as service_main
from contextual_bandit_decision_ops.offers import default_offer_catalog
from contextual_bandit_decision_ops.synthetic_activity import ActivityConfig, build_feature_store

CATALOG = default_offer_catalog()
FIXED_TIME = datetime(2025, 3, 20, 12, tzinfo=UTC)


def make_app(tmp_path: Path, **overrides: object) -> FastAPI:
    settings: dict[str, object] = {
        "decision_log_path": tmp_path / "decisions.jsonl",
        "feedback_log_path": tmp_path / "feedback.jsonl",
    }
    settings.update(overrides)
    return create_offer_app(OfferServiceConfig(**settings), clock=lambda: FIXED_TIME)  # type: ignore[arg-type]


async def _call(app: FastAPI, calls: list[tuple[str, str, dict | None]]) -> list[httpx.Response]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        responses = []
        for method, path, body in calls:
            responses.append(await client.request(method, path, json=body))
        return responses


def call(app: FastAPI, *calls: tuple[str, str, dict | None]) -> list[httpx.Response]:
    return asyncio.run(_call(app, list(calls)))


def decide_body(user_id: str = "user-1", **extra: object) -> dict:
    return {"user_id": user_id, "context": dict(SMOKE_CONTEXT), **extra}


def test_decide_with_request_context(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    (response,) = call(app, ("POST", "/decide", decide_body(metadata={"page": "home"})))
    assert response.status_code == 200
    body = response.json()
    assert body["offer_id"] in CATALOG
    assert 0.0 < body["propensity"] <= 1.0
    assert body["feature_source"] == "request"
    assert body["model_version"] == "untrained"
    assert body["decision_latency_ms"] < 100.0
    assert float(response.headers[LATENCY_HEADER]) >= 0.0

    record = json.loads((tmp_path / "decisions.jsonl").read_text(encoding="utf-8"))
    assert record["event_id"] == body["event_id"]
    assert record["context"]["device"] == "mobile"
    assert record["propensity"] == body["propensity"]
    assert record["metadata"] == {"page": "home"}
    assert record["available_offers"] == list(CATALOG.offer_ids)


def test_features_come_from_the_store_or_cold_start(tmp_path: Path) -> None:
    db = tmp_path / "store.sqlite"
    store, population = build_feature_store(ActivityConfig(n_users=60, simulation_days=2), db)
    store.close()
    known = population.users[0].user_id
    app = make_app(tmp_path, feature_store_path=db)
    stored, cold, missing = call(
        app,
        ("POST", "/decide", {"user_id": known}),
        ("POST", "/decide", {"user_id": "brand-new", "device": "tablet", "region": "south"}),
        ("POST", "/decide", {"user_id": "brand-new-2"}),
    )
    assert stored.status_code == 200
    assert stored.json()["feature_source"] == "feature_store"
    assert cold.json()["feature_source"] == "cold_start"
    assert missing.status_code == 422
    assert "cold-start" in missing.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {"user_id": "u", "context": {**SMOKE_CONTEXT, "sessions_7d": 50}},
        {"user_id": "u", "context": {**SMOKE_CONTEXT, "device": "watch"}},
        {"user_id": "u", "context": {**SMOKE_CONTEXT, "engagement_score": 2.0}},
        {"user_id": "u", "context": SMOKE_CONTEXT, "unexpected": True},
        {"user_id": "", "context": SMOKE_CONTEXT},
        {"user_id": "u", "context": SMOKE_CONTEXT, "available_offers": ["not_an_offer"]},
    ],
)
def test_bad_requests_are_rejected(tmp_path: Path, body: dict) -> None:
    (response,) = call(make_app(tmp_path), ("POST", "/decide", body))
    assert response.status_code == 422


def test_only_available_offers_are_chosen(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    allowed = ["bundle_deal", "loyalty_points"]
    responses = call(app, *[("POST", "/decide", decide_body(f"u{i}", available_offers=allowed))
                            for i in range(30)])
    assert {r.json()["offer_id"] for r in responses} <= set(allowed)
    assert all(r.json()["propensity"] >= 0.05 / 2 for r in responses)


def test_feedback_updates_the_posterior_online(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    service = app.state.offer_service
    (decision,) = call(app, ("POST", "/decide", decide_body()))
    offer_id = decision.json()["offer_id"]
    before = service.policy.posteriors[offer_id].mean.copy()
    feedback, duplicate, unknown, model = call(
        app,
        ("POST", "/feedback", {"event_id": decision.json()["event_id"], "converted": True}),
        ("POST", "/feedback", {"event_id": decision.json()["event_id"], "converted": True}),
        ("POST", "/feedback", {"event_id": "missing", "converted": False}),
        ("GET", "/model", None),
    )
    assert feedback.status_code == 200
    assert feedback.json()["posterior_updated"] is True
    assert duplicate.status_code == 409
    assert unknown.status_code == 404
    assert not np.allclose(service.policy.posteriors[offer_id].mean, before)
    assert model.json()["observations_by_offer"][offer_id] == 1
    record = json.loads((tmp_path / "feedback.jsonl").read_text(encoding="utf-8"))
    assert record["revenue"] == CATALOG.revenue(offer_id)
    assert record["offer_id"] == offer_id


def test_explicit_revenue_and_frozen_mode(tmp_path: Path) -> None:
    app = make_app(tmp_path, online_updates=False)
    (decision,) = call(app, ("POST", "/decide", decide_body()))
    feedback, metrics = call(
        app,
        ("POST", "/feedback",
         {"event_id": decision.json()["event_id"], "converted": True, "revenue": 12.5}),
        ("GET", "/metrics", None),
    )
    assert feedback.json()["posterior_updated"] is False
    assert metrics.json()["revenue"] == 12.5
    assert metrics.json()["posterior_updates"] == 0


def test_exploration_floor_bounds_propensities(tmp_path: Path) -> None:
    app = make_app(tmp_path, exploration_floor=0.5, seed=3)
    responses = call(app, *[("POST", "/decide", decide_body(f"u{i}")) for i in range(200)])
    propensities = [r.json()["propensity"] for r in responses]
    assert min(propensities) >= 0.5 / len(CATALOG) - 1e-9
    explored = np.mean([r.json()["explored"] for r in responses])
    assert 0.38 < explored < 0.62


def test_metrics_and_latency_tracking(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    decisions = call(app, *[("POST", "/decide", decide_body(f"u{i}")) for i in range(25)])
    call(app, *[("POST", "/feedback", {"event_id": d.json()["event_id"], "converted": i % 5 == 0})
                for i, d in enumerate(decisions[:10])])
    (metrics,) = call(app, ("GET", "/metrics", None))
    body = metrics.json()
    assert body["decisions"] == 25
    assert body["feedback_records"] == 10
    assert body["conversions"] == 2
    assert body["pending_feedback"] == 15
    assert sum(body["allocation"].values()) == 25
    for key in ("decision_latency", "request_latency"):
        summary = body[key]
        assert summary["count"] == 25
        assert summary["p50_ms"] <= summary["p95_ms"] <= summary["p99_ms"] <= summary["max_ms"]
        assert summary["budget_ms"] == 100.0
        assert summary["within_budget_rate"] == 1.0


def test_old_pending_decisions_expire(tmp_path: Path) -> None:
    app = make_app(tmp_path, max_pending_decisions=2)
    first, _, _ = call(app, *[("POST", "/decide", decide_body(f"u{i}")) for i in range(3)])
    (late,) = call(app, ("POST", "/feedback",
                         {"event_id": first.json()["event_id"], "converted": True}))
    assert late.status_code == 404


def test_posterior_endpoint_reports_every_offer(tmp_path: Path) -> None:
    (response,) = call(make_app(tmp_path), ("POST", "/posterior", decide_body()))
    body = response.json()
    assert [row["offer_id"] for row in body["offers"]] == list(CATALOG.offer_ids)
    for row in body["offers"]:
        assert 0.0 <= row["lower"] <= row["mean"] <= row["upper"] <= 1.0


def test_model_artifacts_load_and_hot_reload(tmp_path: Path) -> None:
    artifact = tmp_path / "models" / "current.json"
    trained = BayesianLogisticThompsonSampling(CATALOG)
    trained.update(new_user_context("a", "mobile", "east"), "bundle_deal", True)
    save_policy_artifact(trained, artifact, "v1", {"trained_on": 1})
    assert load_policy_artifact(artifact, CATALOG).model_version == "v1"

    app = make_app(tmp_path, model_artifact_path=artifact)
    (health,) = call(app, ("GET", "/health", None))
    assert health.json()["model_version"] == "v1"

    save_policy_artifact(BayesianLogisticThompsonSampling(CATALOG), artifact, "v2")
    reload, decision = call(app, ("POST", "/model/reload", None),
                            ("POST", "/decide", decide_body()))
    assert reload.status_code == 200
    assert reload.json()["model_version"] == "v2"
    assert decision.json()["model_version"] == "v2"

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["policy"]["feature_names"] = ["bias"]
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    (broken,) = call(app, ("POST", "/model/reload", None))
    assert broken.status_code == 422

    (unconfigured,) = call(make_app(tmp_path), ("POST", "/model/reload", None))
    assert unconfigured.status_code == 409


def test_catalog_and_health(tmp_path: Path) -> None:
    offers, health = call(make_app(tmp_path, environment="staging"), ("GET", "/offers", None),
                          ("GET", "/health", None))
    assert [o["offer_id"] for o in offers.json()] == list(CATALOG.offer_ids)
    assert health.json() == {
        "status": "ok",
        "environment": "staging",
        "policy_name": "thompson_sampling",
        "model_version": "untrained",
        "feature_store": False,
    }


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        OfferServiceConfig(exploration_floor=1.0)
    with pytest.raises(ValueError):
        OfferServiceConfig(propensity_samples=0)
    with pytest.raises(ValueError):
        OfferServiceConfig(latency_budget_ms=0)


def test_smoke_test_command() -> None:
    assert service_main(["--smoke-test"]) == 0


def test_latency_benchmark_passes_budget(tmp_path: Path) -> None:
    report = tmp_path / "latency.md"
    code = latency_main(["--requests", "150", "--users", "80", "--report-md", str(report),
                         "--artifact-json", str(tmp_path / "latency.json")])
    assert code == 0
    assert "**Result: PASS.**" in report.read_text(encoding="utf-8")


def test_event_ids_stay_unique_across_restarts(tmp_path: Path) -> None:
    (first,) = call(make_app(tmp_path), ("POST", "/decide", decide_body()))
    (second,) = call(make_app(tmp_path), ("POST", "/decide", decide_body()))
    assert first.json()["event_id"] != second.json()["event_id"]


def test_service_polls_for_new_model_versions(tmp_path: Path) -> None:
    artifact = tmp_path / "models" / "latest.json"
    app = make_app(tmp_path, model_artifact_path=artifact, model_poll_seconds=0.001)
    service = app.state.offer_service
    (before,) = call(app, ("POST", "/decide", decide_body()))
    assert before.json()["model_version"] == "untrained"

    save_policy_artifact(BayesianLogisticThompsonSampling(CATALOG), artifact, "v1")
    service._next_poll = 0.0
    (after,) = call(app, ("POST", "/decide", decide_body()))
    assert after.json()["model_version"] == "v1"

    artifact.write_text("{not json", encoding="utf-8")
    service._next_poll = 0.0
    (still,) = call(app, ("POST", "/decide", decide_body()))
    assert still.status_code == 200
    assert still.json()["model_version"] == "v1"
