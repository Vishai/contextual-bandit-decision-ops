import asyncio
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pytest

from contextual_bandit_decision_ops import retraining
from contextual_bandit_decision_ops.bayesian_policies import BayesianLogisticThompsonSampling
from contextual_bandit_decision_ops.feature_store import FeatureStore
from contextual_bandit_decision_ops.model_registry import ModelRegistry
from contextual_bandit_decision_ops.offer_api import create_offer_app
from contextual_bandit_decision_ops.offer_backtest import OfflineEstimate
from contextual_bandit_decision_ops.offer_features import new_user_context
from contextual_bandit_decision_ops.offer_service import OfferServiceConfig
from contextual_bandit_decision_ops.offer_simulation import default_conversion_model
from contextual_bandit_decision_ops.retrain_cli import bootstrap_demo_store
from contextual_bandit_decision_ops.retrain_cli import main as retrain_main
from contextual_bandit_decision_ops.retraining import (
    RetrainingConfig,
    _gate,
    ingest_service_logs,
    retrain,
)

TRUTH = default_conversion_model()
CATALOG = TRUTH.catalog


@pytest.fixture(scope="module")
def demo_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("store") / "history.sqlite"
    bootstrap_demo_store(path, days=14)
    return path


def fresh_copy(source: Path, tmp_path: Path) -> Path:
    target = tmp_path / "store.sqlite"
    with FeatureStore(source) as store:
        store.connection.execute("PRAGMA wal_checkpoint(FULL)")
    shutil.copyfile(source, target)
    return target


def estimate(name: str, conversion: float, revenue: float, ci: tuple | None = None,
             conversion_ci: tuple | None = None, ess: float = 500.0) -> OfflineEstimate:
    return OfflineEstimate(
        policy_name=name, conversion_ips=conversion, conversion_snips=conversion,
        conversion_dr=conversion, revenue_ips=revenue, revenue_snips=revenue, revenue_dr=revenue,
        effective_sample_size=ess, true_conversion=None, true_revenue_per_decision=None,
        revenue_per_user_dr=revenue * 5, conversion_lift_pts_ci=conversion_ci,
        revenue_lift_ci=ci,
    )


def test_gate_only_rejects_credibly_worse_candidates() -> None:
    config = RetrainingConfig(store_path="x", min_training_events=100)
    rules = estimate("rule_based", 0.13, 5.0)
    noisy = {"candidate": estimate("candidate", 0.12, 4.7, (-0.25, 0.20), (-3.0, 1.0)),
             "rule_based": rules}
    assert all(check.passed for check in _gate(config, 1_000, noisy, None))

    worse = {"candidate": estimate("candidate", 0.08, 3.0, (-0.50, -0.30), (-6.0, -3.0)),
             "rule_based": rules}
    failed = {c.name for c in _gate(config, 1_000, worse, None) if not c.passed}
    assert failed == {"revenue_vs_rules", "conversion_vs_rules"}

    thin = {"candidate": estimate("candidate", 0.13, 5.0, (-0.1, 0.1), (-1, 1), ess=10.0),
            "rule_based": rules}
    failed = {c.name for c in _gate(config, 50, thin, None) if not c.passed}
    assert failed == {"enough_training_data", "holdout_support"}

    current = {"candidate": estimate("candidate", 0.12, 4.0, (-0.30, -0.10), (-2.0, -1.5)),
               "current": estimate("current", 0.14, 5.5)}
    failed = {c.name for c in _gate(config, 1_000, noisy, current) if not c.passed}
    assert failed == {"revenue_vs_current_model", "conversion_vs_current_model"}


def test_retraining_promotes_then_skips_without_new_data(demo_store: Path, tmp_path: Path) -> None:
    store = fresh_copy(demo_store, tmp_path)
    config = RetrainingConfig(store_path=store, model_dir=tmp_path / "models",
                              min_training_events=500)
    first = retrain(config, clock=datetime(2026, 1, 1, tzinfo=UTC))
    assert first.status == "promoted"
    assert first.model_version == "offer-ts-20260101T000000Z"
    assert first.train_events > first.holdout_events > 0

    registry = ModelRegistry(tmp_path / "models")
    current = registry.current(CATALOG)
    assert current is not None and current.model_version == first.model_version
    assert (tmp_path / "models" / "versions" / f"{first.model_version}.json").exists()
    assert current.metadata["window_end"] == first.window_end
    assert sum(p.updates for p in current.policy.posteriors.values()) == (
        first.train_events + first.holdout_events
    )

    second = retrain(config)
    assert second.status == "skipped"
    assert "no new feedback" in second.reason
    assert [entry["status"] for entry in registry.history()] == ["promoted", "skipped"]


def test_retraining_skips_when_data_is_thin(tmp_path: Path) -> None:
    empty = retrain(RetrainingConfig(store_path=tmp_path / "empty.sqlite",
                                     model_dir=tmp_path / "models"))
    assert empty.status == "skipped"
    assert "no logged decisions" in empty.reason


def test_retraining_respects_the_minimum(demo_store: Path, tmp_path: Path) -> None:
    store = fresh_copy(demo_store, tmp_path)
    run = retrain(RetrainingConfig(store_path=store, model_dir=tmp_path / "models",
                                   min_training_events=1_000_000))
    assert run.status == "skipped"
    assert ModelRegistry(tmp_path / "models").current(CATALOG) is None


def test_rejected_candidates_are_recorded(demo_store: Path, tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    store = fresh_copy(demo_store, tmp_path)

    def always_fail(*args: object, **kwargs: object) -> list:
        return [retraining.GateCheck("revenue_vs_rules", False, "forced failure")]

    monkeypatch.setattr(retraining, "_gate", always_fail)
    run = retrain(RetrainingConfig(store_path=store, model_dir=tmp_path / "models",
                                   min_training_events=500))
    assert run.status == "rejected"
    registry = ModelRegistry(tmp_path / "models")
    assert registry.current(CATALOG) is None
    assert registry.history()[-1]["status"] == "rejected"


def test_ingest_is_idempotent(tmp_path: Path) -> None:
    decisions = tmp_path / "decisions.jsonl"
    feedback = tmp_path / "feedback.jsonl"
    context = new_user_context("u1", "mobile", "east").to_dict()
    decisions.write_text(
        "\n".join(
            json.dumps({"event_id": f"e{i}", "user_id": "u1", "context": context,
                        "offer_id": "bundle_deal", "propensity": 0.3,
                        "policy_name": "thompson_sampling",
                        "timestamp": f"2025-03-0{i + 1}T10:00:00+00:00"})
            for i in range(3)
        ) + "\n",
        encoding="utf-8",
    )
    feedback.write_text(
        json.dumps({"event_id": "e0", "converted": True, "revenue": 55.0,
                    "timestamp": "2025-03-01T11:00:00+00:00"}) + "\n"
        + json.dumps({"event_id": "ghost", "converted": False, "revenue": 0.0,
                      "timestamp": "2025-03-01T11:00:00+00:00"}) + "\n",
        encoding="utf-8",
    )
    with FeatureStore() as store:
        first = ingest_service_logs(store, decisions, feedback)
        second = ingest_service_logs(store, decisions, feedback)
        rows = store.connection.execute(
            "SELECT event_id, converted, revenue FROM offer_events ORDER BY event_id"
        ).fetchall()
    assert (first.decisions_inserted, first.feedback_applied, first.feedback_unmatched) == (3, 1, 1)
    assert (second.decisions_inserted, second.feedback_applied) == (0, 0)
    assert [tuple(row) for row in rows] == [("e0", 1, 55.0), ("e1", None, None),
                                            ("e2", None, None)]


def test_service_logs_feed_retraining_and_hot_reload(demo_store: Path, tmp_path: Path) -> None:
    store = fresh_copy(demo_store, tmp_path)
    models = tmp_path / "models"
    first = retrain(RetrainingConfig(store_path=store, model_dir=models, min_training_events=500),
                    clock=datetime(2026, 1, 1, tzinfo=UTC))
    assert first.status == "promoted"

    service_time = datetime.fromisoformat(first.window_end).replace(hour=23, minute=59, second=59)
    app = create_offer_app(
        OfferServiceConfig(
            decision_log_path=tmp_path / "decisions.jsonl",
            feedback_log_path=tmp_path / "feedback.jsonl",
            feature_store_path=store,
            model_artifact_path=models / "latest.json",
        ),
        clock=lambda: service_time,
    )
    with FeatureStore(store) as reader:
        users = [row[0] for row in reader.connection.execute(
            "SELECT DISTINCT user_id FROM user_features LIMIT 300")]

    async def traffic() -> tuple[str, str]:
        rng = np.random.default_rng(0)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://svc") as client:
            before = (await client.get("/model")).json()["model_version"]
            for user in users:
                decision = (await client.post("/decide", json={"user_id": user})).json()
                assert decision["feature_source"] == "feature_store"
                context = app.state.offer_service.feature_store.latest_features(user, service_time)
                converted = bool(rng.random() < TRUTH.conversion_probability(
                    context, decision["offer_id"]))
                await client.post("/feedback", json={"event_id": decision["event_id"],
                                                     "converted": converted})
            after_retrain = retrain(
                RetrainingConfig(
                    store_path=store,
                    model_dir=models,
                    decision_log=tmp_path / "decisions.jsonl",
                    feedback_log=tmp_path / "feedback.jsonl",
                    min_training_events=500,
                    min_comparison_events=50,
                ),
                clock=datetime(2026, 1, 2, tzinfo=UTC),
            )
            assert after_retrain.ingest.decisions_inserted == len(users)
            assert after_retrain.ingest.feedback_applied == len(users)
            assert after_retrain.status == "promoted", after_retrain.reason
            reloaded = (await client.post("/model/reload")).json()["model_version"]
            return before, reloaded

    before, reloaded = asyncio.run(traffic())
    assert before == "offer-ts-20260101T000000Z"
    assert reloaded == "offer-ts-20260102T000000Z"


def test_promotion_can_notify_the_service(demo_store: Path, tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_post(url: str, timeout: float) -> httpx.Response:
        calls.append(url)
        return httpx.Response(200)

    monkeypatch.setattr(retraining.httpx, "post", fake_post)
    run = retrain(RetrainingConfig(store_path=fresh_copy(demo_store, tmp_path),
                                   model_dir=tmp_path / "models", min_training_events=500,
                                   notify_url="http://service:8000/"))
    assert run.service_reloaded is True
    assert calls == ["http://service:8000/model/reload"]


def test_registry_keeps_versions_and_latest(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path)
    assert registry.current(CATALOG) is None
    policy = BayesianLogisticThompsonSampling(CATALOG)
    registry.promote(policy, "v1", {"note": "first"})
    registry.promote(policy, "v2", {"note": "second"})
    assert registry.current(CATALOG).model_version == "v2"
    assert (tmp_path / "versions" / "v1.json").exists()
    assert [entry["model_version"] for entry in registry.history()] == ["v1", "v2"]


def test_retraining_config_validation() -> None:
    with pytest.raises(ValueError):
        RetrainingConfig(store_path="x", holdout_fraction=1.0)
    with pytest.raises(ValueError):
        RetrainingConfig(store_path="x", lookback_days=0)


def test_retrain_cli_bootstraps_and_reports(tmp_path: Path) -> None:
    report = tmp_path / "run.md"
    code = retrain_main(["--store", str(tmp_path / "demo.sqlite"), "--bootstrap-demo-store",
                         "--model-dir", str(tmp_path / "models"), "--report-md", str(report),
                         "--min-training-events", "500", "--demo-days", "10"])
    assert code == 0
    text = report.read_text(encoding="utf-8")
    assert "# Retraining Run" in text
    assert "Promotion gate" in text
    assert (tmp_path / "models" / "latest.json").exists()


def test_ingest_reads_every_task_log_folder(tmp_path: Path) -> None:
    context = new_user_context("u1", "tablet", "west").to_dict()
    for task, event_id in (("task-a", "a1"), ("task-b", "b1")):
        folder = tmp_path / "logs" / task
        folder.mkdir(parents=True)
        (folder / "decisions.jsonl").write_text(
            json.dumps({"event_id": event_id, "user_id": "u1", "context": context,
                        "offer_id": "free_shipping", "propensity": 0.5,
                        "timestamp": "2025-03-02T09:00:00+00:00"}) + "\n",
            encoding="utf-8",
        )
    (tmp_path / "logs" / "task-b" / "feedback.jsonl").write_text(
        json.dumps({"event_id": "a1", "converted": True, "revenue": 38.0,
                    "timestamp": "2025-03-02T10:00:00+00:00"}) + "\n",
        encoding="utf-8",
    )
    with FeatureStore() as store:
        summary = ingest_service_logs(store, tmp_path / "logs", tmp_path / "logs")
    assert (summary.decisions_read, summary.decisions_inserted) == (2, 2)
    assert summary.feedback_applied == 1


def test_ingest_fills_in_revenue_for_unmatched_feedback(tmp_path: Path) -> None:
    context = new_user_context("u1", "mobile", "north").to_dict()
    decisions = tmp_path / "decisions.jsonl"
    feedback = tmp_path / "feedback.jsonl"
    decisions.write_text(json.dumps({
        "event_id": "x1", "user_id": "u1", "context": context, "offer_id": "premium_trial",
        "propensity": 0.4, "timestamp": "2025-03-03T09:00:00+00:00"}) + "\n", encoding="utf-8")
    feedback.write_text(json.dumps({
        "event_id": "x1", "converted": True, "revenue": None, "matched": False,
        "timestamp": "2025-03-03T09:30:00+00:00"}) + "\n", encoding="utf-8")
    with FeatureStore() as store:
        ingest_service_logs(store, decisions, feedback)
        revenue = store.connection.execute(
            "SELECT revenue FROM offer_events WHERE event_id = 'x1'").fetchone()[0]
    assert revenue == CATALOG.revenue("premium_trial")


def test_bootstrap_also_fills_an_empty_store(tmp_path: Path) -> None:
    empty = tmp_path / "service-created.sqlite"
    FeatureStore(empty).close()
    code = retrain_main(["--store", str(empty), "--bootstrap-demo-store", "--demo-days", "6",
                         "--model-dir", str(tmp_path / "models"),
                         "--report-md", str(tmp_path / "run.md"),
                         "--min-training-events", "300"])
    assert code == 0
    with FeatureStore(empty) as store:
        assert store.table_counts()["offer_events"] > 0
