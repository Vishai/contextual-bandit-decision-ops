import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from contextual_bandit_decision_ops.bayesian_policies import BayesianLogisticThompsonSampling
from contextual_bandit_decision_ops.model_store import save_policy_artifact
from contextual_bandit_decision_ops.monitoring_cli import main as monitor_main
from contextual_bandit_decision_ops.offer_features import OfferContext
from contextual_bandit_decision_ops.offer_logs import log_files, read_service_logs
from contextual_bandit_decision_ops.offer_monitoring import (
    MonitoringThresholds,
    analyze_offer_logs,
    population_stability_index,
    split_by_time,
)
from contextual_bandit_decision_ops.offer_simulation import default_conversion_model

TRUTH = default_conversion_model()
CATALOG = TRUTH.catalog
START = datetime(2025, 3, 1, tzinfo=UTC)


def make_logs(
    n: int,
    seed: int,
    start: datetime,
    mobile_share: float = 0.5,
    latency_ms: float = 2.0,
    offer_weights: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2),
    conversion: float = 0.12,
) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    decisions, feedback = [], []
    for i in range(n):
        sessions_30d = int(rng.integers(0, 30))
        context = OfferContext(
            user_id=f"u{seed}-{i % 300}",
            device="mobile" if rng.random() < mobile_share else "desktop",
            region=("north", "south", "east", "west")[int(rng.integers(0, 4))],
            engagement_score=float(rng.uniform()),
            sessions_7d=min(sessions_30d, int(rng.integers(0, 8))),
            sessions_30d=sessions_30d,
            days_since_last_visit=float(rng.exponential(4.0)),
            tenure_days=float(rng.uniform(0, 600)),
            purchases_90d=int(rng.integers(0, 6)),
            revenue_90d=float(rng.uniform(0, 400)),
            avg_order_value=float(rng.uniform(20, 150)),
            days_since_last_purchase=float(rng.uniform(0, 60)) if rng.random() < 0.8 else None,
            offer_impressions_30d=0,
            offer_conversions_30d=0,
        ).to_dict()
        offer = CATALOG.offer_ids[int(rng.choice(5, p=offer_weights))]
        event_id = f"e{seed}-{i}"
        timestamp = (start + timedelta(minutes=i)).isoformat()
        decisions.append({
            "event_id": event_id, "user_id": context["user_id"], "context": context,
            "feature_source": "request", "available_offers": list(CATALOG.offer_ids),
            "offer_id": offer, "propensity": 0.2, "explored": bool(rng.random() < 0.05),
            "decision_latency_ms": float(rng.gamma(4.0, latency_ms / 4.0)),
            "timestamp": timestamp,
        })
        converted = bool(rng.random() < conversion)
        feedback.append({"event_id": event_id, "converted": converted,
                         "revenue": CATALOG.revenue(offer) if converted else 0.0,
                         "timestamp": timestamp})
    return decisions, feedback


def checks_by_name(result) -> dict:
    return {check.name: check for check in result.checks}


def test_psi_separates_stable_and_shifted_samples() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 4_000)
    assert population_stability_index(base, rng.normal(0, 1, 4_000)) < 0.02
    assert population_stability_index(base, rng.normal(1, 1, 4_000)) > 0.5
    counts = rng.poisson(0.2, 4_000)
    assert population_stability_index(counts, rng.poisson(0.2, 4_000)) < 0.02
    assert population_stability_index(counts, rng.poisson(2.0, 4_000)) > 0.5
    assert population_stability_index([1.0] * 50, [1.0] * 50) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        population_stability_index([], [1.0])


def test_stable_traffic_passes_every_check() -> None:
    reference, ref_feedback = make_logs(1_500, seed=1, start=START)
    current, cur_feedback = make_logs(1_500, seed=2, start=START + timedelta(days=2))
    result = analyze_offer_logs(reference, current, ref_feedback + cur_feedback, CATALOG)
    assert result.status == "ok", [c for c in result.checks if c.status != "ok"]
    assert result.reference.feedback_coverage == 1.0
    assert sum(result.current.allocation.values()) == 1_500
    assert all(abs(sum(row.values()) - 1.0) < 1e-9 for row in result.allocation_over_time)


def test_drift_latency_and_allocation_problems_are_flagged() -> None:
    reference, ref_feedback = make_logs(1_500, seed=3, start=START)
    current, cur_feedback = make_logs(
        1_500, seed=4, start=START + timedelta(days=2), mobile_share=0.95, latency_ms=150.0,
        offer_weights=(0.005, 0.6, 0.2, 0.19, 0.005), conversion=0.05,
    )
    result = analyze_offer_logs(reference, current, ref_feedback + cur_feedback[:900], CATALOG)
    checks = checks_by_name(result)
    assert checks["latency_p99"].status == "alert"
    assert checks["latency_budget_rate"].status == "warn"
    assert checks["feature_drift"].status == "alert"
    assert "device" in checks["feature_drift"].detail
    assert checks["allocation_shift"].status in {"warn", "alert"}
    assert checks["offer_starvation"].status == "warn"
    assert checks["feedback_coverage"].status == "warn"
    assert checks["conversion_change"].status == "warn"
    assert checks["revenue_change"].status == "warn"
    assert result.status == "alert"
    device = next(d for d in result.drift if d.feature == "device")
    assert device.statistic == "tvd" and device.value > 0.4


def test_regret_uses_the_model_and_the_truth() -> None:
    reference, feedback = make_logs(400, seed=5, start=START)
    current, more = make_logs(400, seed=6, start=START + timedelta(days=1))
    best = [dict(r, offer_id=TRUTH.best_offer(OfferContext(**r["context"]))) for r in current]
    result = analyze_offer_logs(
        reference, best, feedback + more, CATALOG,
        model=BayesianLogisticThompsonSampling(CATALOG),
        reference_truth=TRUTH, current_truth=TRUTH,
    )
    assert result.current.true_regret_per_decision == pytest.approx(0.0)
    assert result.reference.true_regret_per_decision > 0.0
    assert result.reference.estimated_regret_per_decision is not None
    names = checks_by_name(result)
    assert {"estimated_regret", "true_regret_simulation"} <= set(names)


def test_low_propensities_are_reported() -> None:
    reference, feedback = make_logs(300, seed=7, start=START)
    current, more = make_logs(300, seed=8, start=START + timedelta(days=1))
    current[0]["propensity"] = 0.001
    result = analyze_offer_logs(reference, current, feedback + more, CATALOG,
                                MonitoringThresholds(min_propensity=0.005))
    assert checks_by_name(result)["propensity_health"].status == "warn"


def test_split_by_time_and_fraction() -> None:
    decisions, _ = make_logs(100, seed=9, start=START)
    reference, current = split_by_time(decisions, START + timedelta(minutes=30))
    assert (len(reference), len(current)) == (30, 70)
    reference, current = split_by_time(decisions, reference_fraction=0.25)
    assert (len(reference), len(current)) == (25, 75)
    with pytest.raises(ValueError):
        split_by_time(decisions, reference_fraction=1.0)
    with pytest.raises(ValueError):
        split_by_time(decisions, datetime(2025, 3, 1))
    with pytest.raises(ValueError, match="no decisions"):
        analyze_offer_logs([], decisions, [], CATALOG)


def test_logs_are_read_from_files_or_replica_folders(tmp_path: Path) -> None:
    decisions, feedback = make_logs(20, seed=10, start=START)
    for replica, rows in (("a", decisions[:10]), ("b", decisions[10:])):
        folder = tmp_path / replica
        folder.mkdir()
        (folder / "decisions.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (tmp_path / "a" / "feedback.jsonl").write_text(
        "\n".join(json.dumps(r) for r in feedback) + "\n", encoding="utf-8")
    read_decisions, read_feedback = read_service_logs(tmp_path, tmp_path)
    assert len(read_decisions) == 20 and len(read_feedback) == 20
    assert log_files(tmp_path / "missing.jsonl", "decisions.jsonl") == []
    assert len(log_files(tmp_path / "a" / "decisions.jsonl", "decisions.jsonl")) == 1


def test_monitor_cli_on_real_logs(tmp_path: Path) -> None:
    decisions, feedback = make_logs(600, seed=11, start=START)
    (tmp_path / "decisions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in decisions) + "\n", encoding="utf-8")
    (tmp_path / "feedback.jsonl").write_text(
        "\n".join(json.dumps(r) for r in feedback) + "\n", encoding="utf-8")
    artifact = tmp_path / "model.json"
    save_policy_artifact(BayesianLogisticThompsonSampling(CATALOG), artifact, "v-test")
    report = tmp_path / "report.md"
    code = monitor_main([
        "--decision-log", str(tmp_path / "decisions.jsonl"),
        "--feedback-log", str(tmp_path / "feedback.jsonl"),
        "--model-artifact", str(artifact),
        "--report-md", str(report), "--artifact-json", str(tmp_path / "report.json"),
        "--fail-on-alert",
    ])
    assert code == 0
    text = report.read_text(encoding="utf-8")
    assert "# Offer Service Monitoring" in text
    assert "Regret / decision, model estimate" in text
    payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    # 300 decisions per window is small enough that PSI can drift into warn by chance.
    assert payload["status"] in {"ok", "warn"}


def test_monitor_cli_demo_detects_the_campaign(tmp_path: Path) -> None:
    report = tmp_path / "demo.md"
    code = monitor_main(["--users", "250", "--decisions", "500", "--report-md", str(report),
                         "--artifact-json", str(tmp_path / "demo.json"), "--fail-on-alert"])
    payload = json.loads((tmp_path / "demo.json").read_text(encoding="utf-8"))
    checks = {c["name"]: c["status"] for c in payload["checks"]}
    assert checks["latency_p99"] == "ok"
    assert checks["feature_drift"] in {"warn", "alert"}
    assert code == (1 if payload["status"] == "alert" else 0)
    assert payload["current"]["true_regret_per_decision"] is not None


def test_monitor_cli_without_decisions(tmp_path: Path) -> None:
    (tmp_path / "decisions.jsonl").write_text("", encoding="utf-8")
    assert monitor_main(["--decision-log", str(tmp_path / "decisions.jsonl"),
                         "--report-md", str(tmp_path / "r.md")]) == 2


def test_psi_sees_shifts_in_mostly_zero_features() -> None:
    reference = [0.0] * 950 + [1.0] * 50
    current = [0.0] * 500 + [1.0] * 500
    assert population_stability_index(reference, current) > 0.5
    assert population_stability_index(reference, list(reference)) == pytest.approx(0.0)
    capped = [5.0] * 900 + [1.0] * 100
    assert population_stability_index(capped, [5.0] * 500 + [1.0] * 500) > 0.5
