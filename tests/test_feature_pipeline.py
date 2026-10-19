from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from contextual_bandit_decision_ops.feature_pipeline_cli import main as features_main
from contextual_bandit_decision_ops.feature_store import (
    FeatureStore,
    PurchaseRecord,
    SessionRecord,
    UserRecord,
    from_sql_timestamp,
    to_sql_timestamp,
)
from contextual_bandit_decision_ops.synthetic_activity import (
    ActivityConfig,
    build_feature_store,
    decision_points,
    generate_population,
)

AS_OF = datetime(2025, 3, 10, tzinfo=UTC)


def at(day: int, hour: int = 12) -> datetime:
    return datetime(2025, 3, day, hour, tzinfo=UTC)


@pytest.fixture
def store() -> FeatureStore:
    feature_store = FeatureStore()
    feature_store.add_users(
        [
            UserRecord("buyer", datetime(2025, 1, 1, tzinfo=UTC), "mobile", "east"),
            UserRecord("browser", datetime(2025, 2, 20, tzinfo=UTC), "desktop", "west"),
            UserRecord("future", at(10, 12), "tablet", "north"),
        ]
    )
    feature_store.add_sessions(
        [
            SessionRecord("b1", "buyer", at(1), 10.0, 4),
            SessionRecord("b2", "buyer", at(8), 20.0, 9),
            SessionRecord("b3", "buyer", at(12), 30.0, 12),
            SessionRecord("old", "buyer", datetime(2025, 1, 15, tzinfo=UTC), 5.0, 2),
            SessionRecord("w1", "browser", at(9, 18), 6.0, 3),
        ]
    )
    feature_store.add_purchases(
        [
            PurchaseRecord("p1", "buyer", at(8, 13), 80.0),
            PurchaseRecord("p0", "buyer", datetime(2024, 10, 1, tzinfo=UTC), 40.0),
            PurchaseRecord("p2", "buyer", at(12, 13), 999.0),
        ]
    )
    return feature_store


def test_timestamps_round_trip_as_utc_text() -> None:
    value = datetime(2025, 3, 10, 7, 30, 15, tzinfo=UTC)
    assert to_sql_timestamp(value) == "2025-03-10 07:30:15"
    assert from_sql_timestamp("2025-03-10 07:30:15") == value
    with pytest.raises(ValueError, match="timezone"):
        to_sql_timestamp(datetime(2025, 3, 10))


def test_features_aggregate_only_the_past(store: FeatureStore) -> None:
    features = {context.user_id: context for context in store.compute_features(AS_OF)}
    assert set(features) == {"buyer", "browser"}

    buyer = features["buyer"]
    assert buyer.sessions_30d == 2
    assert buyer.sessions_7d == 1
    assert buyer.days_since_last_visit == pytest.approx(1.5)
    assert buyer.tenure_days == pytest.approx(68.0)
    assert buyer.purchases_90d == 1
    assert buyer.revenue_90d == pytest.approx(80.0)
    assert buyer.avg_order_value == pytest.approx(60.0)
    assert buyer.days_since_last_purchase == pytest.approx(1.0 + 11.0 / 24.0)
    assert buyer.engagement_score == pytest.approx(0.6 * 2 / 20 + 0.4 * 15.0 / 30.0)

    browser = features["browser"]
    assert browser.days_since_last_purchase is None
    assert browser.purchases_90d == 0
    assert browser.avg_order_value == 0.0
    assert browser.days_since_last_visit == pytest.approx(0.25)


def test_offer_conversions_wait_for_feedback_time(store: FeatureStore) -> None:
    context = store.compute_features(AS_OF)[1]
    assert context.user_id == "buyer"
    store.record_offer_decision(
        event_id="e1",
        context=context,
        decided_at=at(9),
        offer_id="bundle_deal",
        policy_name="test",
        propensity=0.5,
    )
    store.record_offer_feedback(event_id="e1", converted=True, revenue=55.0, feedback_at=at(11))
    buyer = {c.user_id: c for c in store.compute_features(AS_OF)}["buyer"]
    assert buyer.offer_impressions_30d == 1
    assert buyer.offer_conversions_30d == 0

    later = {c.user_id: c for c in store.compute_features(at(12))}["buyer"]
    assert later.offer_conversions_30d == 1


def test_feedback_is_recorded_once(store: FeatureStore) -> None:
    context = store.compute_features(AS_OF)[0]
    store.record_offer_decision(
        event_id="e2",
        context=context,
        decided_at=at(9),
        offer_id="free_shipping",
        policy_name="test",
        propensity=0.2,
    )
    store.record_offer_feedback(event_id="e2", converted=False, revenue=0.0, feedback_at=at(9))
    with pytest.raises(KeyError):
        store.record_offer_feedback(event_id="e2", converted=True, revenue=1.0, feedback_at=at(9))
    with pytest.raises(KeyError):
        store.record_offer_feedback(event_id="nope", converted=True, revenue=1.0, feedback_at=at(9))


def test_snapshots_are_idempotent_and_served_by_user(store: FeatureStore) -> None:
    assert store.materialize_features(AS_OF) == 2
    assert store.materialize_features(AS_OF) == 2
    assert store.table_counts()["user_features"] == 2
    store.materialize_features(at(13, 0))
    assert store.snapshot_times() == [AS_OF, at(13, 0)]

    snapshot = store.snapshot(AS_OF)
    assert set(snapshot) == {"buyer", "browser"}
    served = store.latest_features("buyer", at(11))
    assert served == snapshot["buyer"]
    newest = store.latest_features("buyer", at(14))
    assert newest is not None
    assert newest.purchases_90d == 2
    assert store.latest_features("buyer", at(5)) is None
    assert store.latest_features("unknown", at(14)) is None


def test_file_backed_store_persists(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "store.sqlite"
    with FeatureStore(path) as feature_store:
        feature_store.add_users([UserRecord("u", at(1), "mobile", "south")])
    with FeatureStore(path) as reopened:
        assert reopened.table_counts()["users"] == 1


def test_population_is_deterministic_and_offer_independent() -> None:
    config = ActivityConfig(n_users=150, history_days=30, simulation_days=5)
    first = generate_population(config)
    second = generate_population(config)
    assert first == second
    assert len(first.users) == 150
    signups = {user.user_id: user.signup_at for user in first.users}
    assert all(session.started_at >= signups[session.user_id] for session in first.sessions)
    assert all(config.start <= s.started_at < config.end for s in first.decision_sessions())


def test_decision_points_read_the_morning_snapshot() -> None:
    config = ActivityConfig(n_users=200, history_days=30, simulation_days=4, new_user_share=0.3)
    store, population = build_feature_store(config)
    points = decision_points(store, population)
    assert points
    assert [p.timestamp for p in points] == sorted(p.timestamp for p in points)
    assert any(point.is_new_user for point in points)
    for point in points[:200]:
        morning = config.start + timedelta(days=(point.timestamp - config.start).days)
        expected = store.latest_features(point.context.user_id, morning)
        if point.is_new_user:
            assert expected is None
            assert point.context.tenure_days == 0.0
        else:
            assert expected == point.context


def test_activity_config_validation() -> None:
    with pytest.raises(ValueError):
        ActivityConfig(n_users=0)
    with pytest.raises(ValueError):
        ActivityConfig(new_user_share=1.0)
    with pytest.raises(ValueError):
        ActivityConfig(start=datetime(2025, 3, 1))


def test_cli_builds_store_and_report(tmp_path: Path) -> None:
    db = tmp_path / "store.sqlite"
    report = tmp_path / "summary.md"
    args = ["--db", str(db), "--report", str(report), "--users", "120", "--simulation-days", "3"]
    assert features_main(args) == 0
    assert features_main(args) == 0
    text = report.read_text(encoding="utf-8")
    assert "# Feature Pipeline Summary" in text
    assert "| `users` | 120 |" in text
    with FeatureStore(db) as feature_store:
        assert len(feature_store.snapshot_times()) == 3


def test_journal_mode_can_be_set_for_network_filesystems(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with FeatureStore(tmp_path / "default.sqlite") as store:
        mode = store.connection.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.upper() == "WAL"
    monkeypatch.setenv("FEATURE_STORE_JOURNAL_MODE", "delete")
    with FeatureStore(tmp_path / "efs.sqlite") as store:
        mode = store.connection.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.upper() == "DELETE"
        assert store.journal_mode == "DELETE"
    with pytest.raises(ValueError, match="journal_mode"):
        FeatureStore(tmp_path / "bad.sqlite", journal_mode="MEMORY; DROP TABLE users")
