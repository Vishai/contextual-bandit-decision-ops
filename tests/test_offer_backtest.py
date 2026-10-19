import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from contextual_bandit_decision_ops.backtest_cli import main as backtest_main
from contextual_bandit_decision_ops.bayesian_policies import (
    BayesianLogisticThompsonSampling,
    BetaBernoulliThompsonSampling,
)
from contextual_bandit_decision_ops.offer_backtest import (
    BacktestConfig,
    evaluate_offline,
    replay_learning_policies,
    run_backtest,
    target_probability_matrix,
    train_on_history,
)
from contextual_bandit_decision_ops.offer_history import (
    LoggedOfferEvent,
    load_logged_events,
    log_historical_decisions,
)
from contextual_bandit_decision_ops.offer_policies import (
    EpsilonExplorationPolicy,
    RandomOfferPolicy,
    RuleBasedOfferPolicy,
)
from contextual_bandit_decision_ops.offer_simulation import default_conversion_model
from contextual_bandit_decision_ops.synthetic_activity import ActivityConfig, build_feature_store

TRUTH = default_conversion_model()
CATALOG = TRUTH.catalog
OFFERS = CATALOG.offer_ids
ACTIVITY = ActivityConfig(n_users=400, history_days=45, simulation_days=10)


@pytest.fixture(scope="module")
def history() -> list[LoggedOfferEvent]:
    store, population = build_feature_store(ACTIVITY)
    with store:
        logging_policy = EpsilonExplorationPolicy(RuleBasedOfferPolicy(CATALOG), CATALOG, 0.3)
        logged = log_historical_decisions(store, population, TRUTH, logging_policy, seed=3)
        events = load_logged_events(store)
        assert logged == len(events) == len(population.decision_sessions())
        return events


def test_history_is_time_ordered_with_valid_propensities(history: list[LoggedOfferEvent]) -> None:
    assert [e.decided_at for e in history] == sorted(e.decided_at for e in history)
    assert {round(e.propensity, 4) for e in history} == {0.06, 0.76}
    assert all(e.revenue == (CATALOG.revenue(e.offer_id) if e.converted else 0.0)
               for e in history)
    assert all(e.policy_name == "rule_based_eps0.30" for e in history)


def test_later_snapshots_see_earlier_offers(history: list[LoggedOfferEvent]) -> None:
    first_day = [e for e in history if e.decided_at < ACTIVITY.start + timedelta(days=1)]
    last_day = [e for e in history if e.decided_at >= ACTIVITY.end - timedelta(days=1)]
    assert all(e.context.offer_impressions_30d == 0 for e in first_day)
    assert any(e.context.offer_impressions_30d > 0 for e in last_day)


def test_logged_window_filters(history: list[LoggedOfferEvent]) -> None:
    store, population = build_feature_store(ACTIVITY)
    with store:
        logging_policy = EpsilonExplorationPolicy(RuleBasedOfferPolicy(CATALOG), CATALOG, 0.3)
        log_historical_decisions(store, population, TRUTH, logging_policy, seed=3)
        split = ACTIVITY.start + timedelta(days=5)
        before = load_logged_events(store, end=split)
        after = load_logged_events(store, start=split)
    assert len(before) + len(after) == len(history)
    assert all(e.decided_at < split for e in before)
    assert all(e.decided_at >= split for e in after)


def test_logging_needs_known_propensities() -> None:
    store, population = build_feature_store(ActivityConfig(n_users=20, simulation_days=1))
    with store, pytest.raises(ValueError, match="action_probabilities"):
        log_historical_decisions(store, population, TRUTH, object(), seed=0)  # type: ignore[arg-type]


def test_target_probabilities_are_distributions(history: list[LoggedOfferEvent]) -> None:
    events = history[:50]
    rng = np.random.default_rng(0)
    for policy in (RandomOfferPolicy(CATALOG), RuleBasedOfferPolicy(CATALOG),
                   BayesianLogisticThompsonSampling(CATALOG)):
        matrix = target_probability_matrix(policy, events, OFFERS, rng, n_samples=200)
        assert matrix.shape == (50, len(OFFERS))
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
        assert np.all(matrix >= 0.0)


def test_training_on_history_warm_starts_learners(history: list[LoggedOfferEvent]) -> None:
    logistic = train_on_history(BayesianLogisticThompsonSampling(CATALOG), history)
    assert sum(p.updates for p in logistic.posteriors.values()) == len(history)
    beta = train_on_history(BetaBernoulliThompsonSampling(CATALOG), history)
    assert sum(beta.observations.values()) == len(history)
    rules = RuleBasedOfferPolicy(CATALOG)
    assert train_on_history(rules, history) is rules


def test_offline_estimates_track_the_truth(history: list[LoggedOfferEvent]) -> None:
    reward_model = train_on_history(BayesianLogisticThompsonSampling(CATALOG), history)
    estimates = evaluate_offline(
        [RandomOfferPolicy(CATALOG), RuleBasedOfferPolicy(CATALOG)],
        history,
        reward_model,
        CATALOG,
        np.random.default_rng(1),
        truth=TRUTH,
        bootstrap_samples=50,
    )
    rules = estimates["rule_based"]
    assert rules.conversion_lift_pts_ci is None
    assert rules.effective_sample_size > 0.5 * len(history)
    # About 1,300 logged visits from repeat users: across seeds the estimates land
    # within roughly 2 points of the truth on either side, so allow for that noise.
    for estimate in estimates.values():
        assert estimate.conversion_dr == pytest.approx(estimate.true_conversion, abs=0.035)
        assert estimate.revenue_dr == pytest.approx(estimate.true_revenue_per_decision, rel=0.25)
    without_truth = evaluate_offline(
        [RuleBasedOfferPolicy(CATALOG)], history, reward_model, CATALOG,
        np.random.default_rng(1), bootstrap_samples=10,
    )["rule_based"]
    assert without_truth.true_conversion is None
    assert without_truth.revenue_dr == pytest.approx(rules.revenue_dr, rel=1e-9)
    random = estimates["random"]
    low, high = random.revenue_lift_ci
    assert low < high < 0.0
    assert random.revenue_dr < rules.revenue_dr


def test_offline_evaluation_needs_events() -> None:
    with pytest.raises(ValueError, match="no held-out events"):
        evaluate_offline([RandomOfferPolicy(CATALOG)], [], BayesianLogisticThompsonSampling(CATALOG),
                         CATALOG, np.random.default_rng(0))


def test_replay_keeps_about_the_minimum_propensity_share(history: list[LoggedOfferEvent]) -> None:
    results = replay_learning_policies(
        [lambda: RandomOfferPolicy(CATALOG), lambda: BayesianLogisticThompsonSampling(CATALOG)],
        history,
        CATALOG,
        minimum_propensity=0.06,
        rng=np.random.default_rng(2),
    )
    for replay in results.values():
        assert replay.accepted_events == pytest.approx(0.06 * len(history), rel=0.35)
        assert 0.0 <= replay.conversion_rate <= 1.0


def test_backtest_runs_end_to_end() -> None:
    config = BacktestConfig(
        activity=ActivityConfig(n_users=300, history_days=45, simulation_days=8),
        train_days=5,
        propensity_samples=100,
        bootstrap_samples=30,
    )
    run = run_backtest(config)
    assert run.train_events > 0 and run.test_events > 0
    expected = {"random", "rule_based", "greedy", "ucb", "thompson_sampling"}
    assert set(run.offline) == set(run.replay) == set(run.forward) == expected
    assert run.forward["rule_based"].decisions == run.test_events


def test_backtest_config_validation() -> None:
    with pytest.raises(ValueError):
        BacktestConfig(train_days=42)
    with pytest.raises(ValueError):
        BacktestConfig(logging_epsilon=0.0)


def test_backtest_cli_writes_report(tmp_path: Path) -> None:
    report = tmp_path / "backtest.md"
    artifact = tmp_path / "backtest.json"
    code = backtest_main(
        ["--users", "250", "--days", "6", "--train-days", "4", "--bootstrap-samples", "20",
         "--report-md", str(report), "--artifact-json", str(artifact)]
    )
    assert code == 0
    text = report.read_text(encoding="utf-8")
    for heading in ("Offline policy evaluation", "Forward simulation", "Replay of learning"):
        assert heading in text
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["offline"]["thompson_sampling"]["effective_sample_size"] > 0
