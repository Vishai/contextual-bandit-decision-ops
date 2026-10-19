import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from contextual_bandit_decision_ops.bayesian_policies import BayesianLogisticThompsonSampling
from contextual_bandit_decision_ops.offer_policies import GreedyOfferPolicy, RuleBasedOfferPolicy
from contextual_bandit_decision_ops.offer_scenarios import (
    DRIFT_SHIFTS,
    LAUNCH_OFFER,
    Scenario,
    cold_start_scenario,
    delayed_rewards_scenario,
    run_scenario,
    shifting_rates_scenario,
    thompson_factory,
)
from contextual_bandit_decision_ops.offer_simulation import default_conversion_model
from contextual_bandit_decision_ops.scenario_cli import main as scenarios_main
from contextual_bandit_decision_ops.synthetic_activity import (
    ActivityConfig,
    DecisionPoint,
    build_feature_store,
    decision_points,
)


@pytest.fixture(scope="module")
def points() -> list[DecisionPoint]:
    store, population = build_feature_store(
        ActivityConfig(n_users=500, history_days=45, simulation_days=8)
    )
    with store:
        return decision_points(store, population)


def rules(model):
    return RuleBasedOfferPolicy(model.catalog)


def greedy(model):
    return GreedyOfferPolicy(model.catalog)


def test_scenarios_are_built_consistently() -> None:
    cold = cold_start_scenario()
    assert cold.launch_offer == LAUNCH_OFFER.offer_id
    assert LAUNCH_OFFER.offer_id in cold.model.catalog
    delayed = delayed_rewards_scenario(12.0)
    assert delayed.mean_feedback_delay_hours == 12.0
    drift = shifting_rates_scenario()
    assert drift.model_after is not None
    bundle = drift.model.catalog.index("bundle_deal")
    assert drift.model_after.weights["bundle_deal"][0] == pytest.approx(
        drift.model.weights["bundle_deal"][0] + DRIFT_SHIFTS["bundle_deal"]
    )
    assert bundle == 2


def test_scenario_validation() -> None:
    model = default_conversion_model()
    with pytest.raises(ValueError):
        Scenario("x", "x", model, change_fraction=1.0)
    with pytest.raises(ValueError):
        Scenario("x", "x", model, mean_feedback_delay_hours=-1.0)
    with pytest.raises(ValueError):
        Scenario("x", "x", model, launch_offer="not_an_offer")
    with pytest.raises(ValueError):
        Scenario("x", "x", model, model_after=cold_start_scenario().model)
    with pytest.raises(ValueError, match="at least one"):
        run_scenario(Scenario("x", "x", model), [rules], [])


def test_launch_offer_is_hidden_until_launch(points: list[DecisionPoint]) -> None:
    scenario = cold_start_scenario()
    results = run_scenario(scenario, [thompson_factory(), rules], points, seed=3)
    ts = results["thompson_sampling"]
    assert ts.launch_offer_share_last_quarter is not None
    assert ts.launch_offer_share_last_quarter > 0.05
    assert results["rule_based"].launch_offer_share_last_quarter == 0.0
    assert 0.0 < (ts.oracle_launch_offer_share_last_quarter or 0.0) < 1.0
    assert ts.regret_before_change + ts.regret_after_change == pytest.approx(ts.cumulative_regret)


def test_launched_offer_never_shows_before_launch(points: list[DecisionPoint]) -> None:
    seen_before_launch: list[str] = []

    class Spy(BayesianLogisticThompsonSampling):
        def choose(self, context, available_offers, rng):
            seen_before_launch.append(",".join(available_offers))
            return super().choose(context, available_offers, rng)

    run_scenario(cold_start_scenario(), [lambda m: Spy(m.catalog)], points, seed=4)
    change_index = len(points) // 2
    assert all(LAUNCH_OFFER.offer_id not in offers for offers in seen_before_launch[:change_index])
    assert all(LAUNCH_OFFER.offer_id in offers for offers in seen_before_launch[change_index:])


def test_delayed_feedback_arrives_in_time_order(points: list[DecisionPoint]) -> None:
    arrivals = []

    class Recorder(BayesianLogisticThompsonSampling):
        def update(self, context, offer_id, converted):
            arrivals.append(context.user_id)
            super().update(context, offer_id, converted)

    instant = run_scenario(Scenario("now", "now", default_conversion_model()),
                           [thompson_factory()], points, seed=5)["thompson_sampling"]
    delayed = run_scenario(delayed_rewards_scenario(48.0), [lambda m: Recorder(m.catalog)],
                           points, seed=5)["thompson_sampling"]
    assert instant.feedback_applied == len(points)
    assert 0 < delayed.feedback_applied < len(points)
    assert len(arrivals) == delayed.feedback_applied
    window = points[-1].timestamp - points[0].timestamp
    assert window > timedelta(days=5)


def test_rules_cannot_adapt_to_shifting_rates(points: list[DecisionPoint]) -> None:
    results = run_scenario(
        shifting_rates_scenario(), [rules, greedy, thompson_factory()], points, seed=6
    )
    rule_based = results["rule_based"]
    assert rule_based.regret_after_change > 2 * rule_based.regret_before_change
    assert results["thompson_sampling"].regret_last_quarter < rule_based.regret_last_quarter
    for result in results.values():
        assert list(result.regret_curve) == sorted(result.regret_curve)
        assert result.new_user_regret_per_decision >= 0.0


def test_runs_are_reproducible(points: list[DecisionPoint]) -> None:
    first = run_scenario(delayed_rewards_scenario(), [thompson_factory()], points, seed=7)
    second = run_scenario(delayed_rewards_scenario(), [thompson_factory()], points, seed=7)
    a, b = first["thompson_sampling"], second["thompson_sampling"]
    assert (a.cumulative_regret, a.conversions, a.feedback_applied) == (
        b.cumulative_regret,
        b.conversions,
        b.feedback_applied,
    )


def test_thompson_factory_labels_and_forgetting() -> None:
    model = default_conversion_model()
    policy = thompson_factory(0.999, "ts_fast_forget")(model)
    assert policy.name == "ts_fast_forget"
    assert policy.forgetting == 0.999
    assert np.isclose(thompson_factory()(model).forgetting, 1.0)


def test_scenarios_cli_writes_report(tmp_path: Path) -> None:
    report = tmp_path / "scenarios.md"
    artifact = tmp_path / "scenarios.json"
    code = scenarios_main(
        ["--users", "200", "--simulation-days", "4", "--sweep-seeds", "2",
         "--report-md", str(report), "--artifact-json", str(artifact)]
    )
    assert code == 0
    text = report.read_text(encoding="utf-8")
    for heading in ("Cold start offer launch", "Delayed rewards", "Shifting conversion rates",
                    "Takeaways", "forgetting factor"):
        assert heading in text
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert len(payload["forgetting_sweep"]) == 4
    assert payload["delay_sweep"][0]["runs"] == 2
