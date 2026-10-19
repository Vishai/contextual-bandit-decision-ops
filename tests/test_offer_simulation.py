import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from contextual_bandit_decision_ops.bayesian_policies import BayesianLogisticThompsonSampling
from contextual_bandit_decision_ops.offer_comparison_cli import main as offers_main
from contextual_bandit_decision_ops.offer_evaluation import compare_offer_policies
from contextual_bandit_decision_ops.offer_features import OfferContext, new_user_context
from contextual_bandit_decision_ops.offer_policies import (
    EpsilonExplorationPolicy,
    GreedyOfferPolicy,
    OracleOfferPolicy,
    RandomOfferPolicy,
    RuleBasedOfferPolicy,
    UCBOfferPolicy,
)
from contextual_bandit_decision_ops.offer_simulation import (
    ConversionModel,
    coefficient_vector,
    default_conversion_model,
)
from contextual_bandit_decision_ops.offers import Offer
from contextual_bandit_decision_ops.synthetic_activity import (
    ActivityConfig,
    build_feature_store,
    decision_points,
)

MODEL = default_conversion_model()
CATALOG = MODEL.catalog
OFFERS = CATALOG.offer_ids


def context(**overrides: object) -> OfferContext:
    values: dict[str, object] = {
        "user_id": "u",
        "device": "desktop",
        "region": "south",
        "engagement_score": 0.4,
        "sessions_7d": 2,
        "sessions_30d": 8,
        "days_since_last_visit": 3.0,
        "tenure_days": 200.0,
        "purchases_90d": 2,
        "revenue_90d": 120.0,
        "avg_order_value": 60.0,
        "days_since_last_purchase": 20.0,
        "offer_impressions_30d": 0,
        "offer_conversions_30d": 0,
    }
    values.update(overrides)
    return OfferContext(**values)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def points() -> list:
    config = ActivityConfig(n_users=300, history_days=45, simulation_days=6)
    store, population = build_feature_store(config)
    with store:
        return decision_points(store, population)


def test_true_rates_are_valid_and_differ_by_user() -> None:
    mobile_newcomer = new_user_context("n", "mobile", "west")
    engaged_spender = context(engagement_score=0.95, avg_order_value=250.0, sessions_30d=30,
                              sessions_7d=8)
    for ctx in (mobile_newcomer, engaged_spender):
        rates = MODEL.conversion_probabilities(ctx)
        assert np.all((rates > 0.0) & (rates < 1.0))
        revenue = MODEL.expected_revenue(ctx)
        assert MODEL.best_offer(ctx) == OFFERS[int(np.argmax(revenue))]
    assert MODEL.best_offer(engaged_spender) == "premium_trial"
    assert MODEL.best_offer(mobile_newcomer) != "premium_trial"


def test_logit_shift_moves_only_the_named_offer() -> None:
    ctx = context()
    shifted = MODEL.with_logit_shift({"bundle_deal": -1.0})
    before = MODEL.conversion_probabilities(ctx)
    after = shifted.conversion_probabilities(ctx)
    bundle = OFFERS.index("bundle_deal")
    assert after[bundle] < before[bundle]
    np.testing.assert_allclose(np.delete(after, bundle), np.delete(before, bundle))


def test_new_offer_can_join_the_world() -> None:
    grown = MODEL.with_offer(Offer("gift_card", "Gift card", 45.0), {"bias": -2.0})
    assert grown.catalog.offer_ids[-1] == "gift_card"
    assert grown.conversion_probability(context(), "gift_card") > 0.0


def test_model_validation() -> None:
    with pytest.raises(ValueError, match="unknown features"):
        coefficient_vector({"shoe_size": 1.0})
    with pytest.raises(ValueError, match="missing true weights"):
        ConversionModel(CATALOG, {"pct10_discount": np.zeros(18)})


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"days_since_last_purchase": None, "purchases_90d": 0, "device": "mobile"},
         "pct10_discount"),
        ({"device": "mobile"}, "free_shipping"),
        ({"purchases_90d": 5}, "bundle_deal"),
        ({"avg_order_value": 150.0, "engagement_score": 0.8}, "premium_trial"),
        ({}, "loyalty_points"),
    ],
)
def test_rule_based_targeting(overrides: dict, expected: str) -> None:
    policy = RuleBasedOfferPolicy(CATALOG)
    assert policy.choose(context(**overrides), OFFERS, np.random.default_rng(0)) == expected
    probabilities = policy.action_probabilities(context(**overrides), OFFERS)
    assert probabilities[expected] == 1.0
    assert sum(probabilities.values()) == 1.0


def test_rule_based_falls_back_when_an_offer_is_paused() -> None:
    policy = RuleBasedOfferPolicy(CATALOG)
    mobile_frequent = context(device="mobile", purchases_90d=6)
    available = [o for o in OFFERS if o != "free_shipping"]
    assert policy.choose(mobile_frequent, available, np.random.default_rng(0)) == "bundle_deal"


def test_random_policy_is_uniform() -> None:
    policy = RandomOfferPolicy(CATALOG)
    probabilities = policy.action_probabilities(context(), OFFERS)
    assert set(probabilities.values()) == {0.2}
    rng = np.random.default_rng(1)
    counts = Counter(policy.choose(context(), OFFERS, rng) for _ in range(5_000))
    assert all(abs(count / 5_000 - 0.2) < 0.03 for count in counts.values())


def test_greedy_never_explores() -> None:
    policy = GreedyOfferPolicy(CATALOG)
    choices = {policy.choose(context(), OFFERS, np.random.default_rng(seed)) for seed in range(20)}
    assert choices == {"premium_trial"}
    assert policy.action_probabilities(context(), OFFERS, np.random.default_rng(0)) == {
        o: float(o == "premium_trial") for o in OFFERS
    }


def test_ucb_tries_the_offer_it_knows_least_about() -> None:
    policy = UCBOfferPolicy(CATALOG, objective="conversion", ucb_width=2.0)
    rng = np.random.default_rng(2)
    ctx = context()
    for _ in range(200):
        for offer_id in OFFERS:
            if offer_id != "loyalty_points":
                policy.update(ctx, offer_id, bool(rng.random() < 0.05))
    assert policy.choose(ctx, OFFERS, rng) == "loyalty_points"
    with pytest.raises(ValueError):
        UCBOfferPolicy(CATALOG, ucb_width=-1.0)


def test_epsilon_wrapper_reports_exact_propensities() -> None:
    policy = EpsilonExplorationPolicy(RuleBasedOfferPolicy(CATALOG), CATALOG, epsilon=0.25)
    ctx = context(device="mobile")
    probabilities = policy.action_probabilities(ctx, OFFERS)
    assert probabilities["free_shipping"] == pytest.approx(0.25 / 5 + 0.75)
    assert min(probabilities.values()) == pytest.approx(0.05)
    assert sum(probabilities.values()) == pytest.approx(1.0)
    rng = np.random.default_rng(3)
    counts = Counter(policy.choose(ctx, OFFERS, rng) for _ in range(20_000))
    for offer_id in OFFERS:
        assert counts[offer_id] / 20_000 == pytest.approx(probabilities[offer_id], abs=0.01)
    assert policy.name == "rule_based_eps0.25"
    with pytest.raises(ValueError):
        EpsilonExplorationPolicy(RuleBasedOfferPolicy(CATALOG), CATALOG, epsilon=0.0)
    with pytest.raises(ValueError):
        EpsilonExplorationPolicy(OracleOfferPolicy(MODEL), CATALOG, epsilon=0.1)


def test_comparison_is_reproducible_and_consistent(points: list) -> None:
    def run() -> dict:
        return compare_offer_policies(
            [RandomOfferPolicy(CATALOG), BayesianLogisticThompsonSampling(CATALOG),
             OracleOfferPolicy(MODEL)],
            points,
            MODEL,
            seed=5,
        )

    first, second = run(), run()
    for name in first:
        a, b = first[name], second[name]
        assert (a.conversions, a.revenue, a.allocation) == (b.conversions, b.revenue, b.allocation)
        assert sum(a.allocation.values()) == a.decisions == len(points)
        assert sum(sum(day.values()) for day in a.daily_allocation) == a.decisions
        assert a.revenue_per_user == pytest.approx(a.revenue / a.unique_users)
        assert list(a.regret_curve) == sorted(a.regret_curve)
    assert first["oracle"].cumulative_regret == 0.0
    assert first["thompson_sampling"].cumulative_regret < first["random"].cumulative_regret


def test_same_choices_get_the_same_outcomes(points: list) -> None:
    results = compare_offer_policies(
        [RuleBasedOfferPolicy(CATALOG),
         EpsilonExplorationPolicy(RuleBasedOfferPolicy(CATALOG), CATALOG, epsilon=1e-12)],
        points,
        MODEL,
    )
    rule, wrapped = results.values()
    assert rule.allocation == wrapped.allocation
    assert rule.conversions == wrapped.conversions


def test_comparison_rejects_duplicate_names(points: list) -> None:
    with pytest.raises(ValueError, match="unique"):
        compare_offer_policies([RandomOfferPolicy(CATALOG)] * 2, points, MODEL)


def test_offers_cli_writes_report_and_artifact(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    artifact = tmp_path / "results.json"
    code = offers_main(
        ["--users", "150", "--simulation-days", "3", "--report-md", str(report),
         "--artifact-json", str(artifact)]
    )
    assert code == 0
    text = report.read_text(encoding="utf-8")
    assert "# Offer Policy Comparison" in text
    assert "Against the rule-based strategy" in text
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert {"rule_based", "thompson_sampling", "ucb", "greedy", "random", "oracle"} <= set(payload)
