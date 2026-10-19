import json
from collections import Counter

import numpy as np
import pytest

from contextual_bandit_decision_ops.bayesian_policies import (
    BayesianLogisticThompsonSampling,
    BetaBernoulliThompsonSampling,
    LearningOfferPolicy,
    sigmoid,
)
from contextual_bandit_decision_ops.offer_features import (
    FEATURE_DIMENSION,
    OfferContext,
    encode_context,
    new_user_context,
)
from contextual_bandit_decision_ops.offers import Offer, OfferCatalog, default_offer_catalog

CATALOG = default_offer_catalog()
OFFERS = CATALOG.offer_ids


def buyer(user_id: str = "u-1", device: str = "desktop", engagement: float = 0.8) -> OfferContext:
    return OfferContext(
        user_id=user_id,
        device=device,
        region="north",
        engagement_score=engagement,
        sessions_7d=3,
        sessions_30d=10,
        days_since_last_visit=2.0,
        tenure_days=300.0,
        purchases_90d=4,
        revenue_90d=320.0,
        avg_order_value=80.0,
        days_since_last_purchase=6.0,
        offer_impressions_30d=2,
        offer_conversions_30d=1,
    )


def random_contexts(n: int, seed: int) -> list[OfferContext]:
    rng = np.random.default_rng(seed)
    contexts = []
    for index in range(n):
        sessions_30d = int(rng.integers(0, 40))
        purchases = int(rng.integers(0, 8))
        contexts.append(
            OfferContext(
                user_id=f"u{index}",
                device=("mobile", "desktop", "tablet")[int(rng.integers(0, 3))],
                region=("north", "south", "east", "west")[int(rng.integers(0, 4))],
                engagement_score=float(rng.uniform()),
                sessions_7d=int(rng.integers(0, sessions_30d + 1)),
                sessions_30d=sessions_30d,
                days_since_last_visit=float(rng.exponential(5.0)),
                tenure_days=float(rng.uniform(0, 700)),
                purchases_90d=purchases,
                revenue_90d=float(purchases * rng.uniform(20, 120)),
                avg_order_value=float(rng.uniform(0, 200)),
                days_since_last_purchase=None if purchases == 0 else float(rng.uniform(0, 90)),
                offer_impressions_30d=3,
                offer_conversions_30d=int(rng.integers(0, 4)),
            )
        )
    return contexts


# Beta-Bernoulli posterior


def test_beta_posterior_counts_successes_and_failures() -> None:
    policy = BetaBernoulliThompsonSampling(CATALOG)
    context = buyer()
    for converted in (True, True, False, True, False, False, False):
        policy.update(context, "bundle_deal", converted)
    assert policy.parameters(context, "bundle_deal") == (4.0, 5.0)
    posterior = policy.posterior(context, "bundle_deal")
    assert posterior.mean == pytest.approx(4 / 9)
    assert posterior.lower < posterior.mean < posterior.upper
    assert posterior.expected_revenue == pytest.approx(4 / 9 * 55.0)
    assert posterior.observations == 7
    assert policy.parameters(context, "free_shipping") == (1.0, 1.0)


def test_beta_posteriors_are_kept_per_segment() -> None:
    policy = BetaBernoulliThompsonSampling(CATALOG, segment_by=("device", "region"))
    policy.update(buyer(device="mobile"), "free_shipping", True)
    assert policy.parameters(buyer(device="mobile"), "free_shipping") == (2.0, 1.0)
    assert policy.parameters(buyer(device="desktop"), "free_shipping") == (1.0, 1.0)
    assert policy.segment(buyer(device="mobile")) == "mobile|north"
    assert BetaBernoulliThompsonSampling(CATALOG, segment_by=()).segment(buyer()) == "all"


def test_beta_discount_fades_old_evidence() -> None:
    policy = BetaBernoulliThompsonSampling(CATALOG, discount=0.5)
    context = buyer()
    for _ in range(3):
        policy.update(context, "pct10_discount", True)
    alpha, beta = policy.parameters(context, "pct10_discount")
    assert alpha == pytest.approx(1 + 0.25 + 0.5 + 1.0)
    assert beta == pytest.approx(1.0)


def test_beta_thompson_sampling_finds_the_best_revenue_offer() -> None:
    rates = {
        "pct10_discount": 0.08,
        "free_shipping": 0.10,
        "bundle_deal": 0.03,
        "premium_trial": 0.02,
        "loyalty_points": 0.12,
    }
    policy = BetaBernoulliThompsonSampling(CATALOG, segment_by=())
    rng = np.random.default_rng(3)
    context = buyer()
    chosen = Counter()
    for _ in range(4_000):
        offer = policy.choose(context, OFFERS, rng)
        chosen[offer] += 1
        policy.update(context, offer, bool(rng.random() < rates[offer]))
    assert chosen.most_common(1)[0][0] == "free_shipping"
    probabilities = policy.action_probabilities(context, OFFERS, rng)
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert max(probabilities, key=probabilities.get) == "free_shipping"


def test_beta_conversion_objective_ignores_revenue() -> None:
    policy = BetaBernoulliThompsonSampling(CATALOG, segment_by=(), objective="conversion")
    context = buyer()
    for _ in range(300):
        policy.update(context, "loyalty_points", True)
        policy.update(context, "premium_trial", False)
    assert policy.choose(context, ["loyalty_points", "premium_trial"], np.random.default_rng(0)) == (
        "loyalty_points"
    )


def test_beta_state_round_trips_through_json() -> None:
    policy = BetaBernoulliThompsonSampling(CATALOG, discount=0.9)
    policy.update(buyer(device="mobile"), "bundle_deal", True)
    policy.update(buyer(device="tablet"), "premium_trial", False)
    payload = json.loads(json.dumps(policy.to_dict()))
    restored = BetaBernoulliThompsonSampling.from_dict(payload, CATALOG)
    assert restored.to_dict() == policy.to_dict()
    with pytest.raises(ValueError):
        BetaBernoulliThompsonSampling.from_dict({"model_type": "other"}, CATALOG)


def test_beta_rejects_bad_settings_and_offers() -> None:
    with pytest.raises(ValueError):
        BetaBernoulliThompsonSampling(CATALOG, prior_alpha=0.0)
    with pytest.raises(ValueError):
        BetaBernoulliThompsonSampling(CATALOG, discount=0.0)
    with pytest.raises(ValueError):
        BetaBernoulliThompsonSampling(CATALOG, objective="clicks")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        BetaBernoulliThompsonSampling(CATALOG).update(buyer(), "missing", True)


# Bayesian logistic posterior


def test_logistic_prior_matches_base_rate() -> None:
    policy = BayesianLogisticThompsonSampling(CATALOG, prior_conversion_rate=0.05)
    assert policy.dimension == FEATURE_DIMENSION
    for offer_id in OFFERS:
        posterior = policy.posteriors[offer_id]
        assert posterior.mean[0] == pytest.approx(np.log(0.05 / 0.95))
        np.testing.assert_allclose(posterior.precision, np.eye(FEATURE_DIMENSION))


def test_a_conversion_raises_the_logit_and_shrinks_uncertainty() -> None:
    policy = BayesianLogisticThompsonSampling(CATALOG)
    context = buyer()
    before_mean, before_std = policy.logit_distribution(context, ["bundle_deal"])
    untouched = policy.posteriors["free_shipping"].mean.copy()
    policy.update(context, "bundle_deal", True)
    after_mean, after_std = policy.logit_distribution(context, ["bundle_deal"])
    assert after_mean[0] > before_mean[0]
    assert after_std[0] < before_std[0]
    np.testing.assert_array_equal(policy.posteriors["free_shipping"].mean, untouched)
    assert policy.posteriors["bundle_deal"].updates == 1


def test_a_non_conversion_lowers_the_logit() -> None:
    policy = BayesianLogisticThompsonSampling(CATALOG, prior_conversion_rate=0.5)
    context = buyer()
    before, _ = policy.logit_distribution(context, ["premium_trial"])
    policy.update(context, "premium_trial", False)
    after, _ = policy.logit_distribution(context, ["premium_trial"])
    assert after[0] < before[0]


def test_online_covariance_stays_the_inverse_of_precision() -> None:
    policy = BayesianLogisticThompsonSampling(CATALOG)
    rng = np.random.default_rng(4)
    for context in random_contexts(400, seed=5):
        policy.update(context, "loyalty_points", bool(rng.random() < 0.2))
    posterior = policy.posteriors["loyalty_points"]
    np.testing.assert_allclose(posterior.covariance, posterior.covariance.T)
    np.testing.assert_allclose(
        posterior.covariance @ posterior.precision, np.eye(FEATURE_DIMENSION), atol=1e-6
    )
    assert np.all(np.linalg.eigvalsh(posterior.precision) > 0)


def _true_weights(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        offer_id: np.r_[-2.5, rng.normal(0.0, 0.9, FEATURE_DIMENSION - 1)] for offer_id in OFFERS
    }


def test_online_updates_learn_the_true_conversion_model() -> None:
    weights = _true_weights(6)
    contexts = random_contexts(6_000, seed=7)
    rng = np.random.default_rng(8)
    policy = BayesianLogisticThompsonSampling(CATALOG)
    for context in contexts:
        offer_id = OFFERS[int(rng.integers(0, len(OFFERS)))]
        probability = sigmoid(weights[offer_id] @ encode_context(context))
        policy.update(context, offer_id, bool(rng.random() < probability))

    holdout = np.array([encode_context(context) for context in random_contexts(500, seed=9)])
    for offer_id in OFFERS:
        truth = sigmoid(holdout @ weights[offer_id])
        learned = sigmoid(holdout @ policy.posteriors[offer_id].mean)
        assert np.mean(np.abs(truth - learned)) < 0.05


def test_batch_fit_matches_truth_and_agrees_with_online_learning() -> None:
    weights = _true_weights(10)
    contexts = random_contexts(5_000, seed=11)
    rng = np.random.default_rng(12)
    offers = [OFFERS[int(rng.integers(0, len(OFFERS)))] for _ in contexts]
    outcomes = [
        bool(rng.random() < sigmoid(weights[o] @ encode_context(c)))
        for c, o in zip(contexts, offers, strict=True)
    ]
    batch = BayesianLogisticThompsonSampling(CATALOG)
    batch.fit(contexts, offers, outcomes)
    online = BayesianLogisticThompsonSampling(CATALOG)
    for context, offer_id, outcome in zip(contexts, offers, outcomes, strict=True):
        online.update(context, offer_id, outcome)

    holdout = np.array([encode_context(c) for c in random_contexts(500, seed=13)])
    for offer_id in OFFERS:
        truth = sigmoid(holdout @ weights[offer_id])
        from_batch = sigmoid(holdout @ batch.posteriors[offer_id].mean)
        from_online = sigmoid(holdout @ online.posteriors[offer_id].mean)
        assert np.mean(np.abs(truth - from_batch)) < 0.05
        assert np.mean(np.abs(from_batch - from_online)) < 0.03
        assert batch.posteriors[offer_id].updates == offers.count(offer_id)
    with pytest.raises(ValueError, match="equal length"):
        batch.fit(contexts, offers[:-1], outcomes)


def test_forgetting_widens_the_posterior_again() -> None:
    steady = BayesianLogisticThompsonSampling(CATALOG)
    fading = BayesianLogisticThompsonSampling(CATALOG, forgetting=0.98)
    for context in random_contexts(300, seed=14):
        steady.update(context, "pct10_discount", False)
        fading.update(context, "pct10_discount", False)
    context = buyer()
    _, steady_std = steady.logit_distribution(context, ["pct10_discount"])
    _, fading_std = fading.logit_distribution(context, ["pct10_discount"])
    assert fading_std[0] > steady_std[0]


def test_thompson_sampling_favours_the_offer_that_converts() -> None:
    policy = BayesianLogisticThompsonSampling(CATALOG)
    context = buyer()
    rng = np.random.default_rng(15)
    for _ in range(200):
        policy.update(context, "bundle_deal", bool(rng.random() < 0.4))
        for other in ("pct10_discount", "free_shipping", "premium_trial", "loyalty_points"):
            policy.update(context, other, bool(rng.random() < 0.02))
    choices = Counter(policy.choose(context, OFFERS, rng) for _ in range(300))
    assert choices.most_common(1)[0][0] == "bundle_deal"
    probabilities = policy.action_probabilities(context, OFFERS, rng)
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities["bundle_deal"] > 0.9


def test_fresh_model_explores_every_offer() -> None:
    policy = BayesianLogisticThompsonSampling(CATALOG, objective="conversion")
    probabilities = policy.action_probabilities(
        new_user_context("n", "mobile", "east"), OFFERS, np.random.default_rng(16)
    )
    assert all(probability > 0.1 for probability in probabilities.values())


def test_zero_exploration_is_deterministic_greedy() -> None:
    policy = BayesianLogisticThompsonSampling(CATALOG, exploration_scale=0.0)
    context = buyer()
    first = policy.choose(context, OFFERS, np.random.default_rng(1))
    assert all(policy.choose(context, OFFERS, np.random.default_rng(seed)) == first
               for seed in range(10))
    assert first == "premium_trial"


def test_posterior_summary_reports_conversion_intervals() -> None:
    policy = BayesianLogisticThompsonSampling(CATALOG)
    summary = policy.posterior_summary(buyer())
    assert [row.offer_id for row in summary] == list(OFFERS)
    for row in summary:
        assert 0.0 < row.lower < row.mean < row.upper < 1.0
        assert row.expected_revenue == pytest.approx(row.mean * CATALOG.revenue(row.offer_id))


def test_logistic_state_round_trips_and_reproduces_decisions() -> None:
    policy = BayesianLogisticThompsonSampling(CATALOG, forgetting=0.999)
    rng = np.random.default_rng(17)
    for context in random_contexts(200, seed=18):
        policy.update(context, OFFERS[int(rng.integers(0, 5))], bool(rng.random() < 0.1))
    payload = json.loads(json.dumps(policy.to_dict()))
    restored = BayesianLogisticThompsonSampling.from_dict(payload, CATALOG)
    contexts = random_contexts(50, seed=19)
    original_choices = [policy.choose(c, OFFERS, np.random.default_rng(i))
                        for i, c in enumerate(contexts)]
    restored_choices = [restored.choose(c, OFFERS, np.random.default_rng(i))
                        for i, c in enumerate(contexts)]
    assert original_choices == restored_choices

    bad = dict(payload, feature_names=["bias", "something_else"])
    with pytest.raises(ValueError, match="different feature set"):
        BayesianLogisticThompsonSampling.from_dict(bad, CATALOG)


def test_new_offers_start_from_the_prior() -> None:
    policy = BayesianLogisticThompsonSampling(CATALOG)
    policy.update(buyer(), "bundle_deal", True)
    grown = OfferCatalog((*CATALOG.offers, Offer("gift_card", "Gift card", 45.0)))
    restored = BayesianLogisticThompsonSampling.from_dict(policy.to_dict(), grown)
    assert restored.posteriors["gift_card"].updates == 0
    assert restored.posteriors["bundle_deal"].updates == 1
    chosen = restored.choose(buyer(), grown.offer_ids, np.random.default_rng(0))
    assert chosen in grown


def test_logistic_rejects_bad_settings() -> None:
    with pytest.raises(ValueError):
        BayesianLogisticThompsonSampling(CATALOG, prior_variance=0.0)
    with pytest.raises(ValueError):
        BayesianLogisticThompsonSampling(CATALOG, prior_conversion_rate=1.0)
    with pytest.raises(ValueError):
        BayesianLogisticThompsonSampling(CATALOG, forgetting=1.5)
    with pytest.raises(ValueError):
        BayesianLogisticThompsonSampling(CATALOG, feature_names=("x",))
    with pytest.raises(ValueError):
        BayesianLogisticThompsonSampling(CATALOG).choose(buyer(), ["missing"],
                                                         np.random.default_rng(0))


def test_policies_follow_the_learning_protocol() -> None:
    assert isinstance(BayesianLogisticThompsonSampling(CATALOG), LearningOfferPolicy)
    assert isinstance(BetaBernoulliThompsonSampling(CATALOG), LearningOfferPolicy)
