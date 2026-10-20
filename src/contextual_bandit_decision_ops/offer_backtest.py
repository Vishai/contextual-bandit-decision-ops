from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np

from .bayesian_policies import (
    BayesianLogisticThompsonSampling,
    BetaBernoulliThompsonSampling,
    OfferPolicy,
    sigmoid,
)
from .off_policy import estimate_doubly_robust, estimate_ips, estimate_snips
from .offer_evaluation import OfferPolicyResult, compare_offer_policies
from .offer_features import encode_contexts
from .offer_history import LoggedOfferEvent, load_logged_events, log_historical_decisions
from .offer_policies import (
    EpsilonExplorationPolicy,
    GreedyOfferPolicy,
    RandomOfferPolicy,
    RuleBasedOfferPolicy,
    UCBOfferPolicy,
)
from .offer_simulation import ConversionModel, default_conversion_model
from .offers import OfferCatalog
from .synthetic_activity import ActivityConfig, DecisionPoint, build_feature_store

BASELINE = "rule_based"


@dataclass(frozen=True)
class BacktestConfig:
    activity: ActivityConfig = field(default_factory=lambda: ActivityConfig(simulation_days=42))
    train_days: int = 28
    logging_epsilon: float = 0.2
    seed: int = 31
    propensity_samples: int = 500
    bootstrap_samples: int = 300

    def __post_init__(self) -> None:
        if not 0 < self.train_days < self.activity.simulation_days:
            raise ValueError("train_days must leave at least one day for testing")
        if not 0.0 < self.logging_epsilon <= 1.0:
            raise ValueError("logging_epsilon must be in (0, 1]")
        if self.propensity_samples <= 0 or self.bootstrap_samples <= 0:
            raise ValueError("propensity_samples and bootstrap_samples must be positive")


@dataclass(frozen=True)
class OfflineEstimate:
    policy_name: str
    conversion_ips: float
    conversion_snips: float
    conversion_dr: float
    revenue_ips: float
    revenue_snips: float
    revenue_dr: float
    effective_sample_size: float
    true_conversion: float | None
    true_revenue_per_decision: float | None
    revenue_per_user_dr: float
    conversion_lift_pts_ci: tuple[float, float] | None
    revenue_lift_ci: tuple[float, float] | None


@dataclass(frozen=True)
class ReplayEstimate:
    policy_name: str
    accepted_events: int
    conversion_rate: float
    revenue_per_decision: float


@dataclass(frozen=True)
class BacktestRun:
    config: BacktestConfig
    train_events: int
    test_events: int
    test_users: int
    logging_policy: str
    logged_conversion_rate: float
    logged_revenue_per_decision: float
    offline: dict[str, OfflineEstimate]
    replay: dict[str, ReplayEstimate]
    forward: dict[str, OfferPolicyResult]



def _accepts_sample_count(policy: object) -> bool:
    method = getattr(policy, "action_probabilities", None)
    return method is not None and "n_samples" in inspect.signature(method).parameters


def _number(value: float | None) -> float:
    """An estimate the data can't support comes back as NaN, never as a quiet zero,
    so any gate comparing against it fails instead of passing on a fake number."""
    return float("nan") if value is None else float(value)


def target_probability_matrix(
    policy: OfferPolicy,
    events: Sequence[LoggedOfferEvent],
    offers: Sequence[str],
    rng: np.random.Generator,
    n_samples: int = 500,
) -> np.ndarray:
    """Probability the target policy would show each offer at each logged visit.

    The policy only chooses among the offers that were available for that visit,
    so offers the user could never have seen get probability zero.
    """
    if not hasattr(policy, "action_probabilities"):
        raise ValueError(f"{policy.name} must expose action_probabilities")
    extra = {"n_samples": n_samples} if _accepts_sample_count(policy) else {}
    matrix = np.empty((len(events), len(offers)))
    for row, event in enumerate(events):
        probabilities = policy.action_probabilities(  # type: ignore[attr-defined]
            event.context, event.offers(offers), rng, **extra
        )
        matrix[row] = [probabilities.get(offer_id, 0.0) for offer_id in offers]
    return matrix


def train_on_history(
    policy: OfferPolicy,
    events: Sequence[LoggedOfferEvent],
) -> OfferPolicy:
    """Warm start a learner on logged history the way a production retrain would."""
    if isinstance(policy, BayesianLogisticThompsonSampling):
        policy.fit(
            [event.context for event in events],
            [event.offer_id for event in events],
            [event.converted for event in events],
        )
    elif isinstance(policy, BetaBernoulliThompsonSampling):
        for event in events:
            policy.update(event.context, event.offer_id, event.converted)
    return policy


def _bootstrap_lift(
    per_event: np.ndarray,
    baseline: np.ndarray,
    rng: np.random.Generator,
    samples: int,
    relative: bool,
    groups: np.ndarray | None = None,
) -> tuple[float, float]:
    """Paired bootstrap interval for the policy's lift over the baseline.

    With ``groups`` (one id per event, usually the user), whole groups are drawn
    with replacement instead of single events. Visits from the same user are
    correlated, so resampling them one by one makes the interval too narrow.
    """
    if groups is None:
        groups = np.arange(len(per_event))
    _, group_index = np.unique(groups, return_inverse=True)
    n_groups = int(group_index.max()) + 1
    policy_sums = np.bincount(group_index, weights=per_event, minlength=n_groups)
    baseline_sums = np.bincount(group_index, weights=baseline, minlength=n_groups)
    counts = np.bincount(group_index, minlength=n_groups).astype(float)

    draws = rng.integers(0, n_groups, size=(samples, n_groups))
    drawn_counts = counts[draws].sum(axis=1)
    policy_means = policy_sums[draws].sum(axis=1) / drawn_counts
    baseline_means = baseline_sums[draws].sum(axis=1) / drawn_counts
    if relative:
        lifts = policy_means / baseline_means - 1.0
    else:
        lifts = (policy_means - baseline_means) * 100.0
    low, high = np.percentile(lifts, [2.5, 97.5])
    return float(low), float(high)


def evaluate_offline(
    policies: Sequence[OfferPolicy],
    events: Sequence[LoggedOfferEvent],
    reward_model: BayesianLogisticThompsonSampling,
    catalog: OfferCatalog,
    rng: np.random.Generator,
    truth: ConversionModel | None = None,
    propensity_samples: int = 500,
    bootstrap_samples: int = 300,
    baseline: str = BASELINE,
) -> dict[str, OfflineEstimate]:
    """Estimate each frozen policy's conversion and revenue on held-out logs.

    IPS and SNIPS only use logged outcomes and propensities. Doubly robust adds a
    reward model (the posterior mean of a logistic fit on the training window) to
    cut variance. When the logs come from the simulator, pass ``truth`` and every
    estimate is also checked against the true expected value of the same policy.
    Production retraining has no truth, so those fields stay None. Lift
    intervals are paired bootstrap intervals against the ``baseline`` policy.
    """
    if not events:
        raise ValueError("there are no held-out events to evaluate on")
    offers = catalog.offer_ids
    values = catalog.revenue_vector()
    logged_index = np.array([offers.index(event.offer_id) for event in events])
    rows = np.arange(len(events))
    propensities = np.array([event.propensity for event in events])
    converted = np.array([float(event.converted) for event in events])
    revenue = np.array([event.revenue for event in events])

    features = encode_contexts([event.context for event in events])
    predicted = np.column_stack(
        [sigmoid(features @ reward_model.posteriors[offer_id].mean) for offer_id in offers]
    )
    true_rates = (
        np.vstack([truth.conversion_probabilities(event.context, offers) for event in events])
        if truth is not None
        else None
    )
    users_per_decision = len({event.context.user_id for event in events}) / len(events)

    per_event_conversion: dict[str, np.ndarray] = {}
    per_event_revenue: dict[str, np.ndarray] = {}
    partial: dict[str, dict[str, float | None]] = {}
    for policy in policies:
        target = target_probability_matrix(policy, events, offers, rng, propensity_samples)
        on_logged = target[rows, logged_index]
        weights = on_logged / propensities
        conversion_model_value = (target * predicted).sum(axis=1)
        revenue_model_value = (target * predicted * values).sum(axis=1)
        logged_conversion_prediction = predicted[rows, logged_index]
        logged_revenue_prediction = logged_conversion_prediction * values[logged_index]
        per_event_conversion[policy.name] = conversion_model_value + weights * (
            converted - logged_conversion_prediction
        )
        per_event_revenue[policy.name] = revenue_model_value + weights * (
            revenue - logged_revenue_prediction
        )
        revenue_dr = estimate_doubly_robust(
            revenue, propensities, on_logged, revenue_model_value, logged_revenue_prediction
        )
        partial[policy.name] = {
            "conversion_ips": _number(estimate_ips(converted, propensities, on_logged).value),
            "conversion_snips": _number(estimate_snips(converted, propensities, on_logged).value),
            "conversion_dr": _number(
                estimate_doubly_robust(
                    converted,
                    propensities,
                    on_logged,
                    conversion_model_value,
                    logged_conversion_prediction,
                ).value
            ),
            "revenue_ips": _number(estimate_ips(revenue, propensities, on_logged).value),
            "revenue_snips": _number(estimate_snips(revenue, propensities, on_logged).value),
            "revenue_dr": _number(revenue_dr.value),
            "effective_sample_size": revenue_dr.effective_sample_size or 0.0,
            "true_conversion": (
                float((target * true_rates).sum(axis=1).mean()) if true_rates is not None else None
            ),
            "true_revenue_per_decision": (
                float((target * true_rates * values).sum(axis=1).mean())
                if true_rates is not None
                else None
            ),
        }

    baseline_conversion = per_event_conversion.get(baseline)
    baseline_revenue = per_event_revenue.get(baseline)
    estimates: dict[str, OfflineEstimate] = {}
    for name, numbers in partial.items():
        conversion_ci = revenue_ci = None
        if baseline_conversion is not None and baseline_revenue is not None and name != baseline:
            conversion_ci = _bootstrap_lift(
                per_event_conversion[name], baseline_conversion, rng, bootstrap_samples, False
            )
            revenue_ci = _bootstrap_lift(
                per_event_revenue[name], baseline_revenue, rng, bootstrap_samples, True
            )
        estimates[name] = OfflineEstimate(
            policy_name=name,
            revenue_per_user_dr=numbers["revenue_dr"] / users_per_decision,
            conversion_lift_pts_ci=conversion_ci,
            revenue_lift_ci=revenue_ci,
            **numbers,
        )
    return estimates


def replay_learning_policies(
    factories: Sequence[Callable[[], OfferPolicy]],
    events: Sequence[LoggedOfferEvent],
    catalog: OfferCatalog,
    minimum_propensity: float,
    rng: np.random.Generator,
) -> dict[str, ReplayEstimate]:
    """Li et al. replay, adjusted for non-uniform logging.

    Each learner starts empty and walks the history in time order. When it picks
    the offer that was actually logged, the event is kept with probability
    ``minimum_propensity / propensity``. That thins the data so the kept events
    look like they came from uniform logging. Only kept events feed the learner.
    """
    offers = catalog.offer_ids
    results: dict[str, ReplayEstimate] = {}
    for factory in factories:
        policy = factory()
        accepted = conversions = 0
        revenue = 0.0
        for event in events:
            choice = policy.choose(event.context, event.offers(offers), rng)
            if choice != event.offer_id:
                continue
            if rng.random() >= minimum_propensity / event.propensity:
                continue
            accepted += 1
            conversions += event.converted
            revenue += event.revenue
            if hasattr(policy, "update"):
                policy.update(event.context, event.offer_id, event.converted)  # type: ignore[attr-defined]
        results[policy.name] = ReplayEstimate(
            policy_name=policy.name,
            accepted_events=accepted,
            conversion_rate=conversions / accepted if accepted else 0.0,
            revenue_per_decision=revenue / accepted if accepted else 0.0,
        )
    return results


def candidate_factories(catalog: OfferCatalog) -> list[Callable[[], OfferPolicy]]:
    return [
        lambda: RandomOfferPolicy(catalog),
        lambda: RuleBasedOfferPolicy(catalog),
        lambda: GreedyOfferPolicy(catalog),
        lambda: UCBOfferPolicy(catalog),
        lambda: BayesianLogisticThompsonSampling(catalog),
    ]


def run_backtest(config: BacktestConfig | None = None) -> BacktestRun:
    settings = config or BacktestConfig()
    truth = default_conversion_model()
    catalog = truth.catalog
    rng = np.random.default_rng(settings.seed)
    logging_policy = EpsilonExplorationPolicy(
        RuleBasedOfferPolicy(catalog), catalog, settings.logging_epsilon
    )

    store, population = build_feature_store(settings.activity)
    with store:
        log_historical_decisions(store, population, truth, logging_policy, settings.seed)
        split = settings.activity.start + timedelta(days=settings.train_days)
        history = load_logged_events(store)
        train = load_logged_events(store, end=split)
        test = load_logged_events(store, start=split)

    reward_model = BayesianLogisticThompsonSampling(catalog)
    reward_model.fit(
        [event.context for event in train],
        [event.offer_id for event in train],
        [event.converted for event in train],
    )
    factories = candidate_factories(catalog)
    trained = [train_on_history(factory(), train) for factory in factories]
    offline = evaluate_offline(
        trained,
        test,
        reward_model,
        catalog,
        rng,
        truth=truth,
        propensity_samples=settings.propensity_samples,
        bootstrap_samples=settings.bootstrap_samples,
    )
    replay = replay_learning_policies(
        factories,
        history,
        catalog,
        minimum_propensity=settings.logging_epsilon / len(catalog),
        rng=rng,
    )
    forward_points = [
        DecisionPoint(event.decided_at, event.event_id, event.context, False) for event in test
    ]
    warm = [train_on_history(factory(), train) for factory in factories]
    forward = compare_offer_policies(warm, forward_points, truth, seed=settings.seed + 1)

    return BacktestRun(
        config=settings,
        train_events=len(train),
        test_events=len(test),
        test_users=len({event.context.user_id for event in test}),
        logging_policy=logging_policy.name,
        logged_conversion_rate=float(np.mean([event.converted for event in history])),
        logged_revenue_per_decision=float(np.mean([event.revenue for event in history])),
        offline=offline,
        replay=replay,
        forward=forward,
    )
