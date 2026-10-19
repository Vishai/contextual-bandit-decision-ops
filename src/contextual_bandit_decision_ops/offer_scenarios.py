from __future__ import annotations

import heapq
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from .bayesian_policies import BayesianLogisticThompsonSampling, LearningOfferPolicy, OfferPolicy
from .offer_features import OfferContext
from .offer_policies import GreedyOfferPolicy, RuleBasedOfferPolicy, UCBOfferPolicy
from .offer_simulation import ConversionModel, default_conversion_model
from .offers import Offer
from .synthetic_activity import DecisionPoint

PolicyFactory = Callable[[ConversionModel], OfferPolicy]

LAUNCH_OFFER = Offer("flash_sale", "48-hour flash sale on favourites", 48.0)
LAUNCH_OFFER_COEFFICIENTS = {
    "bias": -3.6,
    "device_mobile": 0.7,
    "engagement_score": 1.2,
    "sessions_7d_log": 0.6,
    "region_east": 0.4,
}
DRIFT_SHIFTS = {
    "bundle_deal": -1.2,
    "free_shipping": -0.6,
    "premium_trial": 0.8,
    "loyalty_points": 0.4,
}


@dataclass(frozen=True)
class Scenario:
    """One simulated situation the bandit has to cope with.

    ``model_after`` replaces the true conversion model from the change point on
    (shifting conversion rates). ``launch_offer`` is unavailable before the
    change point and available after it (cold start for a new offer). With
    ``mean_feedback_delay_hours`` above zero, conversions show up after an
    exponential delay instead of right away (delayed rewards).
    """

    name: str
    description: str
    model: ConversionModel
    model_after: ConversionModel | None = None
    launch_offer: str | None = None
    change_fraction: float = 0.5
    mean_feedback_delay_hours: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.change_fraction < 1.0:
            raise ValueError("change_fraction must be between 0 and 1")
        if self.mean_feedback_delay_hours < 0.0:
            raise ValueError("mean_feedback_delay_hours must be non-negative")
        if self.launch_offer is not None and self.launch_offer not in self.model.catalog:
            raise ValueError("launch_offer must exist in the model catalog")
        if self.model_after is not None and (
            self.model_after.catalog.offer_ids != self.model.catalog.offer_ids
        ):
            raise ValueError("model_after must use the same catalog")


@dataclass(frozen=True)
class ScenarioPolicyResult:
    policy_name: str
    decisions: int
    conversions: int
    feedback_applied: int
    conversion_rate: float
    revenue_per_user: float
    cumulative_regret: float
    regret_before_change: float
    regret_after_change: float
    regret_last_quarter: float
    new_user_regret_per_decision: float
    returning_user_regret_per_decision: float
    launch_offer_share_last_quarter: float | None
    oracle_launch_offer_share_last_quarter: float | None
    regret_curve: tuple[float, ...]
    latency_p99_ms: float


def _is_new_user(context: OfferContext) -> bool:
    return context.tenure_days < 7.0


def run_scenario(
    scenario: Scenario,
    factories: Sequence[PolicyFactory],
    points: Sequence[DecisionPoint],
    seed: int = 41,
    n_checkpoints: int = 20,
) -> dict[str, ScenarioPolicyResult]:
    if not points:
        raise ValueError("a scenario needs at least one decision point")
    n_events = len(points)
    change_index = int(n_events * scenario.change_fraction)
    last_quarter = int(n_events * 0.75)
    catalog = scenario.model.catalog
    all_offers = catalog.offer_ids
    early_offers = tuple(o for o in all_offers if o != scenario.launch_offer)
    revenue_by_offer = dict(zip(all_offers, catalog.revenue_vector(), strict=True))
    streams = np.random.SeedSequence(seed).spawn(2 + len(factories))
    reward_draws = np.random.default_rng(streams[0]).random(n_events)
    delays_hours = (
        np.random.default_rng(streams[1]).exponential(scenario.mean_feedback_delay_hours, n_events)
        if scenario.mean_feedback_delay_hours > 0
        else np.zeros(n_events)
    )
    checkpoints = {int(m) for m in np.linspace(1, n_events, min(n_checkpoints, n_events))}
    unique_users = len({point.context.user_id for point in points})

    oracle_launch_share: float | None = None
    if scenario.launch_offer is not None:
        late_model = scenario.model_after or scenario.model
        late_best = [late_model.best_offer(p.context, all_offers) for p in points[last_quarter:]]
        oracle_launch_share = late_best.count(scenario.launch_offer) / len(late_best)

    results: dict[str, ScenarioPolicyResult] = {}
    for factory, stream in zip(factories, streams[2:], strict=True):
        policy = factory(scenario.model_after or scenario.model)
        rng = np.random.default_rng(stream)
        learns = isinstance(policy, LearningOfferPolicy)
        pending: list[tuple[datetime, int, OfferContext, str, bool]] = []
        regret = before = after = tail = 0.0
        new_regret = returning_regret = 0.0
        new_count = returning_count = 0
        conversions = applied = launch_picks = 0
        revenue = 0.0
        curve: list[float] = []
        latencies = np.empty(n_events)

        for index, point in enumerate(points):
            while pending and pending[0][0] <= point.timestamp:
                _, _, context, offer_id, converted = heapq.heappop(pending)
                policy.update(context, offer_id, converted)  # type: ignore[attr-defined]
                applied += 1

            changed = index >= change_index
            model = scenario.model_after if changed and scenario.model_after else scenario.model
            offers = all_offers if changed or scenario.launch_offer is None else early_offers
            started = time.perf_counter()
            offer_id = policy.choose(point.context, offers, rng)
            latencies[index] = (time.perf_counter() - started) * 1_000.0

            values = model.expected_revenue(point.context, offers)
            chosen = values[offers.index(offer_id)]
            converted = bool(reward_draws[index] < chosen / revenue_by_offer[offer_id])
            step_regret = float(values.max() - chosen)
            regret += step_regret
            if changed:
                after += step_regret
            else:
                before += step_regret
            if index >= last_quarter:
                tail += step_regret
                launch_picks += offer_id == scenario.launch_offer
            if _is_new_user(point.context):
                new_regret += step_regret
                new_count += 1
            else:
                returning_regret += step_regret
                returning_count += 1
            conversions += converted
            revenue += revenue_by_offer[offer_id] * converted
            if index + 1 in checkpoints:
                curve.append(regret)

            if learns:
                if scenario.mean_feedback_delay_hours > 0:
                    arrives = point.timestamp + timedelta(hours=float(delays_hours[index]))
                    heapq.heappush(pending, (arrives, index, point.context, offer_id, converted))
                else:
                    policy.update(point.context, offer_id, converted)  # type: ignore[attr-defined]
                    applied += 1

        tail_count = n_events - last_quarter
        results[policy.name] = ScenarioPolicyResult(
            policy_name=policy.name,
            decisions=n_events,
            conversions=conversions,
            feedback_applied=applied,
            conversion_rate=conversions / n_events,
            revenue_per_user=revenue / unique_users,
            cumulative_regret=regret,
            regret_before_change=before,
            regret_after_change=after,
            regret_last_quarter=tail,
            new_user_regret_per_decision=new_regret / new_count if new_count else 0.0,
            returning_user_regret_per_decision=(
                returning_regret / returning_count if returning_count else 0.0
            ),
            launch_offer_share_last_quarter=(
                launch_picks / tail_count if scenario.launch_offer is not None else None
            ),
            oracle_launch_offer_share_last_quarter=oracle_launch_share,
            regret_curve=tuple(curve),
            latency_p99_ms=float(np.percentile(latencies, 99)),
        )
    return results


def cold_start_scenario() -> Scenario:
    return Scenario(
        name="cold_start_offer_launch",
        description=(
            "A new flash-sale offer launches halfway through. It is the best offer for "
            "engaged mobile users, but nobody has data on it. New users with no history "
            "also keep arriving the whole time."
        ),
        model=default_conversion_model().with_offer(LAUNCH_OFFER, LAUNCH_OFFER_COEFFICIENTS),
        launch_offer=LAUNCH_OFFER.offer_id,
    )


def delayed_rewards_scenario(mean_delay_hours: float = 24.0) -> Scenario:
    return Scenario(
        name="delayed_rewards",
        description=(
            f"Conversions are reported after an exponential delay averaging "
            f"{mean_delay_hours:g} hours, so policies decide while lots of outcomes are "
            "still pending."
        ),
        model=default_conversion_model(),
        mean_feedback_delay_hours=mean_delay_hours,
    )


def shifting_rates_scenario() -> Scenario:
    model = default_conversion_model()
    return Scenario(
        name="shifting_conversion_rates",
        description=(
            "Halfway through, the bundle deal and free shipping stop working as well and "
            "the premium trial and loyalty points get better, like a competitor launch "
            "or a seasonal change."
        ),
        model=model,
        model_after=model.with_logit_shift(DRIFT_SHIFTS),
    )


def default_factories(forgetting: float = 0.9998) -> list[PolicyFactory]:
    return [
        lambda model: RuleBasedOfferPolicy(model.catalog),
        lambda model: GreedyOfferPolicy(model.catalog),
        lambda model: UCBOfferPolicy(model.catalog),
        lambda model: BayesianLogisticThompsonSampling(model.catalog),
        lambda model: BayesianLogisticThompsonSampling(
            model.catalog, forgetting=forgetting, label="thompson_sampling_forgetting"
        ),
    ]


def thompson_factory(forgetting: float = 1.0, label: str = "") -> PolicyFactory:
    return lambda model: BayesianLogisticThompsonSampling(
        model.catalog, forgetting=forgetting, label=label
    )
