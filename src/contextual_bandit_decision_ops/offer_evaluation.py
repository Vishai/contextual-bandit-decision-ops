from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .bayesian_policies import LearningOfferPolicy, OfferPolicy
from .offer_simulation import ConversionModel
from .synthetic_activity import DecisionPoint


@dataclass(frozen=True)
class OfferPolicyResult:
    policy_name: str
    decisions: int
    unique_users: int
    conversions: int
    revenue: float
    conversion_rate: float
    revenue_per_decision: float
    revenue_per_user: float
    expected_revenue_per_decision: float
    cumulative_regret: float
    regret_curve: tuple[float, ...]
    allocation: dict[str, int]
    daily_allocation: tuple[dict[str, int], ...]
    latency_p50_ms: float
    latency_p99_ms: float

    @property
    def allocation_share(self) -> dict[str, float]:
        return {offer: count / self.decisions for offer, count in self.allocation.items()}


def _checkpoints(n_events: int, n_checkpoints: int) -> list[int]:
    if n_events == 0:
        return []
    marks = np.linspace(1, n_events, num=min(n_checkpoints, n_events), dtype=int)
    return sorted({int(mark) for mark in marks})


def evaluate_offer_policy(
    policy: OfferPolicy,
    points: Sequence[DecisionPoint],
    model: ConversionModel,
    reward_draws: np.ndarray,
    rng: np.random.Generator,
    n_checkpoints: int = 20,
) -> OfferPolicyResult:
    """Run one policy over the decision stream and score it against the true model.

    Conversions use shared uniform draws, so two policies that pick the same offer
    for the same visit get the same outcome. Regret is measured in expected
    revenue against the best offer for that visit, which is much less noisy than
    realized revenue.
    """
    if len(points) != len(reward_draws):
        raise ValueError("points and reward_draws must have equal length")
    offers = model.catalog.offer_ids
    revenue_by_offer = dict(zip(offers, model.catalog.revenue_vector(), strict=True))
    learns = isinstance(policy, LearningOfferPolicy)
    start_day = points[0].timestamp.date() if points else None

    allocation = {offer_id: 0 for offer_id in offers}
    daily: dict[int, dict[str, int]] = {}
    checkpoints = set(_checkpoints(len(points), n_checkpoints))
    regret_curve: list[float] = []
    latencies = np.empty(len(points))
    conversions = 0
    revenue = 0.0
    expected_revenue = 0.0
    regret = 0.0

    for index, (point, draw) in enumerate(zip(points, reward_draws, strict=True)):
        started = time.perf_counter()
        offer_id = policy.choose(point.context, offers, rng)
        true_values = model.expected_revenue(point.context, offers)
        probability = true_values[offers.index(offer_id)] / revenue_by_offer[offer_id]
        converted = bool(draw < probability)
        if learns:
            policy.update(point.context, offer_id, converted)  # type: ignore[attr-defined]
        latencies[index] = (time.perf_counter() - started) * 1_000.0

        chosen_value = true_values[offers.index(offer_id)]
        expected_revenue += chosen_value
        regret += float(true_values.max() - chosen_value)
        conversions += converted
        revenue += revenue_by_offer[offer_id] * converted
        allocation[offer_id] += 1
        day = (point.timestamp.date() - start_day).days if start_day else 0
        daily.setdefault(day, {o: 0 for o in offers})[offer_id] += 1
        if index + 1 in checkpoints:
            regret_curve.append(regret)

    n_events = len(points)
    unique_users = len({point.context.user_id for point in points})
    return OfferPolicyResult(
        policy_name=policy.name,
        decisions=n_events,
        unique_users=unique_users,
        conversions=conversions,
        revenue=revenue,
        conversion_rate=conversions / n_events if n_events else 0.0,
        revenue_per_decision=revenue / n_events if n_events else 0.0,
        revenue_per_user=revenue / unique_users if unique_users else 0.0,
        expected_revenue_per_decision=expected_revenue / n_events if n_events else 0.0,
        cumulative_regret=regret,
        regret_curve=tuple(regret_curve),
        allocation=allocation,
        daily_allocation=tuple(daily[day] for day in sorted(daily)),
        latency_p50_ms=float(np.percentile(latencies, 50)) if n_events else 0.0,
        latency_p99_ms=float(np.percentile(latencies, 99)) if n_events else 0.0,
    )


def compare_offer_policies(
    policies: Sequence[OfferPolicy],
    points: Sequence[DecisionPoint],
    model: ConversionModel,
    seed: int = 21,
    n_checkpoints: int = 20,
) -> dict[str, OfferPolicyResult]:
    names = [policy.name for policy in policies]
    if not names:
        raise ValueError("at least one policy is required")
    if len(set(names)) != len(names):
        raise ValueError("policy names must be unique")
    streams = np.random.SeedSequence(seed).spawn(1 + len(policies))
    reward_draws = np.random.default_rng(streams[0]).random(len(points))
    return {
        policy.name: evaluate_offer_policy(
            policy,
            points,
            model,
            reward_draws,
            np.random.default_rng(stream),
            n_checkpoints,
        )
        for policy, stream in zip(policies, streams[1:], strict=True)
    }
