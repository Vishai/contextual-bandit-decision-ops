from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite
from typing import Hashable, Sequence, TypeVar

import numpy as np

Category = TypeVar("Category", bound=Hashable)


@dataclass(frozen=True)
class NumericFeatureSummary:
    mean: float
    std: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class NumericFeatureShift:
    reference: NumericFeatureSummary
    current: NumericFeatureSummary
    absolute_mean_difference: float
    standardized_mean_difference: float
    relative_std_difference: float


@dataclass(frozen=True)
class PropensityHealth:
    count: int
    minimum: float | None
    maximum: float | None
    invalid_count: int
    low_count: int
    low_rate: float


@dataclass(frozen=True)
class ServiceCounterSummary:
    decisions: int
    feedback_records: int
    action_counts: dict[int, int]
    average_observed_reward: float | None
    missing_feedback_count: int
    missing_feedback_rate: float
    exploration_rate: float


def distribution(
    values: Sequence[Category],
    categories: Sequence[Category] | None = None,
) -> dict[Category, float]:
    counts = Counter(values)
    ordered_categories = tuple(categories) if categories is not None else tuple(sorted(counts))
    total = len(values)
    return {
        category: counts.get(category, 0) / total if total else 0.0
        for category in ordered_categories
    }


def total_variation_distance(
    reference: dict[Category, float],
    current: dict[Category, float],
) -> float:
    categories = set(reference) | set(current)
    return 0.5 * sum(
        abs(reference.get(category, 0.0) - current.get(category, 0.0)) for category in categories
    )


def numeric_feature_summary(values: Sequence[float]) -> NumericFeatureSummary:
    if len(values) == 0:
        raise ValueError("numeric feature values must not be empty")
    array = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(array)):
        raise ValueError("numeric feature values must be finite")
    return NumericFeatureSummary(
        mean=float(array.mean()),
        std=float(array.std()),
        minimum=float(array.min()),
        maximum=float(array.max()),
    )


def compare_numeric_feature(
    reference_values: Sequence[float],
    current_values: Sequence[float],
) -> NumericFeatureShift:
    reference = numeric_feature_summary(reference_values)
    current = numeric_feature_summary(current_values)
    absolute_mean_difference = abs(current.mean - reference.mean)
    scale = reference.std if reference.std > 0.0 else 1.0
    std_scale = reference.std if reference.std > 0.0 else 1.0
    return NumericFeatureShift(
        reference=reference,
        current=current,
        absolute_mean_difference=absolute_mean_difference,
        standardized_mean_difference=absolute_mean_difference / scale,
        relative_std_difference=abs(current.std - reference.std) / std_scale,
    )


def reward_rate(rewards: Sequence[float]) -> float | None:
    if len(rewards) == 0:
        return None
    reward_array = np.asarray(rewards, dtype=float)
    if np.any(~np.isfinite(reward_array)):
        raise ValueError("rewards must be finite")
    return float(reward_array.mean())


def propensity_health(
    propensities: Sequence[float],
    minimum_healthy_propensity: float,
) -> PropensityHealth:
    if not 0.0 < minimum_healthy_propensity <= 1.0:
        raise ValueError("minimum_healthy_propensity must be in (0, 1]")
    values = [float(value) for value in propensities]
    valid_values = [value for value in values if isfinite(value) and 0.0 < value <= 1.0]
    invalid_count = len(values) - len(valid_values)
    low_count = sum(value < minimum_healthy_propensity for value in valid_values)
    return PropensityHealth(
        count=len(values),
        minimum=min(valid_values) if valid_values else None,
        maximum=max(valid_values) if valid_values else None,
        invalid_count=invalid_count,
        low_count=low_count,
        low_rate=low_count / len(values) if values else 0.0,
    )


def service_counter_summary(
    decision_event_ids: Sequence[str],
    actions: Sequence[int],
    feedback_event_ids: Sequence[str],
    rewards: Sequence[float],
    available_actions: Sequence[int],
) -> ServiceCounterSummary:
    if len(decision_event_ids) != len(actions):
        raise ValueError("decision ids and actions must have equal length")
    if len(feedback_event_ids) != len(rewards):
        raise ValueError("feedback ids and rewards must have equal length")

    decision_ids = set(decision_event_ids)
    feedback_ids = set(feedback_event_ids)
    missing_feedback_count = len(decision_ids - feedback_ids)
    action_counts_counter = Counter(actions)
    action_counts = {action: action_counts_counter.get(action, 0) for action in available_actions}
    decisions = len(decision_event_ids)
    largest_action_share = max(action_counts.values()) / decisions if decisions else 0.0
    return ServiceCounterSummary(
        decisions=decisions,
        feedback_records=len(feedback_event_ids),
        action_counts=action_counts,
        average_observed_reward=reward_rate(rewards),
        missing_feedback_count=missing_feedback_count,
        missing_feedback_rate=missing_feedback_count / decisions if decisions else 0.0,
        exploration_rate=1.0 - largest_action_share if decisions else 0.0,
    )
