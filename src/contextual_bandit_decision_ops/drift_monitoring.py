from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np

from .monitoring_metrics import (
    NumericFeatureShift,
    compare_numeric_feature,
    total_variation_distance,
)
from .policies import EpsilonGreedyPolicy, RandomUniformPolicy
from .schemas import UserContext
from .service_log_analysis import (
    ServiceLogAnalysis,
    ServiceLogWindow,
    analyze_service_logs,
)
from .simulation import reward_probability
from .synthetic import generate_user_contexts

CheckStatus = Literal["passed", "warning", "failed"]


@dataclass(frozen=True)
class ObservabilityConfig:
    reference_events: int = 1_000
    current_events: int = 1_000
    seed: int = 42
    n_actions: int = 3
    max_action_tv_distance: float = 0.10
    max_reward_rate_difference: float = 0.03
    max_standardized_mean_shift: float = 0.20
    max_relative_std_shift: float = 0.25
    max_region_tv_distance: float = 0.10
    minimum_healthy_propensity: float = 0.05
    max_missing_feedback_rate: float = 0.20
    min_exploration_rate: float = 0.10
    report_md: Path | str = Path("reports/staging_observability_report.md")
    artifact_json: Path | str = Path("artifacts/staging_observability_report.json")

    def __post_init__(self) -> None:
        if self.reference_events <= 0 or self.current_events <= 0:
            raise ValueError("reference_events and current_events must be positive")
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        unit_interval_fields = (
            self.max_action_tv_distance,
            self.max_region_tv_distance,
            self.minimum_healthy_propensity,
            self.max_missing_feedback_rate,
            self.min_exploration_rate,
        )
        if any(not 0.0 <= value <= 1.0 for value in unit_interval_fields):
            raise ValueError("distribution and rate thresholds must be between 0 and 1")
        if self.minimum_healthy_propensity == 0.0:
            raise ValueError("minimum_healthy_propensity must be positive")
        if self.max_reward_rate_difference < 0.0:
            raise ValueError("max_reward_rate_difference must be non-negative")
        if self.max_standardized_mean_shift < 0.0:
            raise ValueError("max_standardized_mean_shift must be non-negative")
        if self.max_relative_std_shift < 0.0:
            raise ValueError("max_relative_std_shift must be non-negative")
        object.__setattr__(self, "report_md", Path(self.report_md))
        object.__setattr__(self, "artifact_json", Path(self.artifact_json))


@dataclass(frozen=True)
class MonitoringCheck:
    name: str
    status: CheckStatus
    observed: str
    threshold: str
    message: str


@dataclass(frozen=True)
class ObservabilityRun:
    source: str
    reference: ServiceLogAnalysis
    current: ServiceLogAnalysis
    action_tv_distance: float
    reward_rate_difference: float | None
    age_shift: NumericFeatureShift
    engagement_shift: NumericFeatureShift
    region_tv_distance: float
    passed_checks: tuple[MonitoringCheck, ...]
    warnings: tuple[MonitoringCheck, ...]
    failed_checks: tuple[MonitoringCheck, ...]


def _check(
    name: str,
    status: CheckStatus,
    observed: str,
    threshold: str,
    message: str,
) -> MonitoringCheck:
    return MonitoringCheck(
        name=name,
        status=status,
        observed=observed,
        threshold=threshold,
        message=message,
    )


def _warning_or_pass(
    *,
    name: str,
    concerning: bool,
    observed: str,
    threshold: str,
    message: str,
) -> MonitoringCheck:
    return _check(
        name,
        "warning" if concerning else "passed",
        observed,
        threshold,
        message,
    )


def _decision_record(
    prefix: str,
    index: int,
    context: UserContext,
    action: int,
    propensity: float,
    timestamp: datetime,
) -> dict[str, object]:
    return {
        "event_id": f"{prefix}-event-{index:06d}",
        "user_id": f"{prefix}-user-{index:06d}",
        "context": {
            "age": context.age,
            "engagement": context.engagement,
            "region": context.region,
        },
        "action": action,
        "propensity": propensity,
        "policy_name": prefix,
        "timestamp": timestamp.isoformat(),
        "metadata": {"fixture": True},
    }


def generate_deterministic_log_windows(
    config: ObservabilityConfig,
) -> tuple[ServiceLogWindow, ServiceLogWindow]:
    seed_sequences = np.random.SeedSequence(config.seed).spawn(6)
    reference_contexts = generate_user_contexts(
        config.reference_events,
        np.random.default_rng(seed_sequences[0]),
    )
    unshifted_current_contexts = generate_user_contexts(
        config.current_events,
        np.random.default_rng(seed_sequences[1]),
    )
    current_contexts = [
        UserContext(
            age=min(context.age + 4.0, 70.0),
            engagement=min(context.engagement + 0.12, 1.0),
            region=context.region,
        )
        for context in unshifted_current_contexts
    ]

    reference_policy = RandomUniformPolicy()
    current_policy = EpsilonGreedyPolicy(0.1)
    reference_action_rng = np.random.default_rng(seed_sequences[2])
    current_action_rng = np.random.default_rng(seed_sequences[3])
    reference_reward_rng = np.random.default_rng(seed_sequences[4])
    current_reward_rng = np.random.default_rng(seed_sequences[5])
    available_actions = tuple(range(config.n_actions))
    base_timestamp = datetime(2025, 1, 1, tzinfo=UTC)

    reference_decisions: list[dict[str, object]] = []
    reference_feedback: list[dict[str, object]] = []
    for index, context in enumerate(reference_contexts):
        probabilities = reference_policy.action_probabilities(context, available_actions)
        action = reference_policy.choose_action(
            context,
            available_actions,
            reference_action_rng,
        )
        timestamp = base_timestamp + timedelta(minutes=index)
        decision = _decision_record(
            "reference",
            index,
            context,
            action,
            probabilities[action],
            timestamp,
        )
        reference_decisions.append(decision)
        reference_feedback.append(
            {
                "event_id": decision["event_id"],
                "reward": float(
                    reference_reward_rng.random() < reward_probability(context, action)
                ),
                "timestamp": timestamp.isoformat(),
                "metadata": {"fixture": True},
            }
        )

    current_decisions: list[dict[str, object]] = []
    current_feedback: list[dict[str, object]] = []
    for index, context in enumerate(current_contexts):
        probabilities = current_policy.action_probabilities(context, available_actions)
        action = current_policy.choose_action(
            context,
            available_actions,
            current_action_rng,
        )
        timestamp = base_timestamp + timedelta(days=30, minutes=index)
        decision = _decision_record(
            "current",
            index,
            context,
            action,
            probabilities[action],
            timestamp,
        )
        current_decisions.append(decision)
        if index % 4 != 0:
            current_feedback.append(
                {
                    "event_id": decision["event_id"],
                    "reward": float(
                        current_reward_rng.random() < reward_probability(context, action)
                    ),
                    "timestamp": timestamp.isoformat(),
                    "metadata": {"fixture": True},
                }
            )

    return (
        ServiceLogWindow(
            decisions=tuple(reference_decisions),
            feedback=tuple(reference_feedback),
        ),
        ServiceLogWindow(
            decisions=tuple(current_decisions),
            feedback=tuple(current_feedback),
        ),
    )


def analyze_observability(
    config: ObservabilityConfig,
    reference_window: ServiceLogWindow,
    current_window: ServiceLogWindow,
    *,
    source: str,
) -> ObservabilityRun:
    available_actions = tuple(range(config.n_actions))
    reference = analyze_service_logs(
        reference_window,
        available_actions,
        config.minimum_healthy_propensity,
    )
    current = analyze_service_logs(
        current_window,
        available_actions,
        config.minimum_healthy_propensity,
    )
    action_tv_distance = total_variation_distance(
        reference.action_distribution,
        current.action_distribution,
    )
    region_tv_distance = total_variation_distance(
        reference.region_distribution,
        current.region_distribution,
    )
    age_shift = compare_numeric_feature(reference.ages, current.ages)
    engagement_shift = compare_numeric_feature(
        reference.engagements,
        current.engagements,
    )
    reward_rate_difference = (
        abs(current.counters.average_observed_reward - reference.counters.average_observed_reward)
        if current.counters.average_observed_reward is not None
        and reference.counters.average_observed_reward is not None
        else None
    )

    checks = [
        _warning_or_pass(
            name="action_distribution_shift",
            concerning=action_tv_distance > config.max_action_tv_distance,
            observed=f"{action_tv_distance:.4f}",
            threshold=f"<= {config.max_action_tv_distance:.4f}",
            message="Total variation distance compares reference and current action shares.",
        ),
        (
            _check(
                "reward_rate_shift",
                "failed",
                "N/A",
                f"<= {config.max_reward_rate_difference:.4f}",
                "Reward shift cannot be computed without feedback in both windows.",
            )
            if reward_rate_difference is None
            else _warning_or_pass(
                name="reward_rate_shift",
                concerning=reward_rate_difference > config.max_reward_rate_difference,
                observed=f"{reward_rate_difference:.4f}",
                threshold=f"<= {config.max_reward_rate_difference:.4f}",
                message="Absolute reward-rate change compares feedback windows.",
            )
        ),
        _warning_or_pass(
            name="context_age_shift",
            concerning=(
                age_shift.standardized_mean_difference > config.max_standardized_mean_shift
                or age_shift.relative_std_difference > config.max_relative_std_shift
            ),
            observed=(
                f"mean_z={age_shift.standardized_mean_difference:.3f}, "
                f"std_rel={age_shift.relative_std_difference:.3f}"
            ),
            threshold=(
                f"mean_z <= {config.max_standardized_mean_shift:.3f}, "
                f"std_rel <= {config.max_relative_std_shift:.3f}"
            ),
            message="Age drift compares standardized mean and relative standard deviation.",
        ),
        _warning_or_pass(
            name="context_engagement_shift",
            concerning=(
                engagement_shift.standardized_mean_difference > config.max_standardized_mean_shift
                or engagement_shift.relative_std_difference > config.max_relative_std_shift
            ),
            observed=(
                f"mean_z={engagement_shift.standardized_mean_difference:.3f}, "
                f"std_rel={engagement_shift.relative_std_difference:.3f}"
            ),
            threshold=(
                f"mean_z <= {config.max_standardized_mean_shift:.3f}, "
                f"std_rel <= {config.max_relative_std_shift:.3f}"
            ),
            message="Engagement drift compares standardized mean and relative standard deviation.",
        ),
        _warning_or_pass(
            name="context_region_shift",
            concerning=region_tv_distance > config.max_region_tv_distance,
            observed=f"{region_tv_distance:.4f}",
            threshold=f"<= {config.max_region_tv_distance:.4f}",
            message="Total variation distance compares region distributions.",
        ),
        _check(
            "propensity_validity",
            "failed" if current.propensity_health.invalid_count else "passed",
            str(current.propensity_health.invalid_count),
            "0 invalid values",
            "Propensities must be finite and in (0, 1].",
        ),
        _warning_or_pass(
            name="low_propensity",
            concerning=current.propensity_health.low_count > 0,
            observed=(
                f"{current.propensity_health.low_count} ({current.propensity_health.low_rate:.2%})"
            ),
            threshold=(f"0 below {config.minimum_healthy_propensity:.4f}"),
            message="Very small propensities can make importance weights unstable.",
        ),
        _warning_or_pass(
            name="missing_feedback_rate",
            concerning=(current.counters.missing_feedback_rate > config.max_missing_feedback_rate),
            observed=f"{current.counters.missing_feedback_rate:.2%}",
            threshold=f"<= {config.max_missing_feedback_rate:.2%}",
            message="Missing feedback is the share of decision ids without an outcome.",
        ),
        _warning_or_pass(
            name="low_exploration",
            concerning=(current.counters.exploration_rate < config.min_exploration_rate),
            observed=f"{current.counters.exploration_rate:.2%}",
            threshold=f">= {config.min_exploration_rate:.2%}",
            message="Exploration is one minus the largest current action share.",
        ),
        _check(
            "service_counters",
            "passed",
            (
                f"decisions={current.counters.decisions}, "
                f"feedback={current.counters.feedback_records}"
            ),
            "informational",
            "Local API decision and feedback counters were summarized.",
        ),
    ]
    return ObservabilityRun(
        source=source,
        reference=reference,
        current=current,
        action_tv_distance=action_tv_distance,
        reward_rate_difference=reward_rate_difference,
        age_shift=age_shift,
        engagement_shift=engagement_shift,
        region_tv_distance=region_tv_distance,
        passed_checks=tuple(check for check in checks if check.status == "passed"),
        warnings=tuple(check for check in checks if check.status == "warning"),
        failed_checks=tuple(check for check in checks if check.status == "failed"),
    )


def run_observability(config: ObservabilityConfig) -> ObservabilityRun:
    reference_window, current_window = generate_deterministic_log_windows(config)
    return analyze_observability(
        config,
        reference_window,
        current_window,
        source="deterministic staging fixture logs",
    )
