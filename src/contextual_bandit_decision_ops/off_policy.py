from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .config import OffPolicyEvaluationConfig, SimulationConfig
from .event_log import events_to_frame
from .learning_policies import CONTEXT_FEATURE_DIMENSION, context_vector
from .policies import (
    EpsilonGreedyPolicy,
    FixedActionPolicy,
    GreedyOraclePolicy,
    ProbabilisticBanditPolicy,
    RandomUniformPolicy,
)
from .schemas import REGIONS, UserContext
from .simulation import reward_probability, simulate_bandit_events

REQUIRED_LOG_COLUMNS = {
    "context_age",
    "context_engagement",
    "context_region",
    "action",
    "reward",
    "propensity",
}


@dataclass(frozen=True)
class EstimatorResult:
    value: float | None
    effective_sample_size: float | None = None
    matched_count: int | None = None


@dataclass(frozen=True)
class TargetPolicyEvaluation:
    policy_name: str
    simulator_value: float
    estimators: dict[str, EstimatorResult]


@dataclass(frozen=True)
class OffPolicyEvaluationRun:
    row_count: int
    behavior_policy: str
    observed_behavior_value: float
    expected_behavior_value: float | None
    policies: dict[str, TargetPolicyEvaluation]


def validate_logged_bandit_data(frame: pd.DataFrame, n_actions: int) -> None:
    missing_columns = REQUIRED_LOG_COLUMNS - set(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"logged data is missing columns: {missing}")
    if frame.empty:
        raise ValueError("logged data must not be empty")
    if frame[list(REQUIRED_LOG_COLUMNS)].isna().any().any():
        raise ValueError("logged data contains missing values")
    if not frame["action"].isin(range(n_actions)).all():
        raise ValueError("logged action is out of range")
    if not frame["reward"].isin([0, 1]).all():
        raise ValueError("logged reward must be 0 or 1")
    if not ((frame["propensity"] > 0.0) & (frame["propensity"] <= 1.0)).all():
        raise ValueError("logged propensity must be greater than 0 and at most 1")
    if not frame["context_age"].between(18.0, 70.0).all():
        raise ValueError("context_age must be between 18 and 70")
    if not frame["context_engagement"].between(0.0, 1.0).all():
        raise ValueError("context_engagement must be between 0 and 1")
    if not frame["context_region"].isin(REGIONS).all():
        raise ValueError(f"context_region must be one of {REGIONS}")


def _contexts_from_frame(frame: pd.DataFrame) -> list[UserContext]:
    return [
        UserContext(
            age=float(row.context_age),
            engagement=float(row.context_engagement),
            region=str(row.context_region),
        )
        for row in frame.itertuples(index=False)
    ]


def _validated_arrays(*arrays: Sequence[float]) -> list[np.ndarray]:
    converted = [np.asarray(array, dtype=float) for array in arrays]
    if not converted or converted[0].size == 0:
        raise ValueError("estimator inputs must not be empty")
    expected_shape = converted[0].shape
    if len(expected_shape) != 1 or any(array.shape != expected_shape for array in converted):
        raise ValueError("estimator inputs must be one-dimensional and equal length")
    return converted


def _effective_sample_size(weights: np.ndarray) -> float:
    squared_weight_sum = float(weights @ weights)
    if squared_weight_sum == 0.0:
        return 0.0
    return float(weights.sum() ** 2 / squared_weight_sum)


def _importance_weights(
    propensities: np.ndarray,
    target_probabilities: np.ndarray,
) -> np.ndarray:
    if np.any(~np.isfinite(propensities)) or np.any(propensities <= 0.0):
        raise ValueError("propensities must be finite and positive")
    if np.any(~np.isfinite(target_probabilities)) or np.any(target_probabilities < 0.0):
        raise ValueError("target probabilities must be finite and non-negative")
    return target_probabilities / propensities


def estimate_direct_logged_average(rewards: Sequence[float]) -> EstimatorResult:
    """Return the naive mean reward observed under the behavior policy."""
    (reward_array,) = _validated_arrays(rewards)
    return EstimatorResult(
        value=float(reward_array.mean()),
        effective_sample_size=float(len(reward_array)),
    )


def estimate_replay_matching(
    rewards: Sequence[float],
    logged_actions: Sequence[int],
    target_actions: Sequence[int],
) -> EstimatorResult:
    """Average rewards where a sampled target action matches the logged action."""
    reward_array, logged_array, target_array = _validated_arrays(
        rewards,
        logged_actions,
        target_actions,
    )
    matched = logged_array == target_array
    matched_count = int(matched.sum())
    value = float(reward_array[matched].mean()) if matched_count else None
    return EstimatorResult(value=value, matched_count=matched_count)


def estimate_ips(
    rewards: Sequence[float],
    propensities: Sequence[float],
    logged_target_probabilities: Sequence[float],
) -> EstimatorResult:
    """Estimate E[pi(a|x) * reward / behavior_propensity]."""
    reward_array, propensity_array, target_probability_array = _validated_arrays(
        rewards,
        propensities,
        logged_target_probabilities,
    )
    weights = _importance_weights(propensity_array, target_probability_array)
    return EstimatorResult(
        value=float(np.mean(weights * reward_array)),
        effective_sample_size=_effective_sample_size(weights),
    )


def estimate_snips(
    rewards: Sequence[float],
    propensities: Sequence[float],
    logged_target_probabilities: Sequence[float],
) -> EstimatorResult:
    """Normalize the IPS reward sum by the sum of importance weights."""
    reward_array, propensity_array, target_probability_array = _validated_arrays(
        rewards,
        propensities,
        logged_target_probabilities,
    )
    weights = _importance_weights(propensity_array, target_probability_array)
    weight_sum = float(weights.sum())
    value = float(weights @ reward_array / weight_sum) if weight_sum > 0.0 else None
    return EstimatorResult(
        value=value,
        effective_sample_size=_effective_sample_size(weights),
    )


def estimate_doubly_robust(
    rewards: Sequence[float],
    propensities: Sequence[float],
    logged_target_probabilities: Sequence[float],
    target_model_predictions: Sequence[float],
    logged_model_predictions: Sequence[float],
) -> EstimatorResult:
    """Add an importance-weighted residual correction to target model predictions."""
    (
        reward_array,
        propensity_array,
        target_probability_array,
        target_prediction_array,
        logged_prediction_array,
    ) = _validated_arrays(
        rewards,
        propensities,
        logged_target_probabilities,
        target_model_predictions,
        logged_model_predictions,
    )
    weights = _importance_weights(propensity_array, target_probability_array)
    corrections = weights * (reward_array - logged_prediction_array)
    return EstimatorResult(
        value=float(np.mean(target_prediction_array + corrections)),
        effective_sample_size=_effective_sample_size(weights),
    )


@dataclass
class LinearRewardModel:
    """Simple per-action ridge model used by the doubly robust estimator."""

    n_actions: int
    regularization: float = 1.0

    def __post_init__(self) -> None:
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive")
        if self.regularization <= 0.0:
            raise ValueError("regularization must be positive")
        self.coefficients = np.zeros(
            (self.n_actions, CONTEXT_FEATURE_DIMENSION),
            dtype=float,
        )
        self.is_fitted = False

    def fit(
        self, contexts: Sequence[UserContext], actions: Sequence[int], rewards: Sequence[float]
    ) -> None:
        if len(contexts) == 0 or len(contexts) != len(actions) or len(actions) != len(rewards):
            raise ValueError("reward model inputs must be non-empty and equal length")
        feature_matrix = np.stack([context_vector(context) for context in contexts])
        action_array = np.asarray(actions, dtype=int)
        reward_array = np.asarray(rewards, dtype=float)
        identity = np.eye(CONTEXT_FEATURE_DIMENSION)

        for action in range(self.n_actions):
            action_mask = action_array == action
            action_features = feature_matrix[action_mask]
            action_rewards = reward_array[action_mask]
            precision = self.regularization * identity + action_features.T @ action_features
            reward_vector = action_features.T @ action_rewards
            self.coefficients[action] = np.linalg.solve(precision, reward_vector)
        self.is_fitted = True

    def predict(self, context: UserContext, action: int) -> float:
        if not self.is_fitted:
            raise ValueError("reward model must be fitted before prediction")
        if action not in range(self.n_actions):
            raise ValueError("action is out of range")
        prediction = self.coefficients[action] @ context_vector(context)
        return float(np.clip(prediction, 0.0, 1.0))


def default_target_policies(
    config: OffPolicyEvaluationConfig,
) -> tuple[ProbabilisticBanditPolicy, ...]:
    return (
        RandomUniformPolicy(),
        FixedActionPolicy(config.fixed_action),
        EpsilonGreedyPolicy(config.epsilon),
        GreedyOraclePolicy(),
    )


def _policy_probability_matrix(
    policy: ProbabilisticBanditPolicy,
    contexts: Sequence[UserContext],
    available_actions: tuple[int, ...],
) -> np.ndarray:
    probability_rows: list[list[float]] = []
    for context in contexts:
        probabilities = policy.action_probabilities(context, available_actions)
        row = [float(probabilities.get(action, 0.0)) for action in available_actions]
        if any(not np.isfinite(value) or value < 0.0 for value in row):
            raise ValueError(f"{policy.name} returned invalid action probabilities")
        if not np.isclose(sum(row), 1.0):
            raise ValueError(f"{policy.name} action probabilities must sum to 1")
        probability_rows.append(row)
    return np.asarray(probability_rows, dtype=float)


def evaluate_logged_data(
    frame: pd.DataFrame,
    config: OffPolicyEvaluationConfig,
    policies: Sequence[ProbabilisticBanditPolicy] | None = None,
) -> OffPolicyEvaluationRun:
    validate_logged_bandit_data(frame, config.n_actions)
    selected_policies = tuple(policies) if policies is not None else default_target_policies(config)
    if not selected_policies:
        raise ValueError("at least one target policy is required")
    policy_names = [policy.name for policy in selected_policies]
    if len(set(policy_names)) != len(policy_names):
        raise ValueError("target policy names must be unique")

    contexts = _contexts_from_frame(frame)
    logged_actions = frame["action"].to_numpy(dtype=int)
    rewards = frame["reward"].to_numpy(dtype=float)
    propensities = frame["propensity"].to_numpy(dtype=float)
    available_actions = tuple(range(config.n_actions))

    reward_model = LinearRewardModel(
        config.n_actions,
        regularization=config.reward_model_regularization,
    )
    reward_model.fit(contexts, logged_actions, rewards)
    model_predictions = np.asarray(
        [
            [reward_model.predict(context, action) for action in available_actions]
            for context in contexts
        ]
    )
    simulator_predictions = np.asarray(
        [
            [reward_probability(context, action) for action in available_actions]
            for context in contexts
        ]
    )
    direct_result = estimate_direct_logged_average(rewards)
    policy_seed_sequences = np.random.SeedSequence(config.seed).spawn(len(selected_policies))

    evaluations: dict[str, TargetPolicyEvaluation] = {}
    row_indices = np.arange(len(frame))
    for index, policy in enumerate(selected_policies):
        probability_matrix = _policy_probability_matrix(
            policy,
            contexts,
            available_actions,
        )
        logged_target_probabilities = probability_matrix[
            row_indices,
            logged_actions,
        ]
        policy_rng = np.random.default_rng(policy_seed_sequences[index])
        target_actions = np.asarray(
            [
                policy.choose_action(
                    context,
                    available_actions,
                    policy_rng,
                )
                for context in contexts
            ],
            dtype=int,
        )
        target_model_predictions = np.sum(
            probability_matrix * model_predictions,
            axis=1,
        )
        logged_model_predictions = model_predictions[row_indices, logged_actions]
        simulator_value = float(np.mean(np.sum(probability_matrix * simulator_predictions, axis=1)))

        evaluations[policy.name] = TargetPolicyEvaluation(
            policy_name=policy.name,
            simulator_value=simulator_value,
            estimators={
                "direct_logged_average": direct_result,
                "replay_matching": estimate_replay_matching(
                    rewards,
                    logged_actions,
                    target_actions,
                ),
                "ips": estimate_ips(
                    rewards,
                    propensities,
                    logged_target_probabilities,
                ),
                "snips": estimate_snips(
                    rewards,
                    propensities,
                    logged_target_probabilities,
                ),
                "doubly_robust": estimate_doubly_robust(
                    rewards,
                    propensities,
                    logged_target_probabilities,
                    target_model_predictions,
                    logged_model_predictions,
                ),
            },
        )

    expected_behavior_value = (
        float(frame["reward_probability"].mean()) if "reward_probability" in frame.columns else None
    )
    uniform_propensity = 1.0 / config.n_actions
    behavior_policy = (
        f"random_uniform (propensity={uniform_propensity:.4f})"
        if np.allclose(propensities, uniform_propensity)
        else "logged behavior policy (row-level propensities)"
    )
    return OffPolicyEvaluationRun(
        row_count=len(frame),
        behavior_policy=behavior_policy,
        observed_behavior_value=float(rewards.mean()),
        expected_behavior_value=expected_behavior_value,
        policies=evaluations,
    )


def run_off_policy_evaluation(
    config: OffPolicyEvaluationConfig,
) -> OffPolicyEvaluationRun:
    simulation_config = SimulationConfig(
        n_events=config.n_events,
        seed=config.seed,
        n_actions=config.n_actions,
    )
    frame = events_to_frame(simulate_bandit_events(simulation_config))
    return evaluate_logged_data(frame, config)
