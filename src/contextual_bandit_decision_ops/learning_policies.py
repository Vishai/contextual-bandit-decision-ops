from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from .policies import BanditPolicy
from .schemas import REGIONS, UserContext

CONTEXT_FEATURE_DIMENSION = 8


def context_vector(context: UserContext) -> np.ndarray:
    """Encode bias, scaled numeric features, threshold, and region indicators."""
    region_features = [float(context.region == region) for region in REGIONS]
    return np.array(
        [
            1.0,
            (context.age - 18.0) / 52.0,
            context.engagement,
            float(context.engagement >= 0.5),
            *region_features,
        ],
        dtype=float,
    )


@runtime_checkable
class LearningPolicy(BanditPolicy, Protocol):
    def update(self, context: UserContext, action: int, reward: float) -> None: ...


def _validate_actions(
    available_actions: Sequence[int],
    n_actions: int,
) -> tuple[int, ...]:
    actions = tuple(available_actions)
    if not actions:
        raise ValueError("available_actions must not be empty")
    if any(action not in range(n_actions) for action in actions):
        raise ValueError("available_actions contains an unknown action")
    return actions


def _validate_update(action: int, reward: float, n_actions: int) -> None:
    if action not in range(n_actions):
        raise ValueError("action is out of range")
    if not 0.0 <= reward <= 1.0:
        raise ValueError("reward must be between 0 and 1")


@dataclass
class OnlineEpsilonGreedyPolicy:
    """Non-contextual online baseline using empirical action reward means."""

    n_actions: int
    epsilon: float = 0.1
    action_counts: np.ndarray = field(init=False, repr=False)
    reward_sums: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.reward_sums = np.zeros(self.n_actions, dtype=float)

    @property
    def name(self) -> str:
        return f"online_epsilon_greedy_{self.epsilon:.2f}"

    def choose_action(
        self,
        context: UserContext,
        available_actions: Sequence[int],
        rng: np.random.Generator,
    ) -> int:
        del context
        actions = _validate_actions(available_actions, self.n_actions)
        untried_actions = [action for action in actions if self.action_counts[action] == 0]
        if untried_actions:
            return untried_actions[0]
        if rng.random() < self.epsilon:
            return actions[int(rng.integers(0, len(actions)))]
        return max(
            actions,
            key=lambda action: self.reward_sums[action] / self.action_counts[action],
        )

    def update(self, context: UserContext, action: int, reward: float) -> None:
        del context
        _validate_update(action, reward, self.n_actions)
        self.action_counts[action] += 1
        self.reward_sums[action] += reward


@dataclass
class LinUCBPolicy:
    """Per-action ridge regression with an upper-confidence exploration bonus."""

    n_actions: int
    alpha: float = 0.5
    regularization: float = 1.0
    precision_matrices: np.ndarray = field(init=False, repr=False)
    reward_vectors: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive")
        if self.alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        if self.regularization <= 0.0:
            raise ValueError("regularization must be positive")
        identity = np.eye(CONTEXT_FEATURE_DIMENSION)
        self.precision_matrices = np.repeat(
            (self.regularization * identity)[None, :, :],
            self.n_actions,
            axis=0,
        )
        self.reward_vectors = np.zeros(
            (self.n_actions, CONTEXT_FEATURE_DIMENSION),
            dtype=float,
        )

    @property
    def name(self) -> str:
        return "linucb"

    def choose_action(
        self,
        context: UserContext,
        available_actions: Sequence[int],
        rng: np.random.Generator,
    ) -> int:
        del rng
        actions = _validate_actions(available_actions, self.n_actions)
        features = context_vector(context)
        scores: dict[int, float] = {}
        for action in actions:
            precision = self.precision_matrices[action]
            coefficient_mean = np.linalg.solve(precision, self.reward_vectors[action])
            feature_variance = features @ np.linalg.solve(precision, features)
            confidence_bonus = self.alpha * np.sqrt(max(feature_variance, 0.0))
            scores[action] = float(coefficient_mean @ features + confidence_bonus)
        return max(actions, key=scores.__getitem__)

    def update(self, context: UserContext, action: int, reward: float) -> None:
        _validate_update(action, reward, self.n_actions)
        features = context_vector(context)
        self.precision_matrices[action] += np.outer(features, features)
        self.reward_vectors[action] += reward * features


@dataclass
class LinearThompsonSamplingPolicy:
    """Per-action linear posterior sampling with a Gaussian reward approximation."""

    n_actions: int
    exploration_scale: float = 0.25
    regularization: float = 1.0
    precision_matrices: np.ndarray = field(init=False, repr=False)
    reward_vectors: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive")
        if self.exploration_scale <= 0.0:
            raise ValueError("exploration_scale must be positive")
        if self.regularization <= 0.0:
            raise ValueError("regularization must be positive")
        identity = np.eye(CONTEXT_FEATURE_DIMENSION)
        self.precision_matrices = np.repeat(
            (self.regularization * identity)[None, :, :],
            self.n_actions,
            axis=0,
        )
        self.reward_vectors = np.zeros(
            (self.n_actions, CONTEXT_FEATURE_DIMENSION),
            dtype=float,
        )

    @property
    def name(self) -> str:
        return "linear_thompson_sampling"

    def choose_action(
        self,
        context: UserContext,
        available_actions: Sequence[int],
        rng: np.random.Generator,
    ) -> int:
        actions = _validate_actions(available_actions, self.n_actions)
        features = context_vector(context)
        scores: dict[int, float] = {}
        for action in actions:
            precision = self.precision_matrices[action]
            coefficient_mean = np.linalg.solve(precision, self.reward_vectors[action])
            precision_cholesky = np.linalg.cholesky(precision)
            posterior_noise = np.linalg.solve(
                precision_cholesky.T,
                rng.normal(size=CONTEXT_FEATURE_DIMENSION),
            )
            sampled_coefficients = coefficient_mean + self.exploration_scale * posterior_noise
            scores[action] = float(sampled_coefficients @ features)
        return max(actions, key=scores.__getitem__)

    def update(self, context: UserContext, action: int, reward: float) -> None:
        _validate_update(action, reward, self.n_actions)
        features = context_vector(context)
        self.precision_matrices[action] += np.outer(features, features)
        self.reward_vectors[action] += reward * features
