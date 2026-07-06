from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import LearningComparisonConfig
from .learning_policies import (
    LearningPolicy,
    LinearThompsonSamplingPolicy,
    LinUCBPolicy,
    OnlineEpsilonGreedyPolicy,
)
from .policies import BanditPolicy, GreedyOraclePolicy, RandomUniformPolicy
from .schemas import UserContext
from .simulation import reward_probability
from .synthetic import generate_user_contexts


@dataclass(frozen=True)
class LearningEvaluation:
    policy_name: str
    total_reward: int
    average_reward: float
    cumulative_regret: float
    action_distribution: dict[int, int]
    cumulative_rewards: tuple[int, ...]
    cumulative_regrets: tuple[float, ...]


def default_learning_policies(
    config: LearningComparisonConfig,
) -> tuple[BanditPolicy, ...]:
    return (
        RandomUniformPolicy(),
        OnlineEpsilonGreedyPolicy(config.n_actions, config.epsilon),
        LinUCBPolicy(
            config.n_actions,
            alpha=config.linucb_alpha,
            regularization=config.regularization,
        ),
        LinearThompsonSamplingPolicy(
            config.n_actions,
            exploration_scale=config.thompson_scale,
            regularization=config.regularization,
        ),
        GreedyOraclePolicy(),
    )


def _update_policy(
    policy: BanditPolicy,
    context: UserContext,
    action: int,
    reward: int,
) -> None:
    if isinstance(policy, LearningPolicy):
        policy.update(context, action, reward)


def _evaluate_learning_policy(
    policy: BanditPolicy,
    contexts: Sequence[UserContext],
    reward_draws: np.ndarray,
    available_actions: tuple[int, ...],
    rng: np.random.Generator,
) -> LearningEvaluation:
    action_counts = {action: 0 for action in available_actions}
    cumulative_rewards: list[int] = []
    cumulative_regrets: list[float] = []
    total_reward = 0
    total_regret = 0.0

    for context, reward_draw in zip(contexts, reward_draws, strict=True):
        action = policy.choose_action(context, available_actions, rng)
        if action not in action_counts:
            raise ValueError(f"{policy.name} returned unavailable action {action}")

        chosen_probability = reward_probability(context, action)
        oracle_probability = max(
            reward_probability(context, candidate) for candidate in available_actions
        )
        reward = int(reward_draw < chosen_probability)
        _update_policy(policy, context, action, reward)

        action_counts[action] += 1
        total_reward += reward
        total_regret += oracle_probability - chosen_probability
        cumulative_rewards.append(total_reward)
        cumulative_regrets.append(total_regret)

    return LearningEvaluation(
        policy_name=policy.name,
        total_reward=total_reward,
        average_reward=total_reward / len(contexts),
        cumulative_regret=total_regret,
        action_distribution=action_counts,
        cumulative_rewards=tuple(cumulative_rewards),
        cumulative_regrets=tuple(cumulative_regrets),
    )


def compare_learning_policies(
    config: LearningComparisonConfig,
    policies: Sequence[BanditPolicy] | None = None,
) -> dict[str, LearningEvaluation]:
    selected_policies = (
        tuple(policies) if policies is not None else default_learning_policies(config)
    )
    if not selected_policies:
        raise ValueError("at least one policy is required")
    policy_names = [policy.name for policy in selected_policies]
    if len(set(policy_names)) != len(policy_names):
        raise ValueError("policy names must be unique")

    seed_sequences = np.random.SeedSequence(config.seed).spawn(2 + len(selected_policies))
    contexts = generate_user_contexts(
        config.n_events,
        np.random.default_rng(seed_sequences[0]),
    )
    reward_draws = np.random.default_rng(seed_sequences[1]).random(config.n_events)
    available_actions = tuple(range(config.n_actions))

    return {
        policy.name: _evaluate_learning_policy(
            policy,
            contexts,
            reward_draws,
            available_actions,
            np.random.default_rng(seed_sequences[index + 2]),
        )
        for index, policy in enumerate(selected_policies)
    }
