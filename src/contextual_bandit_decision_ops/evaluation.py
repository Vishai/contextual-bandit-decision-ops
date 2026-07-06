from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import PolicyComparisonConfig
from .policies import (
    BanditPolicy,
    EpsilonGreedyPolicy,
    FixedActionPolicy,
    GreedyOraclePolicy,
    RandomUniformPolicy,
)
from .schemas import UserContext
from .simulation import reward_probability
from .synthetic import generate_user_contexts


@dataclass(frozen=True)
class PolicyEvaluation:
    policy_name: str
    total_reward: int
    average_reward: float
    action_distribution: dict[int, int]
    cumulative_rewards: tuple[int, ...]
    regret_estimate: float


def default_policies(config: PolicyComparisonConfig) -> tuple[BanditPolicy, ...]:
    return (
        RandomUniformPolicy(),
        FixedActionPolicy(config.fixed_action),
        GreedyOraclePolicy(),
        EpsilonGreedyPolicy(config.epsilon),
    )


def _evaluate_policy(
    policy: BanditPolicy,
    contexts: Sequence[UserContext],
    reward_draws: np.ndarray,
    available_actions: tuple[int, ...],
    rng: np.random.Generator,
) -> PolicyEvaluation:
    action_counts = {action: 0 for action in available_actions}
    cumulative_rewards: list[int] = []
    total_reward = 0
    total_probability_gap = 0.0

    for context, reward_draw in zip(contexts, reward_draws, strict=True):
        action = policy.choose_action(context, available_actions, rng)
        if action not in action_counts:
            raise ValueError(f"{policy.name} returned unavailable action {action}")

        chosen_probability = reward_probability(context, action)
        oracle_probability = max(
            reward_probability(context, candidate) for candidate in available_actions
        )
        total_probability_gap += oracle_probability - chosen_probability
        action_counts[action] += 1
        total_reward += int(reward_draw < chosen_probability)
        cumulative_rewards.append(total_reward)

    n_events = len(contexts)
    return PolicyEvaluation(
        policy_name=policy.name,
        total_reward=total_reward,
        average_reward=total_reward / n_events,
        action_distribution=action_counts,
        cumulative_rewards=tuple(cumulative_rewards),
        regret_estimate=total_probability_gap / n_events,
    )


def compare_policies(
    config: PolicyComparisonConfig,
    policies: Sequence[BanditPolicy] | None = None,
) -> dict[str, PolicyEvaluation]:
    selected_policies = tuple(policies) if policies is not None else default_policies(config)
    if not selected_policies:
        raise ValueError("at least one policy is required")
    policy_names = [policy.name for policy in selected_policies]
    if len(set(policy_names)) != len(policy_names):
        raise ValueError("policy names must be unique")

    seed_sequences = np.random.SeedSequence(config.seed).spawn(2 + len(selected_policies))
    context_rng = np.random.default_rng(seed_sequences[0])
    reward_rng = np.random.default_rng(seed_sequences[1])
    contexts = generate_user_contexts(config.n_events, context_rng)
    reward_draws = reward_rng.random(config.n_events)
    available_actions = tuple(range(config.n_actions))

    return {
        policy.name: _evaluate_policy(
            policy,
            contexts,
            reward_draws,
            available_actions,
            np.random.default_rng(seed_sequences[index + 2]),
        )
        for index, policy in enumerate(selected_policies)
    }
