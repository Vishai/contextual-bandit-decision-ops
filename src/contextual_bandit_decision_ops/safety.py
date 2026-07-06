from __future__ import annotations

from dataclasses import dataclass, field
from math import floor
from typing import Sequence

import numpy as np

from .learning_policies import LearningPolicy
from .policies import BanditPolicy
from .schemas import UserContext


@dataclass(frozen=True)
class PolicyConstraints:
    blocked_actions: frozenset[int] = frozenset()
    max_action_share: float = 0.70
    min_exploration_rate: float = 0.10
    min_action_count: int = 10
    min_effective_sample_size: float = 1_000.0
    min_matched_replay_count: int = 1_000
    min_estimated_improvement: float = 0.01
    max_average_regret: float = 0.03
    require_non_synthetic_evidence: bool = True

    def __post_init__(self) -> None:
        if any(action < 0 for action in self.blocked_actions):
            raise ValueError("blocked actions must be non-negative")
        if not 0.0 < self.max_action_share <= 1.0:
            raise ValueError("max_action_share must be greater than 0 and at most 1")
        if not 0.0 <= self.min_exploration_rate <= 1.0:
            raise ValueError("min_exploration_rate must be between 0 and 1")
        if self.min_action_count < 0:
            raise ValueError("min_action_count must be non-negative")
        if self.min_effective_sample_size < 0.0:
            raise ValueError("min_effective_sample_size must be non-negative")
        if self.min_matched_replay_count < 0:
            raise ValueError("min_matched_replay_count must be non-negative")
        if self.max_average_regret < 0.0:
            raise ValueError("max_average_regret must be non-negative")


@dataclass
class ConstrainedPolicy:
    """Enforce blocked actions and horizon-level action capacity on a policy."""

    policy: BanditPolicy
    horizon: int
    blocked_actions: frozenset[int] = frozenset()
    max_action_share: float = 1.0
    action_counts: dict[int, int] = field(default_factory=dict, init=False)
    constraint_warnings: list[str] = field(default_factory=list, init=False)
    decisions: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if not 0.0 < self.max_action_share <= 1.0:
            raise ValueError("max_action_share must be greater than 0 and at most 1")
        if any(action < 0 for action in self.blocked_actions):
            raise ValueError("blocked actions must be non-negative")

    @property
    def name(self) -> str:
        return f"constrained_{self.policy.name}"

    @property
    def action_shares(self) -> dict[int, float]:
        if self.decisions == 0:
            return {action: 0.0 for action in self.action_counts}
        return {action: count / self.decisions for action, count in self.action_counts.items()}

    def _warn_once(self, warning: str) -> None:
        if warning not in self.constraint_warnings:
            self.constraint_warnings.append(warning)

    def choose_action(
        self,
        context: UserContext,
        available_actions: Sequence[int],
        rng: np.random.Generator,
    ) -> int:
        if self.decisions >= self.horizon:
            raise RuntimeError("constrained policy horizon has been exhausted")
        actions = tuple(available_actions)
        allowed_actions = tuple(action for action in actions if action not in self.blocked_actions)
        if not allowed_actions:
            raise ValueError("all available actions are blocked")

        action_capacity = floor(self.max_action_share * self.horizon)
        if action_capacity == 0 or action_capacity * len(allowed_actions) < self.horizon:
            raise ValueError("action-share cap is infeasible for the allowed actions and horizon")
        eligible_actions = tuple(
            action
            for action in allowed_actions
            if self.action_counts.get(action, 0) < action_capacity
        )
        if not eligible_actions:
            raise RuntimeError("all allowed actions have exhausted their capacity")

        proposed_action = self.policy.choose_action(context, actions, rng)
        if proposed_action not in actions:
            raise ValueError(f"wrapped policy returned unavailable action {proposed_action}")
        if proposed_action not in eligible_actions:
            proposed_action = min(
                eligible_actions,
                key=lambda action: (self.action_counts.get(action, 0), action),
            )
            self._warn_once(
                "wrapped policy choice was overridden by a blocked-action or share-cap constraint"
            )

        self.action_counts[proposed_action] = self.action_counts.get(proposed_action, 0) + 1
        self.decisions += 1
        return proposed_action

    def update(self, context: UserContext, action: int, reward: float) -> None:
        if isinstance(self.policy, LearningPolicy):
            self.policy.update(context, action, reward)
