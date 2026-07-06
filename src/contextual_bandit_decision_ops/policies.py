from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .schemas import UserContext
from .simulation import reward_probability


class BanditPolicy(Protocol):
    @property
    def name(self) -> str: ...

    def choose_action(
        self,
        context: UserContext,
        available_actions: Sequence[int],
        rng: np.random.Generator,
    ) -> int: ...


def _actions_tuple(available_actions: Sequence[int]) -> tuple[int, ...]:
    actions = tuple(available_actions)
    if not actions:
        raise ValueError("available_actions must not be empty")
    return actions


@dataclass(frozen=True)
class RandomUniformPolicy:
    @property
    def name(self) -> str:
        return "random_uniform"

    def choose_action(
        self,
        context: UserContext,
        available_actions: Sequence[int],
        rng: np.random.Generator,
    ) -> int:
        del context
        actions = _actions_tuple(available_actions)
        return actions[int(rng.integers(0, len(actions)))]


@dataclass(frozen=True)
class FixedActionPolicy:
    action: int

    @property
    def name(self) -> str:
        return f"fixed_action_{self.action}"

    def choose_action(
        self,
        context: UserContext,
        available_actions: Sequence[int],
        rng: np.random.Generator,
    ) -> int:
        del context, rng
        actions = _actions_tuple(available_actions)
        if self.action not in actions:
            raise ValueError(f"fixed action {self.action} is unavailable")
        return self.action


@dataclass(frozen=True)
class GreedyOraclePolicy:
    @property
    def name(self) -> str:
        return "greedy_oracle"

    def choose_action(
        self,
        context: UserContext,
        available_actions: Sequence[int],
        rng: np.random.Generator,
    ) -> int:
        del rng
        actions = _actions_tuple(available_actions)
        return max(actions, key=lambda action: reward_probability(context, action))


@dataclass(frozen=True)
class EpsilonGreedyPolicy:
    epsilon: float = 0.1

    def __post_init__(self) -> None:
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")

    @property
    def name(self) -> str:
        return f"epsilon_greedy_{self.epsilon:.2f}"

    def choose_action(
        self,
        context: UserContext,
        available_actions: Sequence[int],
        rng: np.random.Generator,
    ) -> int:
        actions = _actions_tuple(available_actions)
        if rng.random() < self.epsilon:
            return actions[int(rng.integers(0, len(actions)))]
        return GreedyOraclePolicy().choose_action(context, actions, rng)
