from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Callable

import numpy as np

from .api_schemas import (
    DecisionRequest,
    DecisionResponse,
    FeedbackRequest,
    FeedbackResponse,
    MetricsResponse,
    PolicyMetadataResponse,
)
from .policies import ProbabilisticBanditPolicy, RandomUniformPolicy
from .safety import PolicyConstraints
from .schemas import UserContext
from .service_logging import LocalEventLogger

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class ServiceConfig:
    available_actions: tuple[int, ...] = (0, 1, 2)
    seed: int = 42
    decision_log_path: Path | str = Path("logs/decisions.jsonl")
    feedback_log_path: Path | str = Path("logs/feedback.jsonl")
    safety_constraints: PolicyConstraints | None = None
    staging_only: bool = True

    def __post_init__(self) -> None:
        if not self.available_actions:
            raise ValueError("available_actions must not be empty")
        if len(set(self.available_actions)) != len(self.available_actions):
            raise ValueError("available_actions must be unique")
        if any(action < 0 for action in self.available_actions):
            raise ValueError("available_actions must be non-negative")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.staging_only:
            raise ValueError("the decision service is local/staging only")
        if self.safety_constraints and not self.effective_actions:
            raise ValueError("safety constraints block every available action")
        object.__setattr__(self, "decision_log_path", Path(self.decision_log_path))
        object.__setattr__(self, "feedback_log_path", Path(self.feedback_log_path))

    @property
    def effective_actions(self) -> tuple[int, ...]:
        blocked_actions = (
            self.safety_constraints.blocked_actions
            if self.safety_constraints is not None
            else frozenset()
        )
        return tuple(action for action in self.available_actions if action not in blocked_actions)


class UnknownDecisionEventError(ValueError):
    pass


class DuplicateFeedbackError(ValueError):
    pass


class DecisionService:
    def __init__(
        self,
        config: ServiceConfig | None = None,
        policy: ProbabilisticBanditPolicy | None = None,
        clock: Clock = utc_now,
    ) -> None:
        self.config = config or ServiceConfig()
        self.policy = policy or RandomUniformPolicy()
        self.clock = clock
        self.logger = LocalEventLogger(
            self.config.decision_log_path,
            self.config.feedback_log_path,
        )
        self.rng = np.random.default_rng(self.config.seed)
        self.lock = Lock()
        self.decision_count = 0
        self.feedback_count = 0
        self.action_counts = {action: 0 for action in self.config.effective_actions}
        self.reward_sum = 0.0
        self.decision_event_ids: set[str] = set()
        self.feedback_event_ids: set[str] = set()

    def _timestamp(self) -> datetime:
        timestamp = self.clock()
        if timestamp.tzinfo is None:
            raise ValueError("service clock must return a timezone-aware datetime")
        return timestamp.astimezone(UTC)

    def policy_metadata(self) -> PolicyMetadataResponse:
        constraints = None
        if self.config.safety_constraints is not None:
            constraints = asdict(self.config.safety_constraints)
            constraints["blocked_actions"] = sorted(self.config.safety_constraints.blocked_actions)
        return PolicyMetadataResponse(
            policy_name=self.policy.name,
            available_actions=list(self.config.effective_actions),
            safety_constraints=constraints,
            staging_only=self.config.staging_only,
        )

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        context = UserContext(
            age=request.context.age,
            engagement=request.context.engagement,
            region=request.context.region,
        )
        actions = self.config.effective_actions
        with self.lock:
            probabilities = self.policy.action_probabilities(context, actions)
            probability_values = np.asarray(
                [probabilities.get(action, 0.0) for action in actions],
                dtype=float,
            )
            if (
                np.any(~np.isfinite(probability_values))
                or np.any(probability_values < 0.0)
                or not np.isclose(probability_values.sum(), 1.0)
            ):
                raise ValueError("policy returned an invalid probability distribution")
            action = self.policy.choose_action(context, actions, self.rng)
            if action not in actions:
                raise ValueError(f"policy returned unavailable action {action}")
            propensity = float(probabilities.get(action, 0.0))
            if not 0.0 < propensity <= 1.0:
                raise ValueError("policy returned an invalid propensity")

            event_id = f"service-event-{self.config.seed}-{self.decision_count:08d}"
            timestamp = self._timestamp()
            response = DecisionResponse(
                event_id=event_id,
                action=action,
                propensity=propensity,
                policy_name=self.policy.name,
                timestamp=timestamp,
            )
            self.logger.log_decision(
                {
                    "event_id": event_id,
                    "user_id": request.user_id,
                    "context": request.context.model_dump(mode="json"),
                    "action": action,
                    "propensity": propensity,
                    "policy_name": self.policy.name,
                    "timestamp": timestamp.isoformat(),
                    "metadata": request.metadata or {},
                    "seed": self.config.seed,
                }
            )
            self.decision_count += 1
            self.action_counts[action] += 1
            self.decision_event_ids.add(event_id)
            return response

    def feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        with self.lock:
            if request.event_id not in self.decision_event_ids:
                raise UnknownDecisionEventError(request.event_id)
            if request.event_id in self.feedback_event_ids:
                raise DuplicateFeedbackError(request.event_id)
            timestamp = self._timestamp()
            response = FeedbackResponse(
                event_id=request.event_id,
                status="accepted",
                timestamp=timestamp,
            )
            self.logger.log_feedback(
                {
                    "event_id": request.event_id,
                    "reward": request.reward,
                    "timestamp": timestamp.isoformat(),
                    "metadata": request.metadata or {},
                }
            )
            self.feedback_count += 1
            self.reward_sum += request.reward
            self.feedback_event_ids.add(request.event_id)
            return response

    def metrics(self) -> MetricsResponse:
        with self.lock:
            average_reward = self.reward_sum / self.feedback_count if self.feedback_count else None
            return MetricsResponse(
                decisions=self.decision_count,
                feedback_records=self.feedback_count,
                action_counts=dict(self.action_counts),
                average_observed_reward=average_reward,
            )
