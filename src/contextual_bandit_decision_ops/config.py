from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class SimulationConfig:
    n_events: int = 100
    seed: int = 42
    n_actions: int = 3
    output_csv: Path | str = Path("data/processed/synthetic_bandit_log.csv")
    report_md: Path | str = Path("reports/synthetic_bandit_log_summary.md")
    base_timestamp: datetime = datetime(2024, 1, 1, tzinfo=UTC)

    def __post_init__(self) -> None:
        if self.n_events <= 0:
            raise ValueError("n_events must be positive")
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.base_timestamp.tzinfo is None:
            raise ValueError("base_timestamp must include a timezone")
        object.__setattr__(self, "output_csv", Path(self.output_csv))
        object.__setattr__(self, "report_md", Path(self.report_md))


@dataclass(frozen=True)
class PolicyComparisonConfig:
    n_events: int = 1_000
    seed: int = 42
    n_actions: int = 3
    fixed_action: int = 0
    epsilon: float = 0.1
    report_md: Path | str = Path("reports/baseline_policy_comparison.md")
    artifact_json: Path | str = Path("artifacts/baseline_policy_comparison.json")

    def __post_init__(self) -> None:
        if self.n_events <= 0:
            raise ValueError("n_events must be positive")
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.fixed_action not in range(self.n_actions):
            raise ValueError("fixed_action must be an available action")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")
        object.__setattr__(self, "report_md", Path(self.report_md))
        object.__setattr__(self, "artifact_json", Path(self.artifact_json))


@dataclass(frozen=True)
class LearningComparisonConfig:
    n_events: int = 5_000
    seed: int = 42
    n_actions: int = 3
    epsilon: float = 0.1
    linucb_alpha: float = 0.5
    thompson_scale: float = 0.25
    regularization: float = 1.0
    report_md: Path | str = Path("reports/contextual_learning_policy_comparison.md")
    artifact_json: Path | str = Path("artifacts/contextual_learning_policy_comparison.json")

    def __post_init__(self) -> None:
        if self.n_events <= 0:
            raise ValueError("n_events must be positive")
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")
        if self.linucb_alpha < 0.0:
            raise ValueError("linucb_alpha must be non-negative")
        if self.thompson_scale <= 0.0:
            raise ValueError("thompson_scale must be positive")
        if self.regularization <= 0.0:
            raise ValueError("regularization must be positive")
        object.__setattr__(self, "report_md", Path(self.report_md))
        object.__setattr__(self, "artifact_json", Path(self.artifact_json))


@dataclass(frozen=True)
class OffPolicyEvaluationConfig:
    n_events: int = 5_000
    seed: int = 42
    n_actions: int = 3
    fixed_action: int = 0
    epsilon: float = 0.1
    reward_model_regularization: float = 1.0
    report_md: Path | str = Path("reports/off_policy_evaluation.md")
    artifact_json: Path | str = Path("artifacts/off_policy_evaluation.json")

    def __post_init__(self) -> None:
        if self.n_events <= 0:
            raise ValueError("n_events must be positive")
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.fixed_action not in range(self.n_actions):
            raise ValueError("fixed_action must be an available action")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")
        if self.reward_model_regularization <= 0.0:
            raise ValueError("reward_model_regularization must be positive")
        object.__setattr__(self, "report_md", Path(self.report_md))
        object.__setattr__(self, "artifact_json", Path(self.artifact_json))
