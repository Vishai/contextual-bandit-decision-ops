from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import OffPolicyEvaluationConfig, PolicyComparisonConfig
from .evaluation import PolicyEvaluation, compare_policies
from .off_policy import TargetPolicyEvaluation, run_off_policy_evaluation
from .safety import PolicyConstraints


@dataclass(frozen=True)
class PromotionGateConfig:
    simulation_events: int = 5_000
    ope_events: int = 5_000
    seed: int = 42
    n_actions: int = 3
    fixed_action: int = 0
    epsilon: float = 0.1
    constraints: PolicyConstraints = field(default_factory=PolicyConstraints)
    report_md: Path | str = Path("reports/policy_promotion_gate.md")
    artifact_json: Path | str = Path("artifacts/policy_promotion_gate.json")

    def __post_init__(self) -> None:
        if self.simulation_events <= 0:
            raise ValueError("simulation_events must be positive")
        if self.ope_events <= 0:
            raise ValueError("ope_events must be positive")
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.fixed_action not in range(self.n_actions):
            raise ValueError("fixed_action must be an available action")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")
        if any(action not in range(self.n_actions) for action in self.constraints.blocked_actions):
            raise ValueError("blocked action is outside the configured action space")
        if len(self.constraints.blocked_actions) == self.n_actions:
            raise ValueError("at least one action must remain unblocked")
        object.__setattr__(self, "report_md", Path(self.report_md))
        object.__setattr__(self, "artifact_json", Path(self.artifact_json))


@dataclass(frozen=True)
class CandidateMetrics:
    policy_name: str
    estimated_value: float | None
    baseline_estimated_value: float | None
    estimated_improvement: float | None
    effective_sample_size: float | None
    matched_replay_count: int | None
    average_regret: float
    action_distribution: dict[int, int]
    max_action_share: float
    exploration_rate: float
    blocked_action_selections: int
    minimum_allowed_action_count: int


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    observed: str
    requirement: str
    message: str


@dataclass(frozen=True)
class CandidateGateResult:
    policy_name: str
    decision: str
    metrics: CandidateMetrics
    passed_checks: tuple[GateCheck, ...]
    failed_checks: tuple[GateCheck, ...]
    warnings: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class PromotionGateRun:
    decision: str
    selected_candidate: str | None
    evidence_source: str
    candidate_results: dict[str, CandidateGateResult]
    warnings: tuple[str, ...]
    reason: str


def build_candidate_metrics(
    simulation: PolicyEvaluation,
    ope: TargetPolicyEvaluation,
    baseline_ope: TargetPolicyEvaluation,
    constraints: PolicyConstraints,
    n_actions: int,
) -> CandidateMetrics:
    doubly_robust = ope.estimators["doubly_robust"]
    baseline_doubly_robust = baseline_ope.estimators["doubly_robust"]
    replay = ope.estimators["replay_matching"]
    total_actions = sum(simulation.action_distribution.values())
    action_shares = {
        action: simulation.action_distribution.get(action, 0) / total_actions
        for action in range(n_actions)
    }
    max_action_share = max(action_shares.values())
    allowed_actions = [
        action for action in range(n_actions) if action not in constraints.blocked_actions
    ]
    estimated_improvement = (
        doubly_robust.value - baseline_doubly_robust.value
        if doubly_robust.value is not None and baseline_doubly_robust.value is not None
        else None
    )
    return CandidateMetrics(
        policy_name=simulation.policy_name,
        estimated_value=doubly_robust.value,
        baseline_estimated_value=baseline_doubly_robust.value,
        estimated_improvement=estimated_improvement,
        effective_sample_size=doubly_robust.effective_sample_size,
        matched_replay_count=replay.matched_count,
        average_regret=simulation.regret_estimate,
        action_distribution=dict(simulation.action_distribution),
        max_action_share=max_action_share,
        exploration_rate=1.0 - max_action_share,
        blocked_action_selections=sum(
            simulation.action_distribution.get(action, 0) for action in constraints.blocked_actions
        ),
        minimum_allowed_action_count=min(
            simulation.action_distribution.get(action, 0) for action in allowed_actions
        ),
    )


def _gate_check(
    name: str,
    passed: bool,
    observed: str,
    requirement: str,
    message: str,
) -> GateCheck:
    return GateCheck(
        name=name,
        passed=passed,
        observed=observed,
        requirement=requirement,
        message=message,
    )


def evaluate_candidate_promotion(
    metrics: CandidateMetrics,
    constraints: PolicyConstraints,
    *,
    evidence_is_synthetic: bool,
) -> CandidateGateResult:
    checks = [
        _gate_check(
            "blocked_actions",
            metrics.blocked_action_selections == 0,
            str(metrics.blocked_action_selections),
            "0 selections",
            "Candidate must never select a blocked action.",
        ),
        _gate_check(
            "max_action_share",
            metrics.max_action_share <= constraints.max_action_share,
            f"{metrics.max_action_share:.2%}",
            f"<= {constraints.max_action_share:.2%}",
            "No action may exceed its capacity share.",
        ),
        _gate_check(
            "minimum_exploration_rate",
            metrics.exploration_rate >= constraints.min_exploration_rate,
            f"{metrics.exploration_rate:.2%}",
            f">= {constraints.min_exploration_rate:.2%}",
            "Exploration is one minus the largest action share.",
        ),
        _gate_check(
            "action_coverage",
            metrics.minimum_allowed_action_count >= constraints.min_action_count,
            str(metrics.minimum_allowed_action_count),
            f">= {constraints.min_action_count} per allowed action",
            "Every allowed action needs minimum observed coverage.",
        ),
        _gate_check(
            "ope_effective_sample_size",
            metrics.effective_sample_size is not None
            and metrics.effective_sample_size >= constraints.min_effective_sample_size,
            (
                "N/A"
                if metrics.effective_sample_size is None
                else f"{metrics.effective_sample_size:.1f}"
            ),
            f">= {constraints.min_effective_sample_size:.1f}",
            "Doubly robust OPE needs adequate effective sample size.",
        ),
        _gate_check(
            "replay_matched_count",
            metrics.matched_replay_count is not None
            and metrics.matched_replay_count >= constraints.min_matched_replay_count,
            "N/A" if metrics.matched_replay_count is None else str(metrics.matched_replay_count),
            f">= {constraints.min_matched_replay_count}",
            "Replay evaluation needs enough matching logged actions.",
        ),
        _gate_check(
            "estimated_improvement",
            metrics.estimated_improvement is not None
            and metrics.estimated_improvement >= constraints.min_estimated_improvement,
            (
                "N/A"
                if metrics.estimated_improvement is None
                else f"{metrics.estimated_improvement:.4f}"
            ),
            f">= {constraints.min_estimated_improvement:.4f}",
            "Doubly robust value must improve over the random baseline.",
        ),
        _gate_check(
            "simulation_regret",
            metrics.average_regret <= constraints.max_average_regret,
            f"{metrics.average_regret:.4f}",
            f"<= {constraints.max_average_regret:.4f}",
            "Average expected regret to the simulator oracle must remain bounded.",
        ),
    ]
    if constraints.require_non_synthetic_evidence:
        checks.append(
            _gate_check(
                "non_synthetic_evidence",
                not evidence_is_synthetic,
                "synthetic" if evidence_is_synthetic else "external",
                "external validation required",
                "Synthetic results alone cannot authorize launch.",
            )
        )

    warnings: list[str] = []
    if evidence_is_synthetic:
        warnings.append("All evidence is synthetic and does not establish real-world safety.")
    if (
        metrics.effective_sample_size is None
        or metrics.effective_sample_size < constraints.min_effective_sample_size
    ):
        warnings.append("OPE effective sample size is below the reliability threshold.")
    if (
        metrics.matched_replay_count is None
        or metrics.matched_replay_count < constraints.min_matched_replay_count
    ):
        warnings.append("Replay matched support is below the reliability threshold.")
    if metrics.max_action_share > constraints.max_action_share:
        warnings.append("Observed action concentration exceeds the configured capacity cap.")
    if metrics.policy_name.startswith("epsilon_greedy_"):
        warnings.append(
            "This epsilon-greedy candidate uses simulator-oracle scores and is not deployable."
        )

    passed_checks = tuple(check for check in checks if check.passed)
    failed_checks = tuple(check for check in checks if not check.passed)
    decision = "promote" if not failed_checks else "hold"
    reason = (
        "All configured promotion checks passed."
        if decision == "promote"
        else "Failed checks: " + ", ".join(check.name for check in failed_checks) + "."
    )
    return CandidateGateResult(
        policy_name=metrics.policy_name,
        decision=decision,
        metrics=metrics,
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        warnings=tuple(warnings),
        reason=reason,
    )


def evaluate_promotion_gate(
    candidate_metrics: list[CandidateMetrics],
    constraints: PolicyConstraints,
    *,
    evidence_is_synthetic: bool,
) -> PromotionGateRun:
    if not candidate_metrics:
        raise ValueError("at least one candidate policy is required")
    candidate_results = {
        metrics.policy_name: evaluate_candidate_promotion(
            metrics,
            constraints,
            evidence_is_synthetic=evidence_is_synthetic,
        )
        for metrics in candidate_metrics
    }
    promotable_results = [
        result for result in candidate_results.values() if result.decision == "promote"
    ]
    selected_candidate = (
        max(
            promotable_results,
            key=lambda result: (
                result.metrics.estimated_improvement
                if result.metrics.estimated_improvement is not None
                else float("-inf")
            ),
        ).policy_name
        if promotable_results
        else None
    )
    decision = "promote" if selected_candidate is not None else "hold"
    warnings = tuple(
        dict.fromkeys(
            warning for result in candidate_results.values() for warning in result.warnings
        )
    )
    reason = (
        f"Promote `{selected_candidate}`; it passed every configured check."
        if selected_candidate is not None
        else "Hold all candidates; none passed every configured check."
    )
    return PromotionGateRun(
        decision=decision,
        selected_candidate=selected_candidate,
        evidence_source="deterministic synthetic simulation and OPE",
        candidate_results=candidate_results,
        warnings=warnings,
        reason=reason,
    )


def run_promotion_gate(config: PromotionGateConfig) -> PromotionGateRun:
    simulation_results = compare_policies(
        PolicyComparisonConfig(
            n_events=config.simulation_events,
            seed=config.seed,
            n_actions=config.n_actions,
            fixed_action=config.fixed_action,
            epsilon=config.epsilon,
        )
    )
    ope_run = run_off_policy_evaluation(
        OffPolicyEvaluationConfig(
            n_events=config.ope_events,
            seed=config.seed,
            n_actions=config.n_actions,
            fixed_action=config.fixed_action,
            epsilon=config.epsilon,
        )
    )
    candidate_names = (
        f"fixed_action_{config.fixed_action}",
        f"epsilon_greedy_{config.epsilon:.2f}",
    )
    baseline_ope = ope_run.policies["random_uniform"]
    candidate_metrics = [
        build_candidate_metrics(
            simulation_results[candidate_name],
            ope_run.policies[candidate_name],
            baseline_ope,
            config.constraints,
            config.n_actions,
        )
        for candidate_name in candidate_names
    ]
    return evaluate_promotion_gate(
        candidate_metrics,
        config.constraints,
        evidence_is_synthetic=True,
    )
