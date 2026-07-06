import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from contextual_bandit_decision_ops.policies import FixedActionPolicy, RandomUniformPolicy
from contextual_bandit_decision_ops.promotion_cli import main
from contextual_bandit_decision_ops.promotion_gate import (
    CandidateMetrics,
    PromotionGateConfig,
    evaluate_candidate_promotion,
    run_promotion_gate,
)
from contextual_bandit_decision_ops.promotion_report import write_promotion_gate_outputs
from contextual_bandit_decision_ops.safety import ConstrainedPolicy, PolicyConstraints
from contextual_bandit_decision_ops.schemas import UserContext


def _passing_metrics() -> CandidateMetrics:
    return CandidateMetrics(
        policy_name="candidate",
        estimated_value=0.35,
        baseline_estimated_value=0.30,
        estimated_improvement=0.05,
        effective_sample_size=1_500.0,
        matched_replay_count=1_200,
        average_regret=0.01,
        action_distribution={0: 40, 1: 30, 2: 30},
        max_action_share=0.40,
        exploration_rate=0.60,
        blocked_action_selections=0,
        minimum_allowed_action_count=30,
    )


def test_blocked_actions_are_never_selected() -> None:
    policy = ConstrainedPolicy(
        RandomUniformPolicy(),
        horizon=120,
        blocked_actions=frozenset({1}),
        max_action_share=0.60,
    )
    context = UserContext(age=36.0, engagement=0.4, region="west")
    rng = np.random.default_rng(5)

    actions = [policy.choose_action(context, (0, 1, 2), rng) for _ in range(120)]

    assert 1 not in actions
    assert policy.action_counts.get(1, 0) == 0


def test_action_share_cap_is_enforced_and_warned_on() -> None:
    policy = ConstrainedPolicy(
        FixedActionPolicy(0),
        horizon=100,
        max_action_share=0.40,
    )
    context = UserContext(age=45.0, engagement=0.6, region="north")
    rng = np.random.default_rng(2)

    for _ in range(100):
        policy.choose_action(context, (0, 1, 2), rng)

    assert max(policy.action_shares.values()) <= 0.40
    assert policy.constraint_warnings


def test_promotion_gate_passes_and_fails_expected_checks() -> None:
    constraints = PolicyConstraints(require_non_synthetic_evidence=False)

    passing_result = evaluate_candidate_promotion(
        _passing_metrics(),
        constraints,
        evidence_is_synthetic=True,
    )
    failing_result = evaluate_candidate_promotion(
        replace(
            _passing_metrics(),
            max_action_share=0.90,
            exploration_rate=0.10,
            estimated_improvement=-0.01,
        ),
        constraints,
        evidence_is_synthetic=True,
    )

    assert passing_result.decision == "promote"
    assert not passing_result.failed_checks
    assert failing_result.decision == "hold"
    assert {"max_action_share", "estimated_improvement"} <= {
        check.name for check in failing_result.failed_checks
    }


def test_insufficient_ope_support_causes_hold_and_warning() -> None:
    constraints = PolicyConstraints(
        min_effective_sample_size=1_000.0,
        min_matched_replay_count=800,
        require_non_synthetic_evidence=False,
    )
    result = evaluate_candidate_promotion(
        replace(
            _passing_metrics(),
            effective_sample_size=25.0,
            matched_replay_count=10,
        ),
        constraints,
        evidence_is_synthetic=False,
    )

    assert result.decision == "hold"
    assert {"ope_effective_sample_size", "replay_matched_count"} <= {
        check.name for check in result.failed_checks
    }
    assert any("support" in warning or "sample size" in warning for warning in result.warnings)


def test_promotion_gate_is_deterministic_and_conservative() -> None:
    config = PromotionGateConfig(simulation_events=300, ope_events=400, seed=11)

    first_run = run_promotion_gate(config)
    second_run = run_promotion_gate(config)

    assert first_run == second_run
    assert first_run.decision == "hold"
    assert first_run.selected_candidate is None


def test_promotion_gate_report_and_json_creation(tmp_path: Path) -> None:
    config = PromotionGateConfig(
        simulation_events=250,
        ope_events=300,
        seed=7,
        report_md=tmp_path / "gate.md",
        artifact_json=tmp_path / "gate.json",
    )
    run = run_promotion_gate(config)

    report_path, artifact_path = write_promotion_gate_outputs(config, run)

    report_text = report_path.read_text(encoding="utf-8")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert "# Policy Promotion Gate" in report_text
    assert "Final decision: HOLD" in report_text
    assert "## Constraints" in report_text
    assert "## Metric summary" in report_text
    assert "## Pass/fail checklist" in report_text
    assert "## Interpretation" in report_text
    assert artifact["decision"] == "hold"


def test_promotion_gate_cli_writes_outputs(tmp_path: Path) -> None:
    report_path = tmp_path / "cli-gate.md"
    artifact_path = tmp_path / "cli-gate.json"

    exit_code = main(
        [
            "--simulation-events",
            "120",
            "--ope-events",
            "150",
            "--seed",
            "4",
            "--report-md",
            str(report_path),
            "--artifact-json",
            str(artifact_path),
        ]
    )

    assert exit_code == 0
    assert report_path.exists()
    assert artifact_path.exists()
