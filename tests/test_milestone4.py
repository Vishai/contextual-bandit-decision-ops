import json
from pathlib import Path

import pytest

from contextual_bandit_decision_ops.config import OffPolicyEvaluationConfig
from contextual_bandit_decision_ops.off_policy import (
    estimate_doubly_robust,
    estimate_ips,
    estimate_replay_matching,
    estimate_snips,
    run_off_policy_evaluation,
)
from contextual_bandit_decision_ops.off_policy_cli import main
from contextual_bandit_decision_ops.off_policy_report import write_off_policy_outputs


def test_off_policy_estimator_outputs_are_valid() -> None:
    run = run_off_policy_evaluation(OffPolicyEvaluationConfig(n_events=400, seed=9))

    assert run.row_count == 400
    assert set(run.policies) == {
        "random_uniform",
        "fixed_action_0",
        "epsilon_greedy_0.10",
        "greedy_oracle",
    }
    for evaluation in run.policies.values():
        assert 0.0 <= evaluation.simulator_value <= 1.0
        for result in evaluation.estimators.values():
            assert result.value is None or 0.0 <= result.value <= 1.0
            if result.effective_sample_size is not None:
                assert 0.0 <= result.effective_sample_size <= run.row_count


def test_off_policy_evaluation_is_deterministic() -> None:
    config = OffPolicyEvaluationConfig(n_events=250, seed=13)

    first_run = run_off_policy_evaluation(config)
    second_run = run_off_policy_evaluation(config)

    assert first_run == second_run


def test_ips_and_snips_handle_zero_target_probability() -> None:
    rewards = [1.0, 0.0, 1.0]
    propensities = [0.5, 0.5, 0.5]
    target_probabilities = [0.0, 0.0, 0.0]

    ips = estimate_ips(rewards, propensities, target_probabilities)
    snips = estimate_snips(rewards, propensities, target_probabilities)

    assert ips.value == 0.0
    assert ips.effective_sample_size == 0.0
    assert snips.value is None
    assert snips.effective_sample_size == 0.0


def test_replay_estimator_returns_matched_count() -> None:
    result = estimate_replay_matching(
        rewards=[1.0, 0.0, 1.0, 1.0],
        logged_actions=[0, 1, 2, 1],
        target_actions=[0, 0, 2, 0],
    )

    assert result.matched_count == 2
    assert result.value == 1.0


def test_doubly_robust_combines_model_and_importance_correction() -> None:
    result = estimate_doubly_robust(
        rewards=[1.0, 0.0],
        propensities=[0.5, 0.5],
        logged_target_probabilities=[0.5, 0.0],
        target_model_predictions=[0.4, 0.6],
        logged_model_predictions=[0.2, 0.1],
    )

    assert result.value == pytest.approx(0.9)
    assert result.value != pytest.approx(0.5)  # model-only mean and IPS are both 0.5


def test_off_policy_report_and_json_creation(tmp_path: Path) -> None:
    config = OffPolicyEvaluationConfig(
        n_events=120,
        seed=3,
        report_md=tmp_path / "ope.md",
        artifact_json=tmp_path / "ope.json",
    )
    run = run_off_policy_evaluation(config)

    report_path, artifact_path = write_off_policy_outputs(config, run)

    report_text = report_path.read_text(encoding="utf-8")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert "# Off-Policy Evaluation" in report_text
    assert "Behavior policy" in report_text
    assert "## Estimator results" in report_text
    assert "effective sample size" in report_text.lower()
    assert "## Why weighting can be unstable" in report_text
    assert "## Interpretation" in report_text
    assert "doubly_robust" in artifact["target_policies"]["greedy_oracle"]["estimators"]


def test_off_policy_cli_writes_outputs(tmp_path: Path) -> None:
    report_path = tmp_path / "cli-ope.md"
    artifact_path = tmp_path / "cli-ope.json"

    exit_code = main(
        [
            "--events",
            "100",
            "--seed",
            "6",
            "--report-md",
            str(report_path),
            "--artifact-json",
            str(artifact_path),
        ]
    )

    assert exit_code == 0
    assert report_path.exists()
    assert artifact_path.exists()
