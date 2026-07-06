import json
from pathlib import Path

import numpy as np
import pytest

from contextual_bandit_decision_ops.config import LearningComparisonConfig
from contextual_bandit_decision_ops.learning_cli import main
from contextual_bandit_decision_ops.learning_evaluation import compare_learning_policies
from contextual_bandit_decision_ops.learning_policies import (
    LinearThompsonSamplingPolicy,
    LinUCBPolicy,
)
from contextual_bandit_decision_ops.learning_report import write_learning_outputs
from contextual_bandit_decision_ops.schemas import UserContext


@pytest.mark.parametrize(
    "policy",
    [
        LinUCBPolicy(n_actions=3),
        LinearThompsonSamplingPolicy(n_actions=3),
    ],
)
def test_contextual_policy_actions_are_valid(
    policy: LinUCBPolicy | LinearThompsonSamplingPolicy,
) -> None:
    context = UserContext(age=42.0, engagement=0.7, region="east")
    actions = (0, 1, 2)
    rng = np.random.default_rng(12)

    chosen_actions = {policy.choose_action(context, actions, rng) for _ in range(50)}

    assert chosen_actions <= set(actions)


def test_learning_evaluation_is_deterministic() -> None:
    config = LearningComparisonConfig(n_events=200, seed=7)

    first_results = compare_learning_policies(config)
    second_results = compare_learning_policies(config)

    assert first_results == second_results
    for result in first_results.values():
        assert result.cumulative_rewards[-1] == result.total_reward
        assert result.cumulative_regrets[-1] == pytest.approx(result.cumulative_regret)
        assert sum(result.action_distribution.values()) == config.n_events


@pytest.mark.parametrize(
    "policy",
    [
        LinUCBPolicy(n_actions=3),
        LinearThompsonSamplingPolicy(n_actions=3),
    ],
)
def test_contextual_policy_parameters_update(
    policy: LinUCBPolicy | LinearThompsonSamplingPolicy,
) -> None:
    context = UserContext(age=31.0, engagement=0.8, region="north")
    precision_before = policy.precision_matrices.copy()
    rewards_before = policy.reward_vectors.copy()

    policy.update(context, action=1, reward=1.0)

    assert not np.array_equal(policy.precision_matrices, precision_before)
    assert not np.array_equal(policy.reward_vectors, rewards_before)


def test_contextual_policies_beat_random_over_default_run() -> None:
    results = compare_learning_policies(LearningComparisonConfig())
    random_reward = results["random_uniform"].total_reward

    assert results["linucb"].total_reward > random_reward
    assert results["linear_thompson_sampling"].total_reward > random_reward
    assert results["linucb"].cumulative_regret < results["random_uniform"].cumulative_regret
    assert (
        results["linear_thompson_sampling"].cumulative_regret
        < results["random_uniform"].cumulative_regret
    )


def test_learning_report_and_json_creation(tmp_path: Path) -> None:
    config = LearningComparisonConfig(
        n_events=80,
        seed=10,
        report_md=tmp_path / "learning.md",
        artifact_json=tmp_path / "learning.json",
    )
    results = compare_learning_policies(config)

    report_path, artifact_path = write_learning_outputs(config, results)

    report_text = report_path.read_text(encoding="utf-8")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert "# Contextual Learning Policy Comparison" in report_text
    assert "## Policy descriptions" in report_text
    assert "Cumulative regret" in report_text
    assert "## Final action distribution" in report_text
    assert "## Interpretation" in report_text
    assert len(artifact["policies"]["linucb"]["cumulative_regrets"]) == 80


def test_learning_cli_writes_outputs(tmp_path: Path) -> None:
    report_path = tmp_path / "cli-learning.md"
    artifact_path = tmp_path / "cli-learning.json"

    exit_code = main(
        [
            "--events",
            "60",
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
