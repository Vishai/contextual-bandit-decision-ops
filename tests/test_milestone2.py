import json
from pathlib import Path

import numpy as np
import pytest

from contextual_bandit_decision_ops.comparison_cli import main
from contextual_bandit_decision_ops.comparison_report import write_comparison_outputs
from contextual_bandit_decision_ops.config import PolicyComparisonConfig
from contextual_bandit_decision_ops.evaluation import compare_policies
from contextual_bandit_decision_ops.policies import (
    BanditPolicy,
    EpsilonGreedyPolicy,
    FixedActionPolicy,
    GreedyOraclePolicy,
    RandomUniformPolicy,
)
from contextual_bandit_decision_ops.schemas import UserContext


@pytest.mark.parametrize(
    "policy",
    [
        RandomUniformPolicy(),
        FixedActionPolicy(1),
        GreedyOraclePolicy(),
        EpsilonGreedyPolicy(0.2),
    ],
)
def test_policy_outputs_are_available_actions(policy: BanditPolicy) -> None:
    context = UserContext(age=35.0, engagement=0.8, region="west")
    available_actions = (0, 1, 2)
    rng = np.random.default_rng(9)

    chosen_actions = {policy.choose_action(context, available_actions, rng) for _ in range(100)}

    assert chosen_actions <= set(available_actions)


def test_evaluation_is_deterministic_with_fixed_seed() -> None:
    config = PolicyComparisonConfig(n_events=80, seed=17)

    first_results = compare_policies(config)
    second_results = compare_policies(config)

    assert first_results == second_results
    for result in first_results.values():
        assert len(result.cumulative_rewards) == config.n_events
        assert result.cumulative_rewards[-1] == result.total_reward
        assert sum(result.action_distribution.values()) == config.n_events


def test_random_uniform_policy_roughly_uses_all_actions() -> None:
    config = PolicyComparisonConfig(n_events=1_200, seed=23)
    result = compare_policies(config, policies=[RandomUniformPolicy()])["random_uniform"]

    assert set(result.action_distribution) == {0, 1, 2}
    assert all(count > 300 for count in result.action_distribution.values())


def test_greedy_oracle_performs_at_least_as_well_as_random() -> None:
    results = compare_policies(PolicyComparisonConfig())
    random_result = results["random_uniform"]
    oracle_result = results["greedy_oracle"]

    assert oracle_result.total_reward >= random_result.total_reward
    assert oracle_result.average_reward >= random_result.average_reward
    assert oracle_result.regret_estimate == pytest.approx(0.0)
    assert random_result.regret_estimate > 0.0


def test_report_and_json_artifact_creation(tmp_path: Path) -> None:
    config = PolicyComparisonConfig(
        n_events=60,
        seed=5,
        report_md=tmp_path / "comparison.md",
        artifact_json=tmp_path / "comparison.json",
    )
    results = compare_policies(config)

    report_path, artifact_path = write_comparison_outputs(config, results)

    report_text = report_path.read_text(encoding="utf-8")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert "# Baseline Policy Comparison" in report_text
    assert "## Run configuration" in report_text
    assert "## Action distribution" in report_text
    assert "## Interpretation" in report_text
    assert "greedy_oracle" in report_text
    assert artifact["config"]["n_events"] == 60
    assert len(artifact["policies"]["random_uniform"]["cumulative_rewards"]) == 60


def test_comparison_cli_writes_outputs(tmp_path: Path) -> None:
    report_path = tmp_path / "cli-report.md"
    artifact_path = tmp_path / "cli-results.json"

    exit_code = main(
        [
            "--events",
            "40",
            "--seed",
            "8",
            "--epsilon",
            "0.15",
            "--report-md",
            str(report_path),
            "--artifact-json",
            str(artifact_path),
        ]
    )

    assert exit_code == 0
    assert report_path.exists()
    assert artifact_path.exists()
