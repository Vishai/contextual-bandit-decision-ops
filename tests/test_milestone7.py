import json
from pathlib import Path

import pytest

from contextual_bandit_decision_ops.drift_monitoring import (
    ObservabilityConfig,
    analyze_observability,
    run_observability,
)
from contextual_bandit_decision_ops.monitoring_metrics import (
    compare_numeric_feature,
    distribution,
    service_counter_summary,
    total_variation_distance,
)
from contextual_bandit_decision_ops.observability_cli import main
from contextual_bandit_decision_ops.observability_report import (
    write_observability_outputs,
)
from contextual_bandit_decision_ops.service_log_analysis import ServiceLogWindow


def _window(
    actions: list[int],
    propensities: list[float],
    rewards: list[float],
) -> ServiceLogWindow:
    decisions = tuple(
        {
            "event_id": f"event-{index}",
            "context": {
                "age": 30.0 + index,
                "engagement": 0.4 + (0.05 * index),
                "region": "north" if index % 2 == 0 else "south",
            },
            "action": action,
            "propensity": propensities[index],
        }
        for index, action in enumerate(actions)
    )
    feedback = tuple(
        {
            "event_id": f"event-{index}",
            "reward": reward,
        }
        for index, reward in enumerate(rewards)
    )
    return ServiceLogWindow(decisions=decisions, feedback=feedback)


def test_monitoring_metrics_compute_on_small_fixtures() -> None:
    reference_distribution = distribution([0, 0, 1, 1], [0, 1])
    current_distribution = distribution([0, 0, 0, 1], [0, 1])
    feature_shift = compare_numeric_feature([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
    counters = service_counter_summary(
        decision_event_ids=["a", "b", "c"],
        actions=[0, 0, 1],
        feedback_event_ids=["a", "c"],
        rewards=[1.0, 0.0],
        available_actions=[0, 1],
    )

    assert reference_distribution == {0: 0.5, 1: 0.5}
    assert total_variation_distance(
        reference_distribution,
        current_distribution,
    ) == pytest.approx(0.25)
    assert feature_shift.absolute_mean_difference == pytest.approx(1.0)
    assert counters.missing_feedback_count == 1
    assert counters.missing_feedback_rate == pytest.approx(1 / 3)
    assert counters.average_observed_reward == pytest.approx(0.5)


def test_action_reward_and_missing_feedback_warnings_trigger() -> None:
    config = ObservabilityConfig(
        reference_events=4,
        current_events=4,
        n_actions=2,
        max_action_tv_distance=0.20,
        max_reward_rate_difference=0.20,
        max_missing_feedback_rate=0.20,
    )
    reference = _window(
        actions=[0, 0, 1, 1],
        propensities=[0.5, 0.5, 0.5, 0.5],
        rewards=[0.0, 0.0, 0.0, 0.0],
    )
    current = _window(
        actions=[0, 0, 0, 0],
        propensities=[1.0, 1.0, 1.0, 1.0],
        rewards=[1.0, 1.0],
    )

    run = analyze_observability(
        config,
        reference,
        current,
        source="small test fixture",
    )
    warning_names = {check.name for check in run.warnings}

    assert run.action_tv_distance == pytest.approx(0.5)
    assert run.reward_rate_difference == pytest.approx(1.0)
    assert "action_distribution_shift" in warning_names
    assert "reward_rate_shift" in warning_names
    assert "missing_feedback_rate" in warning_names
    assert "low_exploration" in warning_names


def test_propensity_health_catches_invalid_and_low_values() -> None:
    config = ObservabilityConfig(
        reference_events=4,
        current_events=4,
        n_actions=2,
        minimum_healthy_propensity=0.05,
    )
    reference = _window(
        actions=[0, 1, 0, 1],
        propensities=[0.5, 0.5, 0.5, 0.5],
        rewards=[0.0, 1.0, 0.0, 1.0],
    )
    current = _window(
        actions=[0, 1, 0, 1],
        propensities=[0.01, 0.0, -0.1, 1.2],
        rewards=[0.0, 1.0, 0.0, 1.0],
    )

    run = analyze_observability(
        config,
        reference,
        current,
        source="propensity test fixture",
    )

    assert run.current.propensity_health.invalid_count == 3
    assert run.current.propensity_health.low_count == 1
    assert "propensity_validity" in {check.name for check in run.failed_checks}
    assert "low_propensity" in {check.name for check in run.warnings}


def test_observability_report_generation_is_deterministic(tmp_path: Path) -> None:
    config_a = ObservabilityConfig(
        reference_events=120,
        current_events=120,
        seed=8,
        report_md=tmp_path / "a.md",
        artifact_json=tmp_path / "a.json",
    )
    config_b = ObservabilityConfig(
        reference_events=120,
        current_events=120,
        seed=8,
        report_md=tmp_path / "b.md",
        artifact_json=tmp_path / "b.json",
    )

    write_observability_outputs(config_a, run_observability(config_a))
    write_observability_outputs(config_b, run_observability(config_b))

    assert config_a.report_md.read_bytes() == config_b.report_md.read_bytes()
    assert config_a.artifact_json.read_bytes() == config_b.artifact_json.read_bytes()
    report_text = config_a.report_md.read_text(encoding="utf-8")
    assert "# Staging Observability Report" in report_text
    assert "## Log summary" in report_text
    assert "## Metric table" in report_text
    assert "## Passed checks" in report_text
    assert "## Warnings" in report_text
    assert "## Failed checks" in report_text


def test_observability_cli_writes_outputs(tmp_path: Path) -> None:
    report_path = tmp_path / "observability.md"
    artifact_path = tmp_path / "observability.json"

    exit_code = main(
        [
            "--reference-events",
            "100",
            "--current-events",
            "100",
            "--seed",
            "5",
            "--report-md",
            str(report_path),
            "--artifact-json",
            str(artifact_path),
        ]
    )

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report_path.exists()
    assert artifact["source"] == "deterministic staging fixture logs"
