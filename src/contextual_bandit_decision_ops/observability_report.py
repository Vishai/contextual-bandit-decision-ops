from __future__ import annotations

import json
from pathlib import Path

from .drift_monitoring import (
    MonitoringCheck,
    ObservabilityConfig,
    ObservabilityRun,
)
from .monitoring_metrics import NumericFeatureShift
from .service_log_analysis import ServiceLogAnalysis


def _check_record(check: MonitoringCheck) -> dict[str, str]:
    return {
        "name": check.name,
        "status": check.status,
        "observed": check.observed,
        "threshold": check.threshold,
        "message": check.message,
    }


def _feature_shift_record(shift: NumericFeatureShift) -> dict[str, float]:
    return {
        "reference_mean": shift.reference.mean,
        "reference_std": shift.reference.std,
        "current_mean": shift.current.mean,
        "current_std": shift.current.std,
        "absolute_mean_difference": shift.absolute_mean_difference,
        "standardized_mean_difference": shift.standardized_mean_difference,
        "relative_std_difference": shift.relative_std_difference,
    }


def _log_summary_record(analysis: ServiceLogAnalysis) -> dict[str, object]:
    return {
        "decisions": analysis.decision_count,
        "feedback_records": analysis.feedback_count,
        "action_distribution": {
            str(action): share for action, share in analysis.action_distribution.items()
        },
        "region_distribution": analysis.region_distribution,
        "average_observed_reward": analysis.counters.average_observed_reward,
        "missing_feedback_count": analysis.counters.missing_feedback_count,
        "missing_feedback_rate": analysis.counters.missing_feedback_rate,
        "exploration_rate": analysis.counters.exploration_rate,
        "propensity": {
            "minimum": analysis.propensity_health.minimum,
            "maximum": analysis.propensity_health.maximum,
            "invalid_count": analysis.propensity_health.invalid_count,
            "low_count": analysis.propensity_health.low_count,
            "low_rate": analysis.propensity_health.low_rate,
        },
    }


def write_observability_json(
    config: ObservabilityConfig,
    run: ObservabilityRun,
) -> Path:
    config.artifact_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "reference_events": config.reference_events,
            "current_events": config.current_events,
            "seed": config.seed,
            "n_actions": config.n_actions,
            "max_action_tv_distance": config.max_action_tv_distance,
            "max_reward_rate_difference": config.max_reward_rate_difference,
            "max_standardized_mean_shift": config.max_standardized_mean_shift,
            "max_relative_std_shift": config.max_relative_std_shift,
            "max_region_tv_distance": config.max_region_tv_distance,
            "minimum_healthy_propensity": config.minimum_healthy_propensity,
            "max_missing_feedback_rate": config.max_missing_feedback_rate,
            "min_exploration_rate": config.min_exploration_rate,
        },
        "source": run.source,
        "reference": _log_summary_record(run.reference),
        "current": _log_summary_record(run.current),
        "drift_metrics": {
            "action_tv_distance": run.action_tv_distance,
            "reward_rate_difference": run.reward_rate_difference,
            "age": _feature_shift_record(run.age_shift),
            "engagement": _feature_shift_record(run.engagement_shift),
            "region_tv_distance": run.region_tv_distance,
        },
        "passed_checks": [_check_record(check) for check in run.passed_checks],
        "warnings": [_check_record(check) for check in run.warnings],
        "failed_checks": [_check_record(check) for check in run.failed_checks],
    }
    config.artifact_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return config.artifact_json


def _optional_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _append_check_section(
    lines: list[str],
    heading: str,
    checks: tuple[MonitoringCheck, ...],
) -> None:
    lines.extend(["", f"## {heading}", ""])
    if not checks:
        lines.append("- None")
        return
    lines.extend(
        [
            "| Check | Observed | Threshold | Explanation |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for check in checks:
        lines.append(f"| {check.name} | {check.observed} | {check.threshold} | {check.message} |")


def write_observability_report(
    config: ObservabilityConfig,
    run: ObservabilityRun,
) -> Path:
    config.report_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Staging Observability Report",
        "",
        "## Run configuration",
        "",
        "| Setting | Value |",
        "| --- | ---: |",
        f"| Source | {run.source} |",
        f"| Reference events | {run.reference.decision_count} |",
        f"| Current events | {run.current.decision_count} |",
        f"| Seed | {config.seed} |",
        f"| Actions | {config.n_actions} |",
        f"| Action TV warning threshold | {config.max_action_tv_distance:.4f} |",
        f"| Reward-rate warning threshold | {config.max_reward_rate_difference:.4f} |",
        f"| Minimum healthy propensity | {config.minimum_healthy_propensity:.4f} |",
        f"| Maximum missing feedback | {config.max_missing_feedback_rate:.2%} |",
        f"| Minimum exploration | {config.min_exploration_rate:.2%} |",
        "",
        "## Log summary",
        "",
        (
            "| Window | Decisions | Feedback | Missing feedback | Avg reward | "
            "Exploration | Min propensity | Max propensity |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, analysis in (
        ("reference", run.reference),
        ("current", run.current),
    ):
        lines.append(
            f"| {name} | {analysis.counters.decisions} | "
            f"{analysis.counters.feedback_records} | "
            f"{analysis.counters.missing_feedback_rate:.2%} | "
            f"{_optional_rate(analysis.counters.average_observed_reward)} | "
            f"{analysis.counters.exploration_rate:.2%} | "
            f"{_optional_rate(analysis.propensity_health.minimum)} | "
            f"{_optional_rate(analysis.propensity_health.maximum)} |"
        )

    lines.extend(
        [
            "",
            "## Action distribution",
            "",
            "| Window | Action | Count | Share |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, analysis in (
        ("reference", run.reference),
        ("current", run.current),
    ):
        for action, count in analysis.counters.action_counts.items():
            lines.append(
                f"| {name} | {action} | {count} | {analysis.action_distribution[action]:.2%} |"
            )

    lines.extend(
        [
            "",
            "## Metric table",
            "",
            "| Metric | Reference | Current | Shift |",
            "| --- | ---: | ---: | ---: |",
            (f"| Action distribution (TV) | — | — | {run.action_tv_distance:.4f} |"),
            (
                f"| Reward rate | "
                f"{_optional_rate(run.reference.counters.average_observed_reward)} | "
                f"{_optional_rate(run.current.counters.average_observed_reward)} | "
                f"{_optional_rate(run.reward_rate_difference)} |"
            ),
            (
                f"| Context age mean / std | {run.age_shift.reference.mean:.2f} / "
                f"{run.age_shift.reference.std:.2f} | "
                f"{run.age_shift.current.mean:.2f} / "
                f"{run.age_shift.current.std:.2f} | "
                f"mean_z={run.age_shift.standardized_mean_difference:.3f} |"
            ),
            (
                f"| Context engagement mean / std | "
                f"{run.engagement_shift.reference.mean:.3f} / "
                f"{run.engagement_shift.reference.std:.3f} | "
                f"{run.engagement_shift.current.mean:.3f} / "
                f"{run.engagement_shift.current.std:.3f} | "
                f"mean_z={run.engagement_shift.standardized_mean_difference:.3f} |"
            ),
            (f"| Context region distribution (TV) | — | — | {run.region_tv_distance:.4f} |"),
        ]
    )
    _append_check_section(lines, "Passed checks", run.passed_checks)
    _append_check_section(lines, "Warnings", run.warnings)
    _append_check_section(lines, "Failed checks", run.failed_checks)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Warnings identify transparent staging changes rather than proving an incident. "
                "The deterministic fixture intentionally shifts context, policy behavior, and "
                "feedback coverage so the checks remain easy to inspect."
            ),
            (
                "Invalid propensities are treated as failures because they break logged-policy "
                "reasoning; small but valid propensities remain warnings because they can make "
                "importance-weighted estimates unstable."
            ),
            (
                "This report is local/staging diagnostics only. It is not production "
                "observability, alerting, or a launch guarantee."
            ),
        ]
    )
    config.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config.report_md


def write_observability_outputs(
    config: ObservabilityConfig,
    run: ObservabilityRun,
) -> tuple[Path, Path]:
    return (
        write_observability_report(config, run),
        write_observability_json(config, run),
    )
