from __future__ import annotations

import json
from pathlib import Path

from .config import OffPolicyEvaluationConfig
from .off_policy import EstimatorResult, OffPolicyEvaluationRun

ESTIMATOR_LABELS = {
    "direct_logged_average": "Direct logged average",
    "replay_matching": "Replay / matching",
    "ips": "IPS",
    "snips": "SNIPS",
    "doubly_robust": "Doubly robust",
}


def _target_policy_description(policy_name: str) -> str:
    if policy_name == "random_uniform":
        return "uniform probability over available actions"
    if policy_name.startswith("fixed_action_"):
        return "always chooses the configured action"
    if policy_name.startswith("epsilon_greedy_"):
        return "simulator-oracle greedy action with uniform epsilon exploration"
    if policy_name == "greedy_oracle":
        return "simulator-only upper-bound policy"
    return "custom target policy"


def _format_value(value: float | None, digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _estimator_record(result: EstimatorResult) -> dict[str, float | int | None]:
    return {
        "value": result.value,
        "effective_sample_size": result.effective_sample_size,
        "matched_count": result.matched_count,
    }


def write_off_policy_json(
    config: OffPolicyEvaluationConfig,
    run: OffPolicyEvaluationRun,
) -> Path:
    config.artifact_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "n_events": config.n_events,
            "seed": config.seed,
            "n_actions": config.n_actions,
            "fixed_action": config.fixed_action,
            "epsilon": config.epsilon,
            "reward_model_regularization": config.reward_model_regularization,
        },
        "logged_data": {
            "row_count": run.row_count,
            "behavior_policy": run.behavior_policy,
            "observed_behavior_value": run.observed_behavior_value,
            "expected_behavior_value": run.expected_behavior_value,
        },
        "target_policies": {
            policy_name: {
                "simulator_value": evaluation.simulator_value,
                "estimators": {
                    estimator_name: _estimator_record(result)
                    for estimator_name, result in evaluation.estimators.items()
                },
            }
            for policy_name, evaluation in run.policies.items()
        },
    }
    config.artifact_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return config.artifact_json


def write_off_policy_report(
    config: OffPolicyEvaluationConfig,
    run: OffPolicyEvaluationRun,
) -> Path:
    config.report_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Off-Policy Evaluation",
        "",
        "## Run configuration",
        "",
        "| Setting | Value |",
        "| --- | ---: |",
        f"| Logged events | {run.row_count} |",
        f"| Seed | {config.seed} |",
        f"| Actions | {config.n_actions} |",
        f"| Behavior policy | {run.behavior_policy} |",
        f"| Observed behavior reward | {run.observed_behavior_value:.4f} |",
        (
            "| Expected behavior reward from simulator | "
            f"{_format_value(run.expected_behavior_value)} |"
        ),
        f"| Reward-model regularization | {config.reward_model_regularization:.2f} |",
        "",
        "## Target policies",
        "",
    ]
    lines.extend(
        f"- `{policy_name}` — {_target_policy_description(policy_name)}"
        for policy_name in run.policies
    )
    lines.extend(
        [
            "",
            "## Estimator results",
            "",
            "| Target policy | Estimator | Estimated value | ESS | Matched rows |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for evaluation in run.policies.values():
        for estimator_name, result in evaluation.estimators.items():
            effective_sample_size = _format_value(result.effective_sample_size, digits=1)
            matched_count = "N/A" if result.matched_count is None else str(result.matched_count)
            lines.append(
                f"| {evaluation.policy_name} | {ESTIMATOR_LABELS[estimator_name]} | "
                f"{_format_value(result.value)} | {effective_sample_size} | "
                f"{matched_count} |"
            )
        lines.append(
            f"| {evaluation.policy_name} | Simulator truth (reference) | "
            f"{evaluation.simulator_value:.4f} | N/A | N/A |"
        )

    lines.extend(
        [
            "",
            "## Why weighting can be unstable",
            "",
            (
                "IPS divides each target-policy probability by the logged propensity. "
                "Rare logged actions therefore create large weights, high variance, and a "
                "small effective sample size. SNIPS normalizes the weights, which controls "
                "their overall scale but does not repair weak action support or eliminate variance."
            ),
            "",
            "## Interpretation",
            "",
            (
                "The direct logged average measures the behavior policy and is repeated only "
                "as a deliberately naive target-policy comparison. Replay is intuitive but "
                "discards non-matching rows."
            ),
            (
                "In this synthetic run, the behavior policy has known uniform propensities and "
                "positive support for every action, so IPS and SNIPS are meaningful. The doubly "
                "robust estimate combines a fitted linear reward prediction with an IPS residual "
                "correction and is generally the most informative estimate when either component "
                "is reasonably specified."
            ),
            (
                "Simulator truth is shown only for validation. These estimates do not establish "
                "real production policy quality."
            ),
        ]
    )
    config.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config.report_md


def write_off_policy_outputs(
    config: OffPolicyEvaluationConfig,
    run: OffPolicyEvaluationRun,
) -> tuple[Path, Path]:
    return (
        write_off_policy_report(config, run),
        write_off_policy_json(config, run),
    )
