from __future__ import annotations

import json
from pathlib import Path

from .config import PolicyComparisonConfig
from .evaluation import PolicyEvaluation


def _config_record(config: PolicyComparisonConfig) -> dict[str, int | float]:
    return {
        "n_events": config.n_events,
        "seed": config.seed,
        "n_actions": config.n_actions,
        "fixed_action": config.fixed_action,
        "epsilon": config.epsilon,
    }


def _evaluation_record(result: PolicyEvaluation) -> dict[str, object]:
    return {
        "total_reward": result.total_reward,
        "average_reward": result.average_reward,
        "regret_estimate": result.regret_estimate,
        "action_distribution": {
            str(action): count for action, count in result.action_distribution.items()
        },
        "cumulative_rewards": list(result.cumulative_rewards),
    }


def write_comparison_json(
    config: PolicyComparisonConfig,
    results: dict[str, PolicyEvaluation],
) -> Path:
    config.artifact_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": _config_record(config),
        "policies": {
            policy_name: _evaluation_record(result) for policy_name, result in results.items()
        },
    }
    config.artifact_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return config.artifact_json


def write_comparison_report(
    config: PolicyComparisonConfig,
    results: dict[str, PolicyEvaluation],
) -> Path:
    if not results:
        raise ValueError("results must not be empty")
    config.report_md.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Baseline Policy Comparison",
        "",
        "## Run configuration",
        "",
        "| Setting | Value |",
        "| --- | ---: |",
        f"| Events | {config.n_events} |",
        f"| Seed | {config.seed} |",
        f"| Actions | {config.n_actions} |",
        f"| Fixed action | {config.fixed_action} |",
        f"| Epsilon | {config.epsilon:.2f} |",
        "",
        "## Policy results",
        "",
        "| Policy | Total reward | Average reward | Regret estimate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for result in results.values():
        lines.append(
            f"| {result.policy_name} | {result.total_reward} | "
            f"{result.average_reward:.3f} | {result.regret_estimate:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Action distribution",
            "",
            "| Policy | Action | Count | Share |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for result in results.values():
        for action, count in result.action_distribution.items():
            lines.append(
                f"| {result.policy_name} | {action} | {count} | {count / config.n_events:.2%} |"
            )

    best_result = max(results.values(), key=lambda result: result.average_reward)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"`{best_result.policy_name}` had the highest realized average reward "
                f"({best_result.average_reward:.3f}) on this deterministic replay."
            ),
            (
                "The greedy oracle can inspect the simulator reward model, so it is an "
                "upper-bound baseline rather than a deployable online policy."
            ),
            (
                "Regret is the average expected reward-probability gap to that oracle. "
                "These results describe an offline simulator comparison, not real online learning."
            ),
        ]
    )

    config.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config.report_md


def write_comparison_outputs(
    config: PolicyComparisonConfig,
    results: dict[str, PolicyEvaluation],
) -> tuple[Path, Path]:
    return (
        write_comparison_report(config, results),
        write_comparison_json(config, results),
    )
