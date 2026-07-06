from __future__ import annotations

import json
from pathlib import Path

from .config import LearningComparisonConfig
from .learning_evaluation import LearningEvaluation

POLICY_DESCRIPTIONS = {
    "random_uniform": "Chooses every available action with equal probability.",
    "online_epsilon_greedy": (
        "Learns per-action reward averages and explores with probability epsilon."
    ),
    "linucb": "Fits one ridge-linear reward model per action and adds a confidence bonus.",
    "linear_thompson_sampling": (
        "Samples coefficients from each action's approximate linear posterior."
    ),
    "greedy_oracle": "Selects the simulator's highest-probability action as an upper bound.",
}


def _policy_description(policy_name: str) -> str:
    if policy_name.startswith("online_epsilon_greedy_"):
        return POLICY_DESCRIPTIONS["online_epsilon_greedy"]
    return POLICY_DESCRIPTIONS.get(policy_name, "Custom sequential policy.")


def _config_record(config: LearningComparisonConfig) -> dict[str, int | float]:
    return {
        "n_events": config.n_events,
        "seed": config.seed,
        "n_actions": config.n_actions,
        "epsilon": config.epsilon,
        "linucb_alpha": config.linucb_alpha,
        "thompson_scale": config.thompson_scale,
        "regularization": config.regularization,
    }


def _evaluation_record(result: LearningEvaluation) -> dict[str, object]:
    return {
        "description": _policy_description(result.policy_name),
        "total_reward": result.total_reward,
        "average_reward": result.average_reward,
        "cumulative_regret": result.cumulative_regret,
        "action_distribution": {
            str(action): count for action, count in result.action_distribution.items()
        },
        "cumulative_rewards": list(result.cumulative_rewards),
        "cumulative_regrets": list(result.cumulative_regrets),
    }


def write_learning_json(
    config: LearningComparisonConfig,
    results: dict[str, LearningEvaluation],
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


def write_learning_report(
    config: LearningComparisonConfig,
    results: dict[str, LearningEvaluation],
) -> Path:
    if not results:
        raise ValueError("results must not be empty")
    config.report_md.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Contextual Learning Policy Comparison",
        "",
        "## Run configuration",
        "",
        "| Setting | Value |",
        "| --- | ---: |",
        f"| Events | {config.n_events} |",
        f"| Seed | {config.seed} |",
        f"| Actions | {config.n_actions} |",
        f"| Epsilon | {config.epsilon:.2f} |",
        f"| LinUCB alpha | {config.linucb_alpha:.2f} |",
        f"| Thompson scale | {config.thompson_scale:.2f} |",
        f"| Regularization | {config.regularization:.2f} |",
        "",
        "## Policy descriptions",
        "",
        "| Policy | Description |",
        "| --- | --- |",
    ]
    for result in results.values():
        lines.append(f"| {result.policy_name} | {_policy_description(result.policy_name)} |")

    lines.extend(
        [
            "",
            "## Learning results",
            "",
            "| Policy | Total reward | Average reward | Cumulative regret |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for result in results.values():
        lines.append(
            f"| {result.policy_name} | {result.total_reward} | "
            f"{result.average_reward:.3f} | {result.cumulative_regret:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Final action distribution",
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

    non_oracle_results = [
        result for result in results.values() if result.policy_name != "greedy_oracle"
    ]
    best_learning_result = max(
        non_oracle_results or list(results.values()),
        key=lambda result: result.average_reward,
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"`{best_learning_result.policy_name}` had the highest realized average "
                f"reward among non-oracle policies ({best_learning_result.average_reward:.3f})."
            ),
            (
                "LinUCB and Thompson Sampling improve by updating action-specific linear "
                "models from observed rewards; their regret includes early exploration."
            ),
            (
                "The greedy oracle sees the simulator reward probabilities and is only an "
                "upper bound. This is deterministic online-style simulation, not a production "
                "online learner or off-policy evaluation."
            ),
        ]
    )

    config.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config.report_md


def write_learning_outputs(
    config: LearningComparisonConfig,
    results: dict[str, LearningEvaluation],
) -> tuple[Path, Path]:
    return (
        write_learning_report(config, results),
        write_learning_json(config, results),
    )
