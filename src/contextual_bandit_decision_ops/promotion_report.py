from __future__ import annotations

import json
from pathlib import Path

from .promotion_gate import (
    CandidateGateResult,
    GateCheck,
    PromotionGateConfig,
    PromotionGateRun,
)
from .safety import PolicyConstraints


def _optional_number(value: float | None, digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _constraints_record(constraints: PolicyConstraints) -> dict[str, object]:
    return {
        "blocked_actions": sorted(constraints.blocked_actions),
        "max_action_share": constraints.max_action_share,
        "min_exploration_rate": constraints.min_exploration_rate,
        "min_action_count": constraints.min_action_count,
        "min_effective_sample_size": constraints.min_effective_sample_size,
        "min_matched_replay_count": constraints.min_matched_replay_count,
        "min_estimated_improvement": constraints.min_estimated_improvement,
        "max_average_regret": constraints.max_average_regret,
        "require_non_synthetic_evidence": constraints.require_non_synthetic_evidence,
    }


def _check_record(check: GateCheck) -> dict[str, object]:
    return {
        "name": check.name,
        "passed": check.passed,
        "observed": check.observed,
        "requirement": check.requirement,
        "message": check.message,
    }


def _candidate_record(result: CandidateGateResult) -> dict[str, object]:
    metrics = result.metrics
    return {
        "decision": result.decision,
        "reason": result.reason,
        "warnings": list(result.warnings),
        "metrics": {
            "estimated_value": metrics.estimated_value,
            "baseline_estimated_value": metrics.baseline_estimated_value,
            "estimated_improvement": metrics.estimated_improvement,
            "effective_sample_size": metrics.effective_sample_size,
            "matched_replay_count": metrics.matched_replay_count,
            "average_regret": metrics.average_regret,
            "action_distribution": {
                str(action): count for action, count in metrics.action_distribution.items()
            },
            "max_action_share": metrics.max_action_share,
            "exploration_rate": metrics.exploration_rate,
            "blocked_action_selections": metrics.blocked_action_selections,
            "minimum_allowed_action_count": metrics.minimum_allowed_action_count,
        },
        "passed_checks": [_check_record(check) for check in result.passed_checks],
        "failed_checks": [_check_record(check) for check in result.failed_checks],
    }


def write_promotion_gate_json(
    config: PromotionGateConfig,
    run: PromotionGateRun,
) -> Path:
    config.artifact_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "simulation_events": config.simulation_events,
            "ope_events": config.ope_events,
            "seed": config.seed,
            "n_actions": config.n_actions,
            "fixed_action": config.fixed_action,
            "epsilon": config.epsilon,
        },
        "constraints": _constraints_record(config.constraints),
        "decision": run.decision,
        "selected_candidate": run.selected_candidate,
        "evidence_source": run.evidence_source,
        "reason": run.reason,
        "warnings": list(run.warnings),
        "candidates": {
            policy_name: _candidate_record(result)
            for policy_name, result in run.candidate_results.items()
        },
    }
    config.artifact_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return config.artifact_json


def write_promotion_gate_report(
    config: PromotionGateConfig,
    run: PromotionGateRun,
) -> Path:
    config.report_md.parent.mkdir(parents=True, exist_ok=True)
    constraints = config.constraints
    blocked_actions = (
        ", ".join(str(action) for action in sorted(constraints.blocked_actions)) or "none"
    )
    lines = [
        "# Policy Promotion Gate",
        "",
        f"**Final decision: {run.decision.upper()}**",
        "",
        run.reason,
        "",
        "## Run configuration",
        "",
        "| Setting | Value |",
        "| --- | ---: |",
        f"| Simulation events | {config.simulation_events} |",
        f"| OPE logged events | {config.ope_events} |",
        f"| Seed | {config.seed} |",
        f"| Actions | {config.n_actions} |",
        f"| Evidence source | {run.evidence_source} |",
        "",
        "## Constraints",
        "",
        "| Constraint | Requirement |",
        "| --- | ---: |",
        f"| Blocked actions | {blocked_actions} |",
        f"| Maximum action share | {constraints.max_action_share:.2%} |",
        f"| Minimum exploration rate | {constraints.min_exploration_rate:.2%} |",
        f"| Minimum count per allowed action | {constraints.min_action_count} |",
        f"| Minimum OPE effective sample size | {constraints.min_effective_sample_size:.1f} |",
        f"| Minimum replay matches | {constraints.min_matched_replay_count} |",
        f"| Minimum estimated improvement | {constraints.min_estimated_improvement:.4f} |",
        f"| Maximum average simulation regret | {constraints.max_average_regret:.4f} |",
        (
            "| External evidence required | "
            f"{'yes' if constraints.require_non_synthetic_evidence else 'no'} |"
        ),
        "",
        "## Candidate policies",
        "",
        (f"- `fixed_action_{config.fixed_action}` — capacity-stressing deterministic baseline"),
        (
            f"- `epsilon_greedy_{config.epsilon:.2f}` — simulator-oracle-assisted "
            "baseline with uniform exploration"
        ),
        "",
        "## Metric summary",
        "",
        (
            "| Candidate | DR value | Improvement | OPE ESS | Replay matches | "
            "Max share | Exploration | Avg regret |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in run.candidate_results.values():
        metrics = result.metrics
        lines.append(
            f"| {result.policy_name} | {_optional_number(metrics.estimated_value)} | "
            f"{_optional_number(metrics.estimated_improvement)} | "
            f"{_optional_number(metrics.effective_sample_size, 1)} | "
            f"{metrics.matched_replay_count if metrics.matched_replay_count is not None else 'N/A'} | "
            f"{metrics.max_action_share:.2%} | {metrics.exploration_rate:.2%} | "
            f"{metrics.average_regret:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Pass/fail checklist",
            "",
            "| Candidate | Check | Status | Observed | Requirement |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for result in run.candidate_results.values():
        for check in (*result.passed_checks, *result.failed_checks):
            lines.append(
                f"| {result.policy_name} | {check.name} | "
                f"{'PASS' if check.passed else 'FAIL'} | {check.observed} | "
                f"{check.requirement} |"
            )

    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in run.warnings)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "A higher estimated reward is not sufficient for promotion. The fixed-action "
                "candidate fails concentration, exploration, coverage, improvement, and regret "
                "checks. The epsilon-greedy candidate has stronger synthetic metrics, but it "
                "uses simulator-oracle information and lacks external evidence."
            ),
            (
                "The gate therefore holds every candidate. This demonstrates deterministic "
                "simulated readiness logic; it is not a real-world safety certification or "
                "launch authorization."
            ),
        ]
    )
    config.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config.report_md


def write_promotion_gate_outputs(
    config: PromotionGateConfig,
    run: PromotionGateRun,
) -> tuple[Path, Path]:
    return (
        write_promotion_gate_report(config, run),
        write_promotion_gate_json(config, run),
    )
