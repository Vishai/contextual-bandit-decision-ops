from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .monitoring_metrics import (
    NumericFeatureSummary,
    PropensityHealth,
    ServiceCounterSummary,
    distribution,
    numeric_feature_summary,
    propensity_health,
    service_counter_summary,
)
from .schemas import REGIONS


@dataclass(frozen=True)
class ServiceLogWindow:
    decisions: tuple[dict[str, Any], ...]
    feedback: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ServiceLogAnalysis:
    decision_count: int
    feedback_count: int
    action_distribution: dict[int, float]
    region_distribution: dict[str, float]
    age_summary: NumericFeatureSummary
    engagement_summary: NumericFeatureSummary
    propensity_health: PropensityHealth
    counters: ServiceCounterSummary
    ages: tuple[float, ...]
    engagements: tuple[float, ...]


def load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.exists():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        records.append(record)
    return tuple(records)


def load_service_log_window(
    decision_path: Path,
    feedback_path: Path,
) -> ServiceLogWindow:
    return ServiceLogWindow(
        decisions=load_jsonl(decision_path),
        feedback=load_jsonl(feedback_path),
    )


def _required(record: dict[str, Any], field: str, record_type: str) -> Any:
    if field not in record:
        raise ValueError(f"{record_type} record is missing {field}")
    return record[field]


def analyze_service_logs(
    window: ServiceLogWindow,
    available_actions: Sequence[int],
    minimum_healthy_propensity: float,
) -> ServiceLogAnalysis:
    if not window.decisions:
        raise ValueError("decision log must not be empty")

    decision_event_ids: list[str] = []
    actions: list[int] = []
    propensities: list[float] = []
    ages: list[float] = []
    engagements: list[float] = []
    regions: list[str] = []
    for record in window.decisions:
        decision_event_ids.append(str(_required(record, "event_id", "decision")))
        actions.append(int(_required(record, "action", "decision")))
        propensities.append(float(_required(record, "propensity", "decision")))
        context = _required(record, "context", "decision")
        if not isinstance(context, dict):
            raise ValueError("decision context must be a JSON object")
        ages.append(float(_required(context, "age", "decision context")))
        engagements.append(float(_required(context, "engagement", "decision context")))
        regions.append(str(_required(context, "region", "decision context")))

    feedback_event_ids: list[str] = []
    rewards: list[float] = []
    for record in window.feedback:
        feedback_event_ids.append(str(_required(record, "event_id", "feedback")))
        rewards.append(float(_required(record, "reward", "feedback")))

    counters = service_counter_summary(
        decision_event_ids,
        actions,
        feedback_event_ids,
        rewards,
        available_actions,
    )
    return ServiceLogAnalysis(
        decision_count=len(window.decisions),
        feedback_count=len(window.feedback),
        action_distribution=distribution(actions, available_actions),
        region_distribution=distribution(regions, REGIONS),
        age_summary=numeric_feature_summary(ages),
        engagement_summary=numeric_feature_summary(engagements),
        propensity_health=propensity_health(
            propensities,
            minimum_healthy_propensity,
        ),
        counters=counters,
        ages=tuple(ages),
        engagements=tuple(engagements),
    )
