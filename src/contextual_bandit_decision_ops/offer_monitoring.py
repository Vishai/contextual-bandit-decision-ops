from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import numpy as np

from .bayesian_policies import BayesianLogisticThompsonSampling
from .monitoring_metrics import total_variation_distance
from .offer_api_schemas import LatencySummary
from .offer_features import OfferContext, context_from_mapping
from .offer_service import summarize_latencies
from .offer_simulation import ConversionModel
from .offers import OfferCatalog

Status = Literal["ok", "warn", "alert"]
NUMERIC_DRIFT_FEATURES = (
    "engagement_score",
    "sessions_7d",
    "sessions_30d",
    "days_since_last_visit",
    "tenure_days",
    "purchases_90d",
    "revenue_90d",
    "avg_order_value",
    "offer_impressions_30d",
    "offer_conversions_30d",
)
CATEGORICAL_DRIFT_FEATURES = ("device", "region", "has_purchased", "feature_source")
_SEVERITY = {"ok": 0, "warn": 1, "alert": 2}


@dataclass(frozen=True)
class MonitoringThresholds:
    latency_budget_ms: float = 100.0
    min_within_budget_rate: float = 0.99
    psi_warn: float = 0.1
    psi_alert: float = 0.25
    category_tvd_warn: float = 0.1
    category_tvd_alert: float = 0.2
    allocation_tvd_warn: float = 0.15
    allocation_tvd_alert: float = 0.3
    min_offer_share: float = 0.01
    min_feedback_coverage: float = 0.8
    min_propensity: float = 0.005
    conversion_drop_warn_pts: float = 2.0
    regret_increase_warn: float = 0.25


@dataclass(frozen=True)
class WindowSummary:
    label: str
    decisions: int
    start: str | None
    end: str | None
    unique_users: int
    allocation: dict[str, int]
    allocation_share: dict[str, float]
    exploration_rate: float
    latency: LatencySummary
    feedback_coverage: float
    conversion_rate: float | None
    revenue_per_decision: float | None
    conversion_by_offer: dict[str, float | None]
    min_propensity: float | None
    low_propensity_share: float
    estimated_regret_per_decision: float | None
    true_regret_per_decision: float | None


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    statistic: Literal["psi", "tvd"]
    value: float
    status: Status


@dataclass(frozen=True)
class MonitoringCheck:
    name: str
    status: Status
    detail: str


@dataclass(frozen=True)
class MonitoringResult:
    reference: WindowSummary
    current: WindowSummary
    allocation_tvd: float
    drift: list[FeatureDrift]
    allocation_over_time: list[dict[str, float]]
    checks: list[MonitoringCheck]

    @property
    def status(self) -> Status:
        return max((check.status for check in self.checks), key=_SEVERITY.__getitem__)


def population_stability_index(
    reference: Sequence[float],
    current: Sequence[float],
    bins: int = 10,
) -> float:
    """PSI with bin edges taken from reference quantiles.

    Under 0.1 is usually read as stable, 0.1 to 0.25 as a moderate shift, and above
    0.25 as a large one. Repeated values (lots of zero counts) collapse into fewer
    bins rather than producing empty ones.
    """
    reference_values = np.asarray(reference, dtype=float)
    current_values = np.asarray(current, dtype=float)
    if reference_values.size == 0 or current_values.size == 0:
        raise ValueError("both samples need at least one value")
    edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1)[1:-1]))
    n_bins = len(edges) + 1
    reference_bins = np.bincount(np.searchsorted(edges, reference_values, "right"), minlength=n_bins)
    current_bins = np.bincount(np.searchsorted(edges, current_values, "right"), minlength=n_bins)
    epsilon = 1e-4
    reference_share = np.maximum(reference_bins / reference_values.size, epsilon)
    current_share = np.maximum(current_bins / current_values.size, epsilon)
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def _status(value: float, warn: float, alert: float) -> Status:
    if value >= alert:
        return "alert"
    if value >= warn:
        return "warn"
    return "ok"


def _context(record: dict[str, Any]) -> OfferContext:
    return context_from_mapping(record["context"])


def _feature_value(record: dict[str, Any], feature: str) -> Any:
    if feature == "feature_source":
        return record.get("feature_source", "unknown")
    if feature == "has_purchased":
        return record["context"]["days_since_last_purchase"] is not None
    return record["context"][feature]


def _regret(
    records: Sequence[dict[str, Any]],
    expected_revenue: Callable[[OfferContext, Sequence[str]], np.ndarray],
    catalog: OfferCatalog,
) -> float:
    total = 0.0
    for record in records:
        offers = [o for o in record.get("available_offers", catalog.offer_ids) if o in catalog]
        values = expected_revenue(_context(record), offers)
        total += float(values.max() - values[offers.index(record["offer_id"])])
    return total / len(records)


def _posterior_expected_revenue(
    model: BayesianLogisticThompsonSampling,
) -> Callable[[OfferContext, Sequence[str]], np.ndarray]:
    def expected(context: OfferContext, offers: Sequence[str]) -> np.ndarray:
        return np.array([row.expected_revenue for row in model.posterior_summary(context, offers)])

    return expected


def summarize_window(
    label: str,
    decisions: Sequence[dict[str, Any]],
    feedback_by_event: dict[str, dict[str, Any]],
    catalog: OfferCatalog,
    thresholds: MonitoringThresholds,
    model: BayesianLogisticThompsonSampling | None = None,
    truth: ConversionModel | None = None,
) -> WindowSummary:
    if not decisions:
        raise ValueError(f"the {label} window has no decisions")
    offers = catalog.offer_ids
    allocation = Counter(record["offer_id"] for record in decisions)
    propensities = np.array([float(record["propensity"]) for record in decisions])
    latencies = [
        float(record["decision_latency_ms"])
        for record in decisions
        if record.get("decision_latency_ms") is not None
    ]
    joined = [
        (record, feedback_by_event[record["event_id"]])
        for record in decisions
        if record["event_id"] in feedback_by_event
    ]
    conversions_by_offer: dict[str, list[bool]] = {offer: [] for offer in offers}
    for record, outcome in joined:
        conversions_by_offer.setdefault(record["offer_id"], []).append(bool(outcome["converted"]))
    revenue = [
        float(outcome["revenue"]) if outcome.get("revenue") is not None
        else (catalog.revenue(record["offer_id"]) if outcome["converted"] else 0.0)
        for record, outcome in joined
    ]
    timestamps = sorted(record["timestamp"] for record in decisions)
    return WindowSummary(
        label=label,
        decisions=len(decisions),
        start=timestamps[0],
        end=timestamps[-1],
        unique_users=len({record["user_id"] for record in decisions}),
        allocation={offer: allocation.get(offer, 0) for offer in offers},
        allocation_share={offer: allocation.get(offer, 0) / len(decisions) for offer in offers},
        exploration_rate=float(np.mean([bool(r.get("explored", False)) for r in decisions])),
        latency=summarize_latencies(latencies, thresholds.latency_budget_ms),
        feedback_coverage=len(joined) / len(decisions),
        conversion_rate=float(np.mean([o["converted"] for _, o in joined])) if joined else None,
        revenue_per_decision=float(np.mean(revenue)) if revenue else None,
        conversion_by_offer={
            offer: float(np.mean(values)) if values else None
            for offer, values in conversions_by_offer.items()
        },
        min_propensity=float(propensities.min()),
        low_propensity_share=float(np.mean(propensities < thresholds.min_propensity)),
        estimated_regret_per_decision=(
            _regret(decisions, _posterior_expected_revenue(model), catalog) if model else None
        ),
        true_regret_per_decision=(
            _regret(decisions, truth.expected_revenue, catalog) if truth else None
        ),
    )


def feature_drift(
    reference: Sequence[dict[str, Any]],
    current: Sequence[dict[str, Any]],
    thresholds: MonitoringThresholds,
) -> list[FeatureDrift]:
    drift: list[FeatureDrift] = []
    for feature in NUMERIC_DRIFT_FEATURES:
        value = population_stability_index(
            [float(_feature_value(r, feature)) for r in reference],
            [float(_feature_value(r, feature)) for r in current],
        )
        drift.append(
            FeatureDrift(feature, "psi", value, _status(value, thresholds.psi_warn,
                                                        thresholds.psi_alert))
        )
    for feature in CATEGORICAL_DRIFT_FEATURES:
        reference_counts = Counter(str(_feature_value(r, feature)) for r in reference)
        current_counts = Counter(str(_feature_value(r, feature)) for r in current)
        value = total_variation_distance(
            {k: v / len(reference) for k, v in reference_counts.items()},
            {k: v / len(current) for k, v in current_counts.items()},
        )
        drift.append(
            FeatureDrift(
                feature,
                "tvd",
                value,
                _status(value, thresholds.category_tvd_warn, thresholds.category_tvd_alert),
            )
        )
    return drift


def allocation_over_time(
    decisions: Sequence[dict[str, Any]],
    catalog: OfferCatalog,
    buckets: int = 8,
) -> list[dict[str, float]]:
    ordered = sorted(decisions, key=lambda record: record["timestamp"])
    rows = []
    for chunk in np.array_split(np.arange(len(ordered)), min(buckets, len(ordered))):
        counts = Counter(ordered[i]["offer_id"] for i in chunk)
        rows.append({offer: counts.get(offer, 0) / len(chunk) for offer in catalog.offer_ids})
    return rows


def _checks(
    reference: WindowSummary,
    current: WindowSummary,
    allocation_tvd: float,
    drift: list[FeatureDrift],
    thresholds: MonitoringThresholds,
) -> list[MonitoringCheck]:
    checks: list[MonitoringCheck] = []
    latency = current.latency
    if latency.p99_ms is None:
        checks.append(MonitoringCheck("latency_p99", "warn", "no latency data was logged"))
    else:
        checks.append(
            MonitoringCheck(
                "latency_p99",
                "ok" if latency.p99_ms <= thresholds.latency_budget_ms else "alert",
                f"p99 {latency.p99_ms:.2f} ms against a {thresholds.latency_budget_ms:g} ms budget",
            )
        )
        within = latency.within_budget_rate or 0.0
        checks.append(
            MonitoringCheck(
                "latency_budget_rate",
                "ok" if within >= thresholds.min_within_budget_rate else "warn",
                f"{within:.2%} of decisions inside the budget",
            )
        )

    worst = max(drift, key=lambda d: (_SEVERITY[d.status], d.value))
    drifted = [d.feature for d in drift if d.status != "ok"]
    checks.append(
        MonitoringCheck(
            "feature_drift",
            worst.status,
            (
                f"{len(drifted)} of {len(drift)} features shifted ({', '.join(drifted)}); "
                f"largest is {worst.feature} at {worst.statistic.upper()} {worst.value:.3f}"
            )
            if drifted
            else f"all {len(drift)} features stable",
        )
    )
    checks.append(
        MonitoringCheck(
            "allocation_shift",
            _status(allocation_tvd, thresholds.allocation_tvd_warn, thresholds.allocation_tvd_alert),
            f"total variation {allocation_tvd:.3f} between the offer mixes",
        )
    )
    starved = [o for o, share in current.allocation_share.items() if share < thresholds.min_offer_share]
    checks.append(
        MonitoringCheck(
            "offer_starvation",
            "warn" if starved else "ok",
            f"offers under {thresholds.min_offer_share:.0%} of traffic: {', '.join(starved)}"
            if starved
            else "every offer still gets traffic",
        )
    )
    checks.append(
        MonitoringCheck(
            "propensity_health",
            "warn" if current.low_propensity_share > 0 else "ok",
            f"min propensity {current.min_propensity:.4f}, "
            f"{current.low_propensity_share:.2%} under {thresholds.min_propensity}",
        )
    )
    checks.append(
        MonitoringCheck(
            "feedback_coverage",
            "ok" if current.feedback_coverage >= thresholds.min_feedback_coverage else "warn",
            f"{current.feedback_coverage:.1%} of decisions have an outcome",
        )
    )
    if reference.conversion_rate is not None and current.conversion_rate is not None:
        change = (current.conversion_rate - reference.conversion_rate) * 100.0
        checks.append(
            MonitoringCheck(
                "conversion_change",
                "warn" if change <= -thresholds.conversion_drop_warn_pts else "ok",
                f"{reference.conversion_rate:.2%} to {current.conversion_rate:.2%} "
                f"({change:+.2f} pts)",
            )
        )
    if reference.estimated_regret_per_decision and current.estimated_regret_per_decision is not None:
        increase = current.estimated_regret_per_decision / reference.estimated_regret_per_decision - 1
        checks.append(
            MonitoringCheck(
                "estimated_regret",
                "warn" if increase > thresholds.regret_increase_warn else "ok",
                f"{reference.estimated_regret_per_decision:.3f} to "
                f"{current.estimated_regret_per_decision:.3f} per decision ({increase:+.0%})",
            )
        )
    return checks


def analyze_offer_logs(
    reference_decisions: Sequence[dict[str, Any]],
    current_decisions: Sequence[dict[str, Any]],
    feedback: Sequence[dict[str, Any]],
    catalog: OfferCatalog,
    thresholds: MonitoringThresholds | None = None,
    model: BayesianLogisticThompsonSampling | None = None,
    reference_truth: ConversionModel | None = None,
    current_truth: ConversionModel | None = None,
) -> MonitoringResult:
    """Compare a reference window of service logs with the current one.

    Covers latency against the budget, input drift per feature, how the offer mix
    moved, outcomes, propensity health and regret. Regret comes in two forms: as
    judged by the served model's own posterior (always available when a model is
    passed), and the true regret, which is only known in simulation.
    """
    settings = thresholds or MonitoringThresholds()
    feedback_by_event = {record["event_id"]: record for record in feedback}
    reference = summarize_window(
        "reference", reference_decisions, feedback_by_event, catalog, settings, model,
        reference_truth,
    )
    current = summarize_window(
        "current", current_decisions, feedback_by_event, catalog, settings, model, current_truth,
    )
    allocation_tvd = total_variation_distance(reference.allocation_share, current.allocation_share)
    drift = feature_drift(reference_decisions, current_decisions, settings)
    return MonitoringResult(
        reference=reference,
        current=current,
        allocation_tvd=allocation_tvd,
        drift=drift,
        allocation_over_time=allocation_over_time(
            [*reference_decisions, *current_decisions], catalog
        ),
        checks=_checks(reference, current, allocation_tvd, drift, settings),
    )


def split_by_time(
    decisions: Sequence[dict[str, Any]],
    split_at: datetime | None = None,
    reference_fraction: float = 0.5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(decisions, key=lambda record: datetime.fromisoformat(record["timestamp"]))
    if split_at is not None:
        if split_at.tzinfo is None:
            raise ValueError("split_at must be timezone aware")
        return (
            [r for r in ordered if datetime.fromisoformat(r["timestamp"]) < split_at],
            [r for r in ordered if datetime.fromisoformat(r["timestamp"]) >= split_at],
        )
    if not 0.0 < reference_fraction < 1.0:
        raise ValueError("reference_fraction must be between 0 and 1")
    index = int(len(ordered) * reference_fraction)
    return ordered[:index], ordered[index:]
