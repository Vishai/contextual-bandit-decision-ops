from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import httpx
import numpy as np

from .bayesian_policies import BayesianLogisticThompsonSampling
from .feature_store import FeatureStore, from_sql_timestamp, to_sql_timestamp
from .model_registry import ModelRegistry
from .offer_backtest import OfflineEstimate, evaluate_offline
from .offer_features import context_from_mapping
from .offer_history import LoggedOfferEvent, load_logged_events
from .offer_policies import RuleBasedOfferPolicy
from .offers import OfferCatalog, default_offer_catalog

RetrainingStatus = Literal["promoted", "rejected", "skipped"]


@dataclass(frozen=True)
class IngestSummary:
    decisions_read: int = 0
    decisions_inserted: int = 0
    feedback_read: int = 0
    feedback_applied: int = 0
    feedback_unmatched: int = 0


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class RetrainingConfig:
    store_path: Path | str
    model_dir: Path | str = Path("artifacts/models")
    decision_log: Path | str | None = None
    feedback_log: Path | str | None = None
    lookback_days: int = 28
    holdout_fraction: float = 0.25
    min_training_events: int = 1_000
    min_effective_sample_size: float = 100.0
    max_revenue_regression: float = 0.02
    max_conversion_drop_pts: float = 1.0
    min_comparison_events: int = 200
    propensity_samples: int = 300
    bootstrap_samples: int = 200
    seed: int = 17
    notify_url: str | None = None

    def __post_init__(self) -> None:
        if self.lookback_days <= 0:
            raise ValueError("lookback_days must be positive")
        if not 0.0 < self.holdout_fraction < 1.0:
            raise ValueError("holdout_fraction must be between 0 and 1")
        if self.min_training_events <= 0:
            raise ValueError("min_training_events must be positive")
        if not 0.0 <= self.max_revenue_regression < 1.0:
            raise ValueError("max_revenue_regression must be in [0, 1)")
        object.__setattr__(self, "store_path", Path(self.store_path))
        object.__setattr__(self, "model_dir", Path(self.model_dir))


@dataclass(frozen=True)
class RetrainingRun:
    status: RetrainingStatus
    reason: str
    model_version: str | None
    previous_version: str | None
    window_start: str | None
    window_end: str | None
    train_events: int
    holdout_events: int
    ingest: IngestSummary
    estimates: dict[str, OfflineEstimate] = field(default_factory=dict)
    checks: list[GateCheck] = field(default_factory=list)
    service_reloaded: bool | None = None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _log_files(path: Path | None, file_name: str) -> list[Path]:
    """A single log file, or every matching file under a folder (one per service task)."""
    if path is None:
        return []
    if path.is_dir():
        return sorted(path.rglob(file_name))
    return [path]


def _sql_time(value: str) -> str:
    return to_sql_timestamp(datetime.fromisoformat(value))


def ingest_service_logs(
    store: FeatureStore,
    decision_log: Path | None,
    feedback_log: Path | None,
    catalog: OfferCatalog | None = None,
) -> IngestSummary:
    """Load the service's JSONL logs into ``offer_events``. Running it twice is harmless.

    Either argument can be a folder, in which case every ``decisions.jsonl`` or
    ``feedback.jsonl`` under it is read. All decisions go in before any feedback,
    so feedback logged by a different task than its decision still matches up.
    Feedback logged without a revenue gets the catalog revenue of the offer shown.
    """
    catalog = catalog or default_offer_catalog()
    decisions = [
        record for path in _log_files(decision_log, "decisions.jsonl") for record in _read_jsonl(path)
    ]
    feedback = [
        record for path in _log_files(feedback_log, "feedback.jsonl") for record in _read_jsonl(path)
    ]
    inserted = applied = unmatched = 0
    connection = store.connection
    for record in decisions:
        context = context_from_mapping(record["context"])
        cursor = connection.execute(
            "INSERT OR IGNORE INTO offer_events "
            "(event_id, user_id, decided_at, offer_id, policy_name, propensity, context_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                record["event_id"],
                record["user_id"],
                _sql_time(record["timestamp"]),
                record["offer_id"],
                record.get("policy_name", "unknown"),
                float(record["propensity"]),
                json.dumps(context.to_dict(), sort_keys=True),
            ),
        )
        inserted += cursor.rowcount
    revenue_by_offer = {offer.offer_id: offer.revenue_per_conversion for offer in catalog.offers}
    for record in feedback:
        converted = bool(record["converted"])
        revenue = record.get("revenue")
        if revenue is None:
            row = connection.execute(
                "SELECT offer_id FROM offer_events WHERE event_id = ?", (record["event_id"],)
            ).fetchone()
            offer_id = row[0] if row else None
            revenue = revenue_by_offer.get(offer_id, 0.0) if converted else 0.0
        cursor = connection.execute(
            "UPDATE offer_events SET converted = ?, revenue = ?, feedback_at = ? "
            "WHERE event_id = ? AND converted IS NULL",
            (int(converted), float(revenue), _sql_time(record["timestamp"]), record["event_id"]),
        )
        if cursor.rowcount:
            applied += 1
        else:
            exists = connection.execute(
                "SELECT 1 FROM offer_events WHERE event_id = ?", (record["event_id"],)
            ).fetchone()
            unmatched += exists is None
    connection.commit()
    return IngestSummary(len(decisions), inserted, len(feedback), applied, unmatched)


def _latest_decision_time(store: FeatureStore) -> datetime | None:
    row = store.connection.execute(
        "SELECT MAX(decided_at) FROM offer_events WHERE converted IS NOT NULL"
    ).fetchone()
    return from_sql_timestamp(row[0]) if row and row[0] else None


def _fit(
    catalog: OfferCatalog,
    events: list[LoggedOfferEvent],
    label: str = "",
) -> BayesianLogisticThompsonSampling:
    policy = BayesianLogisticThompsonSampling(catalog, label=label)
    policy.fit(
        [event.context for event in events],
        [event.offer_id for event in events],
        [event.converted for event in events],
    )
    return policy


def _not_credibly_worse(
    estimate: OfflineEstimate,
    reference: OfflineEstimate,
    reference_name: str,
    config: RetrainingConfig,
) -> list[GateCheck]:
    """Do-no-harm checks against one reference policy.

    Held-out logs only overlap with a new policy on a slice of traffic, so a point
    estimate can easily come in a few percent low or high by chance. A candidate
    fails only when the whole 95% bootstrap interval of its lift sits below the
    allowed regression, meaning the logs are confident it would do worse.
    """
    revenue_ci = estimate.revenue_lift_ci or (float("-inf"), float("inf"))
    conversion_ci = estimate.conversion_lift_pts_ci or (float("-inf"), float("inf"))
    revenue_lift = estimate.revenue_dr / reference.revenue_dr - 1.0 if reference.revenue_dr else 0.0
    conversion_change = (estimate.conversion_dr - reference.conversion_dr) * 100.0
    return [
        GateCheck(
            f"revenue_vs_{reference_name}",
            revenue_ci[1] >= -config.max_revenue_regression,
            f"lift {revenue_lift:+.1%} (95% interval {revenue_ci[0]:+.1%} to "
            f"{revenue_ci[1]:+.1%}), {estimate.revenue_dr:.3f} vs {reference.revenue_dr:.3f}",
        ),
        GateCheck(
            f"conversion_vs_{reference_name}",
            conversion_ci[1] >= -config.max_conversion_drop_pts,
            f"change {conversion_change:+.2f} pts (95% interval {conversion_ci[0]:+.2f} to "
            f"{conversion_ci[1]:+.2f}), {estimate.conversion_dr:.2%} vs "
            f"{reference.conversion_dr:.2%}",
        ),
    ]


def _gate(
    config: RetrainingConfig,
    train_events: int,
    against_rules: dict[str, OfflineEstimate],
    against_current: dict[str, OfflineEstimate] | None,
    current_note: str | None = None,
) -> list[GateCheck]:
    candidate = against_rules["candidate"]
    checks = [
        GateCheck(
            "enough_training_data",
            train_events >= config.min_training_events,
            f"{train_events:,} training events (minimum {config.min_training_events:,})",
        ),
        GateCheck(
            "holdout_support",
            candidate.effective_sample_size >= config.min_effective_sample_size,
            f"effective sample size {candidate.effective_sample_size:,.0f} "
            f"(minimum {config.min_effective_sample_size:,.0f})",
        ),
        *_not_credibly_worse(candidate, against_rules["rule_based"], "rules", config),
    ]
    if against_current is not None:
        checks.extend(
            _not_credibly_worse(
                against_current["candidate"], against_current["current"], "current_model", config
            )
        )
    elif current_note is not None:
        checks.append(GateCheck("revenue_vs_current_model", True, current_note))
    return checks


def _notify(url: str) -> bool:
    try:
        response = httpx.post(f"{url.rstrip('/')}/model/reload", timeout=10.0)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def retrain(
    config: RetrainingConfig,
    catalog: OfferCatalog | None = None,
    clock: datetime | None = None,
) -> RetrainingRun:
    """One retraining pass: ingest logs, refit, check offline, promote or reject.

    The candidate is fit on the older part of the window and scored on the most
    recent part with doubly robust estimates, next to the rule-based strategy and
    the model currently in production. It only gets promoted if it clears every
    gate. A promoted model is then refit on the whole window, saved as a new
    version, and the service is told to reload if a URL was given.
    """
    offer_catalog = catalog or default_offer_catalog()
    registry = ModelRegistry(config.model_dir)
    current = registry.current(offer_catalog)
    previous_version = current.model_version if current else None

    with FeatureStore(config.store_path) as store:
        ingest = ingest_service_logs(
            store,
            Path(config.decision_log) if config.decision_log else None,
            Path(config.feedback_log) if config.feedback_log else None,
            offer_catalog,
        )
        window_end = _latest_decision_time(store)
        if window_end is None:
            run = RetrainingRun(
                "skipped", "no logged decisions with feedback yet", None, previous_version,
                None, None, 0, 0, ingest,
            )
            registry.record("skipped", {"reason": run.reason})
            return run
        trained_until = (
            datetime.fromisoformat(current.metadata["window_end"])
            if current is not None and current.metadata.get("window_end")
            else None
        )
        if trained_until is not None and trained_until >= window_end:
            reason = f"no new feedback since {previous_version} was trained"
            registry.record("skipped", {"reason": reason})
            return RetrainingRun(
                "skipped", reason, None, previous_version, None, None, 0, 0, ingest,
            )
        window_start = window_end - timedelta(days=config.lookback_days)
        events = load_logged_events(store, start=window_start, end=window_end + timedelta(seconds=1))

    split = int(len(events) * (1.0 - config.holdout_fraction))
    train, holdout = events[:split], events[split:]
    window = (window_start.isoformat(), window_end.isoformat())
    if len(train) < config.min_training_events or not holdout:
        reason = (
            f"only {len(train):,} training events in the last {config.lookback_days} days "
            f"(minimum {config.min_training_events:,})"
        )
        registry.record("skipped", {"reason": reason, "events": len(events)})
        return RetrainingRun(
            "skipped", reason, None, previous_version, *window, len(train), len(holdout), ingest,
        )

    candidate = _fit(offer_catalog, train, label="candidate")
    policies = [candidate, RuleBasedOfferPolicy(offer_catalog)]
    evaluation = {
        "propensity_samples": config.propensity_samples,
        "bootstrap_samples": config.bootstrap_samples,
    }
    estimates = evaluate_offline(
        policies,
        holdout,
        candidate,
        offer_catalog,
        np.random.default_rng(config.seed),
        **evaluation,
    )
    # The current model must only be judged on outcomes it was not trained on,
    # otherwise it gets an unfair in-sample advantage.
    against_current = None
    current_note = None
    if current is not None:
        current.policy.label = "current"
        unseen = [e for e in holdout if trained_until is None or e.decided_at > trained_until]
        if len(unseen) < config.min_comparison_events:
            current_note = (
                f"skipped: only {len(unseen):,} held-out events are newer than the current "
                f"model's training data (minimum {config.min_comparison_events:,})"
            )
    if current is not None and current_note is None:
        against_current = evaluate_offline(
            [candidate, current.policy],
            unseen,
            candidate,
            offer_catalog,
            np.random.default_rng(config.seed),
            baseline="current",
            **evaluation,
        )
    checks = _gate(config, len(train), estimates, against_current, current_note)
    summary = {
        "window_start": window[0],
        "window_end": window[1],
        "train_events": len(train),
        "holdout_events": len(holdout),
        "previous_version": previous_version,
        "estimates": {name: asdict(estimate) for name, estimate in estimates.items()},
        "checks": [asdict(check) for check in checks],
    }

    if not all(check.passed for check in checks):
        failed = ", ".join(check.name for check in checks if not check.passed)
        registry.record("rejected", {"reason": f"failed gates: {failed}", **summary})
        return RetrainingRun(
            "rejected", f"failed gates: {failed}", None, previous_version, *window,
            len(train), len(holdout), ingest, estimates, checks,
        )

    now = clock or datetime.now(UTC)
    model_version = f"offer-ts-{now:%Y%m%dT%H%M%SZ}"
    final = _fit(offer_catalog, events)
    registry.promote(final, model_version, summary, created_at=now)
    reloaded = _notify(config.notify_url) if config.notify_url else None
    return RetrainingRun(
        "promoted", "candidate passed every gate", model_version, previous_version, *window,
        len(train), len(holdout), ingest, estimates, checks, reloaded,
    )
