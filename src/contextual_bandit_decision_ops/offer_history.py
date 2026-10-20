from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from .bayesian_policies import OfferPolicy
from .feature_store import FeatureStore, from_sql_timestamp, to_sql_timestamp
from .offer_features import OfferContext, context_from_mapping, new_user_context
from .offer_simulation import ConversionModel
from .synthetic_activity import SyntheticPopulation

FEEDBACK_DELAY = timedelta(hours=1)


@dataclass(frozen=True)
class LoggedOfferEvent:
    event_id: str
    decided_at: datetime
    context: OfferContext
    offer_id: str
    policy_name: str
    propensity: float
    converted: bool
    revenue: float
    available_offers: tuple[str, ...] | None = None
    """Offers the user could have been shown. None means the whole catalog."""

    def offers(self, catalog_offers: Sequence[str]) -> tuple[str, ...]:
        if self.available_offers is None:
            return tuple(catalog_offers)
        return tuple(o for o in catalog_offers if o in self.available_offers)


def log_historical_decisions(
    store: FeatureStore,
    population: SyntheticPopulation,
    model: ConversionModel,
    logging_policy: OfferPolicy,
    seed: int,
) -> int:
    """Replay the simulation window day by day as the old production system would have.

    Each morning the feature snapshot is rebuilt, so it includes the offers shown
    and conversions seen so far. Every visit that day gets an offer from the
    logging policy, and the decision, its exact propensity and the outcome go into
    ``offer_events``. The logging policy must report choice probabilities.
    """
    if not hasattr(logging_policy, "action_probabilities"):
        raise ValueError("the logging policy must expose action_probabilities")
    config = population.config
    offers = model.catalog.offer_ids
    revenue = model.catalog.revenue_vector()
    rng = np.random.default_rng(seed)
    device_by_user = {user.user_id: (user.device, user.region) for user in population.users}
    sessions_by_day: dict[int, list] = {}
    for session in population.decision_sessions():
        sessions_by_day.setdefault((session.started_at - config.start).days, []).append(session)

    logged = 0
    for day, morning in enumerate(config.snapshot_times()):
        store.materialize_features(morning)
        snapshot = store.snapshot(morning)
        for session in sessions_by_day.get(day, []):
            context = snapshot.get(session.user_id)
            if context is None:
                device, region = device_by_user[session.user_id]
                context = new_user_context(session.user_id, device, region)
            probabilities = logging_policy.action_probabilities(  # type: ignore[attr-defined]
                context, offers, rng
            )
            offer_id = logging_policy.choose(context, offers, rng)
            index = offers.index(offer_id)
            converted = bool(rng.random() < model.conversion_probability(context, offer_id))
            event_id = f"hist-{logged:07d}"
            store.record_offer_decision(
                event_id=event_id,
                context=context,
                decided_at=session.started_at,
                offer_id=offer_id,
                policy_name=logging_policy.name,
                propensity=probabilities[offer_id],
                available_offers=offers,
                commit=False,
            )
            store.record_offer_feedback(
                event_id=event_id,
                converted=converted,
                revenue=float(revenue[index]) if converted else 0.0,
                feedback_at=session.started_at + FEEDBACK_DELAY,
                commit=False,
            )
            logged += 1
        store.commit()
    return logged


LOGGED_EVENTS_QUERY = """
SELECT event_id, decided_at, offer_id, policy_name, propensity, context_json,
       available_offers, converted, revenue
FROM offer_events
WHERE converted IS NOT NULL
  AND decided_at >= :start
  AND decided_at < :end
ORDER BY decided_at, event_id
"""


def load_logged_events(
    store: FeatureStore,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[LoggedOfferEvent]:
    """Pull logged decisions that already have feedback, in time order."""
    rows = store.connection.execute(
        LOGGED_EVENTS_QUERY,
        {
            "start": to_sql_timestamp(start) if start else "0000-01-01 00:00:00",
            "end": to_sql_timestamp(end) if end else "9999-12-31 23:59:59",
        },
    ).fetchall()
    return [
        LoggedOfferEvent(
            event_id=row["event_id"],
            decided_at=from_sql_timestamp(row["decided_at"]),
            context=context_from_mapping(json.loads(row["context_json"])),
            offer_id=row["offer_id"],
            policy_name=row["policy_name"],
            propensity=float(row["propensity"]),
            converted=bool(row["converted"]),
            revenue=float(row["revenue"]),
            available_offers=(
                tuple(json.loads(row["available_offers"])) if row["available_offers"] else None
            ),
        )
        for row in rows
    ]
