from __future__ import annotations

from datetime import datetime
from math import isfinite

from .config import SimulationConfig
from .schemas import REGIONS, BanditEvent


def validate_bandit_event(event: BanditEvent, config: SimulationConfig) -> None:
    if not isinstance(event, BanditEvent):
        raise TypeError("event must be a BanditEvent")
    if not event.event_id:
        raise ValueError("event_id is required")
    if not event.user_id:
        raise ValueError("user_id is required")
    if not event.timestamp:
        raise ValueError("timestamp is required")
    try:
        timestamp = datetime.fromisoformat(event.timestamp)
    except ValueError as error:
        raise ValueError("timestamp must be ISO 8601") from error
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    if event.seed != config.seed:
        raise ValueError("seed mismatch")
    if not 18.0 <= event.context_age <= 70.0:
        raise ValueError("context_age must be between 18 and 70")
    if not 0.0 <= event.context_engagement <= 1.0:
        raise ValueError("context_engagement must be between 0 and 1")
    if event.context_region not in REGIONS:
        raise ValueError(f"context_region must be one of {REGIONS}")
    if event.action not in range(config.n_actions):
        raise ValueError("action is out of range")
    if event.reward not in {0, 1}:
        raise ValueError("reward must be 0 or 1")
    if not isfinite(event.reward_probability) or not 0.0 <= event.reward_probability <= 1.0:
        raise ValueError("reward_probability must be between 0 and 1")
    if not isfinite(event.propensity) or not 0.0 < event.propensity <= 1.0:
        raise ValueError("propensity must be greater than 0 and at most 1")


def validate_event_log(events: list[BanditEvent], config: SimulationConfig) -> None:
    if len(events) != config.n_events:
        raise ValueError(f"expected {config.n_events} events, received {len(events)}")

    event_ids: set[str] = set()
    for event in events:
        validate_bandit_event(event, config)
        if event.event_id in event_ids:
            raise ValueError(f"duplicate event_id: {event.event_id}")
        event_ids.add(event.event_id)
