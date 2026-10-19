from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

from .schemas import REGIONS

DEVICES = ("mobile", "desktop", "tablet")
GEOGRAPHIES = REGIONS


@dataclass(frozen=True)
class OfferContext:
    """Everything we know about a user at decision time.

    Grouped as user history (visits, tenure, past offers), device, geography,
    engagement, and purchase behaviour. ``days_since_last_purchase`` is None for
    users who have never bought anything.
    """

    user_id: str
    device: str
    region: str
    engagement_score: float
    sessions_7d: int
    sessions_30d: int
    days_since_last_visit: float
    tenure_days: float
    purchases_90d: int
    revenue_90d: float
    avg_order_value: float
    days_since_last_purchase: float | None
    offer_impressions_30d: int
    offer_conversions_30d: int

    def __post_init__(self) -> None:
        if not self.user_id:
            raise ValueError("user_id must not be empty")
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}")
        if self.region not in GEOGRAPHIES:
            raise ValueError(f"region must be one of {GEOGRAPHIES}")
        if not 0.0 <= self.engagement_score <= 1.0:
            raise ValueError("engagement_score must be between 0 and 1")
        for name in (
            "sessions_7d",
            "sessions_30d",
            "days_since_last_visit",
            "tenure_days",
            "purchases_90d",
            "revenue_90d",
            "avg_order_value",
            "offer_impressions_30d",
            "offer_conversions_30d",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.sessions_7d > self.sessions_30d:
            raise ValueError("sessions_7d cannot exceed sessions_30d")
        if self.offer_conversions_30d > self.offer_impressions_30d:
            raise ValueError("offer_conversions_30d cannot exceed offer_impressions_30d")
        if self.days_since_last_purchase is not None and (
            not math.isfinite(self.days_since_last_purchase) or self.days_since_last_purchase < 0
        ):
            raise ValueError("days_since_last_purchase must be non-negative or None")

    @property
    def has_purchased(self) -> bool:
        return self.days_since_last_purchase is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CONTEXT_FIELDS = tuple(field.name for field in fields(OfferContext))
_INTEGER_FIELDS = frozenset(
    {
        "sessions_7d",
        "sessions_30d",
        "purchases_90d",
        "offer_impressions_30d",
        "offer_conversions_30d",
    }
)


def context_from_mapping(values: Mapping[str, Any]) -> OfferContext:
    missing = [name for name in CONTEXT_FIELDS if name not in values]
    if missing:
        raise ValueError(f"missing context fields: {', '.join(missing)}")
    parsed: dict[str, Any] = {}
    for name in CONTEXT_FIELDS:
        value = values[name]
        if name in {"user_id", "device", "region"}:
            parsed[name] = str(value)
        elif name == "days_since_last_purchase":
            parsed[name] = None if value is None else float(value)
        elif name in _INTEGER_FIELDS:
            parsed[name] = int(value)
        else:
            parsed[name] = float(value)
    return OfferContext(**parsed)


def new_user_context(user_id: str, device: str, region: str) -> OfferContext:
    """Context for someone we have never seen before: the cold-start case."""
    return OfferContext(
        user_id=user_id,
        device=device,
        region=region,
        engagement_score=0.0,
        sessions_7d=0,
        sessions_30d=0,
        days_since_last_visit=0.0,
        tenure_days=0.0,
        purchases_90d=0,
        revenue_90d=0.0,
        avg_order_value=0.0,
        days_since_last_purchase=None,
        offer_impressions_30d=0,
        offer_conversions_30d=0,
    )
