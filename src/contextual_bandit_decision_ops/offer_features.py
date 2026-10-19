from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

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


def _log_scaled(value: float, reference: float) -> float:
    return math.log1p(value) / math.log1p(reference)


NUMERIC_FEATURE_NAMES = (
    "engagement_score",
    "sessions_7d_log",
    "sessions_30d_log",
    "visit_recency",
    "tenure_log",
    "purchases_90d_log",
    "avg_order_value_log",
    "purchase_recency",
    "never_purchased",
    "offer_response_rate_30d",
)
FEATURE_NAMES = (
    "bias",
    *NUMERIC_FEATURE_NAMES,
    *(f"device_{device}" for device in DEVICES),
    *(f"region_{region}" for region in GEOGRAPHIES),
)
FEATURE_DIMENSION = len(FEATURE_NAMES)


def encode_context(context: OfferContext) -> np.ndarray:
    """Turn a context into a fixed-length vector, with most values roughly in [0, 1].

    Counts and money are log scaled so heavy users don't dominate, recency uses
    exponential decay so "never" and "long ago" both land near zero, and device
    and region are one-hot.
    """
    purchase_recency = (
        math.exp(-context.days_since_last_purchase / 30.0)
        if context.days_since_last_purchase is not None
        else 0.0
    )
    response_rate = context.offer_conversions_30d / (context.offer_impressions_30d + 1.0)
    vector = [
        1.0,
        context.engagement_score,
        _log_scaled(context.sessions_7d, 14.0),
        _log_scaled(context.sessions_30d, 60.0),
        math.exp(-context.days_since_last_visit / 7.0),
        _log_scaled(context.tenure_days, 730.0),
        _log_scaled(context.purchases_90d, 12.0),
        _log_scaled(context.avg_order_value, 400.0),
        purchase_recency,
        float(not context.has_purchased),
        response_rate,
        *(float(context.device == device) for device in DEVICES),
        *(float(context.region == region) for region in GEOGRAPHIES),
    ]
    return np.asarray(vector, dtype=float)


def encode_contexts(contexts: list[OfferContext]) -> np.ndarray:
    if not contexts:
        return np.zeros((0, FEATURE_DIMENSION), dtype=float)
    return np.vstack([encode_context(context) for context in contexts])
