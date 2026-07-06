from __future__ import annotations

from dataclasses import dataclass

REGIONS = ("north", "south", "east", "west")

EVENT_COLUMNS = (
    "event_id",
    "user_id",
    "context_age",
    "context_engagement",
    "context_region",
    "action",
    "reward",
    "reward_probability",
    "propensity",
    "timestamp",
    "seed",
)


@dataclass(frozen=True)
class UserContext:
    age: float
    engagement: float
    region: str


@dataclass(frozen=True)
class BanditEvent:
    event_id: str
    user_id: str
    context_age: float
    context_engagement: float
    context_region: str
    action: int
    reward: int
    reward_probability: float
    propensity: float
    timestamp: str
    seed: int
