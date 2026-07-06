from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class StrictApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ContextPayload(StrictApiModel):
    age: float = Field(ge=18.0, le=70.0)
    engagement: float = Field(ge=0.0, le=1.0)
    region: Literal["north", "south", "east", "west"]


class DecisionRequest(StrictApiModel):
    user_id: str = Field(min_length=1)
    context: ContextPayload
    metadata: dict[str, JsonValue] | None = None


class DecisionResponse(StrictApiModel):
    event_id: str
    action: int
    propensity: float = Field(gt=0.0, le=1.0)
    policy_name: str
    timestamp: datetime


class FeedbackRequest(StrictApiModel):
    event_id: str = Field(min_length=1)
    reward: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, JsonValue] | None = None


class FeedbackResponse(StrictApiModel):
    event_id: str
    status: Literal["accepted"]
    timestamp: datetime


class HealthResponse(StrictApiModel):
    status: Literal["ok"]
    environment: Literal["local-staging"]
    staging_only: bool


class PolicyMetadataResponse(StrictApiModel):
    policy_name: str
    available_actions: list[int]
    safety_constraints: dict[str, JsonValue] | None
    staging_only: bool


class MetricsResponse(StrictApiModel):
    decisions: int
    feedback_records: int
    action_counts: dict[int, int]
    average_observed_reward: float | None
