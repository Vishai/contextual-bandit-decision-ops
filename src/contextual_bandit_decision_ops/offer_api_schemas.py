from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

Device = Literal["mobile", "desktop", "tablet"]
Region = Literal["north", "south", "east", "west"]
FeatureSource = Literal["request", "feature_store", "cold_start"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OfferContextPayload(StrictModel):
    device: Device
    region: Region
    engagement_score: float = Field(ge=0.0, le=1.0)
    sessions_7d: int = Field(ge=0)
    sessions_30d: int = Field(ge=0)
    days_since_last_visit: float = Field(ge=0.0)
    tenure_days: float = Field(ge=0.0)
    purchases_90d: int = Field(ge=0)
    revenue_90d: float = Field(ge=0.0)
    avg_order_value: float = Field(ge=0.0)
    days_since_last_purchase: float | None = Field(default=None, ge=0.0)
    offer_impressions_30d: int = Field(ge=0)
    offer_conversions_30d: int = Field(ge=0)

    @model_validator(mode="after")
    def _consistent_counts(self) -> OfferContextPayload:
        if self.sessions_7d > self.sessions_30d:
            raise ValueError("sessions_7d cannot exceed sessions_30d")
        if self.offer_conversions_30d > self.offer_impressions_30d:
            raise ValueError("offer_conversions_30d cannot exceed offer_impressions_30d")
        return self


class OfferRequestBase(StrictModel):
    """Who the decision is for.

    Send the full ``context`` if the caller already has features. Otherwise the
    service looks the user up in the feature store, and if they aren't there yet
    it builds a cold-start context from ``device`` and ``region``.
    """

    user_id: str = Field(min_length=1, max_length=128)
    context: OfferContextPayload | None = None
    device: Device | None = None
    region: Region | None = None
    available_offers: list[str] | None = Field(default=None, min_length=1)


class OfferDecisionRequest(OfferRequestBase):
    metadata: dict[str, JsonValue] | None = None


class OfferDecisionResponse(StrictModel):
    event_id: str
    user_id: str
    offer_id: str
    offer_name: str
    propensity: float = Field(gt=0.0, le=1.0)
    explored: bool
    expected_conversion: float
    expected_revenue: float
    feature_source: FeatureSource
    policy_name: str
    model_version: str
    decision_latency_ms: float
    timestamp: datetime


class OfferFeedbackRequest(StrictModel):
    event_id: str = Field(min_length=1)
    converted: bool
    revenue: float | None = Field(default=None, ge=0.0)
    metadata: dict[str, JsonValue] | None = None


class OfferFeedbackResponse(StrictModel):
    event_id: str
    status: Literal["accepted", "logged"]
    posterior_updated: bool
    timestamp: datetime


class ConversionPosteriorPayload(StrictModel):
    offer_id: str
    mean: float
    lower: float
    upper: float
    expected_revenue: float
    observations: int


class OfferPosteriorResponse(StrictModel):
    user_id: str
    feature_source: FeatureSource
    model_version: str
    offers: list[ConversionPosteriorPayload]


class CatalogOffer(StrictModel):
    offer_id: str
    name: str
    revenue_per_conversion: float


class LatencySummary(StrictModel):
    count: int
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    max_ms: float | None
    budget_ms: float
    within_budget_rate: float | None


class OfferMetricsResponse(StrictModel):
    decisions: int
    feedback_records: int
    conversions: int
    conversion_rate: float | None
    revenue: float
    revenue_per_decision: float | None
    exploration_rate: float | None
    allocation: dict[str, int]
    pending_feedback: int
    posterior_updates: int
    model_version: str
    decision_latency: LatencySummary
    request_latency: LatencySummary


class OfferHealthResponse(StrictModel):
    status: Literal["ok"]
    environment: str
    policy_name: str
    model_version: str
    feature_store: bool


class ModelInfoResponse(StrictModel):
    policy_name: str
    model_version: str
    created_at: str | None
    source: str | None
    posterior_updates: int
    observations_by_offer: dict[str, int]
    metadata: dict[str, JsonValue]
