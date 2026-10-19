from __future__ import annotations

import os
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from .bayesian_policies import BayesianLogisticThompsonSampling
from .feature_store import FeatureStore
from .model_store import load_policy_artifact
from .offer_api_schemas import (
    CatalogOffer,
    ConversionPosteriorPayload,
    FeatureSource,
    LatencySummary,
    ModelInfoResponse,
    OfferDecisionRequest,
    OfferDecisionResponse,
    OfferFeedbackRequest,
    OfferFeedbackResponse,
    OfferMetricsResponse,
    OfferPosteriorResponse,
    OfferRequestBase,
)
from .offer_features import OfferContext, new_user_context
from .offers import OfferCatalog, default_offer_catalog
from .service_logging import LocalEventLogger

Clock = Callable[[], datetime]
UNTRAINED_VERSION = "untrained"


def utc_now() -> datetime:
    return datetime.now(UTC)


class UnknownDecisionEventError(KeyError):
    pass


class DuplicateFeedbackError(ValueError):
    pass


class MissingContextError(ValueError):
    pass


@dataclass(frozen=True)
class OfferServiceConfig:
    seed: int = 42
    environment: str = "local"
    decision_log_path: Path | str = Path("logs/offer_decisions.jsonl")
    feedback_log_path: Path | str = Path("logs/offer_feedback.jsonl")
    feature_store_path: Path | str | None = None
    model_artifact_path: Path | str | None = None
    exploration_floor: float = 0.05
    propensity_samples: int = 1_000
    latency_budget_ms: float = 100.0
    latency_window: int = 10_000
    max_pending_decisions: int = 200_000
    online_updates: bool = True
    model_poll_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not 0.0 <= self.exploration_floor < 1.0:
            raise ValueError("exploration_floor must be in [0, 1)")
        if self.propensity_samples <= 0:
            raise ValueError("propensity_samples must be positive")
        if self.latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms must be positive")
        if self.model_poll_seconds < 0:
            raise ValueError("model_poll_seconds must be non-negative")
        if self.latency_window <= 0 or self.max_pending_decisions <= 0:
            raise ValueError("latency_window and max_pending_decisions must be positive")
        object.__setattr__(self, "decision_log_path", Path(self.decision_log_path))
        object.__setattr__(self, "feedback_log_path", Path(self.feedback_log_path))
        if self.feature_store_path is not None:
            object.__setattr__(self, "feature_store_path", Path(self.feature_store_path))
        if self.model_artifact_path is not None:
            object.__setattr__(self, "model_artifact_path", Path(self.model_artifact_path))


@dataclass
class _PendingDecision:
    context: OfferContext
    offer_id: str


def summarize_latencies(values: list[float] | deque[float], budget_ms: float) -> LatencySummary:
    if not values:
        return LatencySummary(
            count=0,
            p50_ms=None,
            p95_ms=None,
            p99_ms=None,
            max_ms=None,
            budget_ms=budget_ms,
            within_budget_rate=None,
        )
    array = np.asarray(values, dtype=float)
    p50, p95, p99 = np.percentile(array, [50, 95, 99])
    return LatencySummary(
        count=len(array),
        p50_ms=float(p50),
        p95_ms=float(p95),
        p99_ms=float(p99),
        max_ms=float(array.max()),
        budget_ms=budget_ms,
        within_budget_rate=float(np.mean(array <= budget_ms)),
    )


class OfferDecisionService:
    """Serves offer decisions and learns from every piece of feedback as it arrives.

    Each decision is Thompson Sampling with a small uniform exploration floor
    mixed in. That keeps every offer's propensity above zero, so the logs stay
    usable for off-policy evaluation and retraining. Feedback updates the
    posterior straight away, so the next request already sees what was learned.
    """

    def __init__(
        self,
        config: OfferServiceConfig | None = None,
        policy: BayesianLogisticThompsonSampling | None = None,
        catalog: OfferCatalog | None = None,
        clock: Clock = utc_now,
    ) -> None:
        self.config = config or OfferServiceConfig()
        self.catalog = catalog or (policy.catalog if policy else default_offer_catalog())
        self.clock = clock
        self.lock = Lock()
        self.store_lock = Lock()
        self.logger = LocalEventLogger(
            self.config.decision_log_path,  # type: ignore[arg-type]
            self.config.feedback_log_path,  # type: ignore[arg-type]
        )
        self.feature_store = (
            FeatureStore(self.config.feature_store_path)
            if self.config.feature_store_path is not None
            else None
        )
        # Event ids have to stay unique across restarts, or ingesting the logs later
        # would mix up decisions from different runs.
        self.run_id = uuid.uuid4().hex[:12]
        seeds = np.random.SeedSequence(self.config.seed).spawn(2)
        self.choice_rng = np.random.default_rng(seeds[0])
        self.propensity_rng = np.random.default_rng(seeds[1])

        self.policy = policy or BayesianLogisticThompsonSampling(self.catalog)
        self.model_version = UNTRAINED_VERSION
        self.model_created_at: str | None = None
        self.model_source: str | None = None
        self.model_metadata: dict[str, Any] = {}
        self._artifact_mtime: int | None = None
        self._next_poll = time.monotonic() + self.config.model_poll_seconds
        artifact_path = self.config.model_artifact_path
        if policy is None and artifact_path is not None and Path(artifact_path).exists():
            self.reload_model()

        self.pending: OrderedDict[str, _PendingDecision] = OrderedDict()
        self.completed: OrderedDict[str, None] = OrderedDict()
        self.decision_latencies: deque[float] = deque(maxlen=self.config.latency_window)
        self.request_latencies: deque[float] = deque(maxlen=self.config.latency_window)
        self.allocation = {offer_id: 0 for offer_id in self.catalog.offer_ids}
        self.decision_count = 0
        self.explored_count = 0
        self.feedback_count = 0
        self.conversions = 0
        self.revenue = 0.0
        self.posterior_updates = 0

    # -- model management -------------------------------------------------

    def reload_model(self, path: Path | str | None = None) -> str:
        source = Path(path) if path is not None else self.config.model_artifact_path
        if source is None:
            raise FileNotFoundError("no model artifact path is configured")
        modified = os.stat(source).st_mtime_ns
        artifact = load_policy_artifact(source, self.catalog)
        with self.lock:
            self._artifact_mtime = modified
            self.policy = artifact.policy
            self.model_version = artifact.model_version
            self.model_created_at = artifact.created_at
            self.model_source = str(source)
            self.model_metadata = artifact.metadata
        return self.model_version

    def maybe_reload(self) -> bool:
        """Pick up a newly promoted model without a restart.

        With several service tasks behind a load balancer, a single reload call
        only reaches one of them, so each task also checks the artifact's
        modification time every ``model_poll_seconds``. A broken or half-copied
        artifact is ignored and the current model keeps serving.
        """
        path = self.config.model_artifact_path
        if self.config.model_poll_seconds <= 0 or path is None:
            return False
        now = time.monotonic()
        if now < self._next_poll:
            return False
        self._next_poll = now + self.config.model_poll_seconds
        try:
            if os.stat(path).st_mtime_ns == self._artifact_mtime:
                return False
            self.reload_model()
        except (OSError, ValueError, KeyError):
            return False
        return True

    def model_info(self) -> ModelInfoResponse:
        with self.lock:
            return ModelInfoResponse(
                policy_name=self.policy.name,
                model_version=self.model_version,
                created_at=self.model_created_at,
                source=self.model_source,
                posterior_updates=self.posterior_updates,
                observations_by_offer={
                    offer_id: posterior.updates
                    for offer_id, posterior in self.policy.posteriors.items()
                },
                metadata=self.model_metadata,
            )

    def catalog_payload(self) -> list[CatalogOffer]:
        return [
            CatalogOffer(
                offer_id=offer.offer_id,
                name=offer.name,
                revenue_per_conversion=offer.revenue_per_conversion,
            )
            for offer in self.catalog.offers
        ]

    # -- request handling -------------------------------------------------

    def resolve_context(self, request: OfferRequestBase) -> tuple[OfferContext, FeatureSource]:
        if request.context is not None:
            return (
                OfferContext(user_id=request.user_id, **request.context.model_dump()),
                "request",
            )
        if self.feature_store is not None:
            with self.store_lock:
                stored = self.feature_store.latest_features(request.user_id, self.clock())
            if stored is not None:
                return stored, "feature_store"
        if request.device is None or request.region is None:
            raise MissingContextError(
                "no features found for this user; send a context, or device and region "
                "for a cold-start decision"
            )
        return new_user_context(request.user_id, request.device, request.region), "cold_start"

    def _offers(self, request: OfferRequestBase) -> tuple[str, ...]:
        return self.catalog.validate_subset(request.available_offers or self.catalog.offer_ids)

    def decide(self, request: OfferDecisionRequest) -> OfferDecisionResponse:
        self.maybe_reload()
        started = time.perf_counter()
        context, source = self.resolve_context(request)
        offers = self._offers(request)
        floor = self.config.exploration_floor
        with self.lock:
            explored = bool(self.choice_rng.random() < floor)
            if explored:
                offer_id = offers[int(self.choice_rng.integers(0, len(offers)))]
            else:
                offer_id = self.policy.choose(context, offers, self.choice_rng)
            sampled = self.policy.action_probabilities(
                context, offers, self.propensity_rng, n_samples=self.config.propensity_samples
            )
            propensity = floor / len(offers) + (1.0 - floor) * sampled[offer_id]
            propensity = min(1.0, max(propensity, 1.0 / self.config.propensity_samples))
            posterior = self.policy.posterior(context, offer_id)
            timestamp = self.clock().astimezone(UTC)
            event_id = f"offer-{self.run_id}-{self.decision_count:010d}"
            latency_ms = (time.perf_counter() - started) * 1_000.0

            self.pending[event_id] = _PendingDecision(context, offer_id)
            while len(self.pending) > self.config.max_pending_decisions:
                self.pending.popitem(last=False)
            self.decision_count += 1
            self.explored_count += explored
            self.allocation[offer_id] += 1
            self.decision_latencies.append(latency_ms)
            model_version = self.model_version
            policy_name = self.policy.name
            self.logger.log_decision(
                {
                    "event_id": event_id,
                    "user_id": request.user_id,
                    "context": context.to_dict(),
                    "feature_source": source,
                    "available_offers": list(offers),
                    "offer_id": offer_id,
                    "propensity": propensity,
                    "explored": explored,
                    "expected_conversion": posterior.mean,
                    "policy_name": policy_name,
                    "model_version": model_version,
                    "decision_latency_ms": latency_ms,
                    "timestamp": timestamp.isoformat(),
                    "metadata": request.metadata or {},
                }
            )
        return OfferDecisionResponse(
            event_id=event_id,
            user_id=request.user_id,
            offer_id=offer_id,
            offer_name=self.catalog.get(offer_id).name,
            propensity=propensity,
            explored=explored,
            expected_conversion=posterior.mean,
            expected_revenue=posterior.expected_revenue,
            feature_source=source,
            policy_name=policy_name,
            model_version=model_version,
            decision_latency_ms=latency_ms,
            timestamp=timestamp,
        )

    def feedback(self, request: OfferFeedbackRequest) -> OfferFeedbackResponse:
        with self.lock:
            if request.event_id in self.completed:
                raise DuplicateFeedbackError(request.event_id)
            decision = self.pending.pop(request.event_id, None)
            if decision is None:
                raise UnknownDecisionEventError(request.event_id)
            revenue = request.revenue
            if revenue is None:
                revenue = self.catalog.revenue(decision.offer_id) if request.converted else 0.0
            updated = False
            if self.config.online_updates:
                self.policy.update(decision.context, decision.offer_id, request.converted)
                self.posterior_updates += 1
                updated = True
            self.completed[request.event_id] = None
            while len(self.completed) > self.config.max_pending_decisions:
                self.completed.popitem(last=False)
            self.feedback_count += 1
            self.conversions += request.converted
            self.revenue += revenue
            timestamp = self.clock().astimezone(UTC)
            self.logger.log_feedback(
                {
                    "event_id": request.event_id,
                    "user_id": decision.context.user_id,
                    "offer_id": decision.offer_id,
                    "converted": request.converted,
                    "revenue": revenue,
                    "timestamp": timestamp.isoformat(),
                    "metadata": request.metadata or {},
                }
            )
        return OfferFeedbackResponse(
            event_id=request.event_id,
            status="accepted",
            posterior_updated=updated,
            timestamp=timestamp,
        )

    def posterior(self, request: OfferRequestBase) -> OfferPosteriorResponse:
        context, source = self.resolve_context(request)
        offers = self._offers(request)
        with self.lock:
            rows = self.policy.posterior_summary(context, offers)
            version = self.model_version
        return OfferPosteriorResponse(
            user_id=request.user_id,
            feature_source=source,
            model_version=version,
            offers=[ConversionPosteriorPayload(**row.__dict__) for row in rows],
        )

    def record_request_latency(self, latency_ms: float) -> None:
        with self.lock:
            self.request_latencies.append(latency_ms)

    def metrics(self) -> OfferMetricsResponse:
        with self.lock:
            decisions = self.decision_count
            feedback = self.feedback_count
            return OfferMetricsResponse(
                decisions=decisions,
                feedback_records=feedback,
                conversions=self.conversions,
                conversion_rate=self.conversions / feedback if feedback else None,
                revenue=self.revenue,
                revenue_per_decision=self.revenue / feedback if feedback else None,
                exploration_rate=self.explored_count / decisions if decisions else None,
                allocation=dict(self.allocation),
                pending_feedback=len(self.pending),
                posterior_updates=self.posterior_updates,
                model_version=self.model_version,
                decision_latency=summarize_latencies(
                    self.decision_latencies, self.config.latency_budget_ms
                ),
                request_latency=summarize_latencies(
                    self.request_latencies, self.config.latency_budget_ms
                ),
            )

    def close(self) -> None:
        if self.feature_store is not None:
            self.feature_store.close()
