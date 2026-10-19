from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request, Response, status

from .bayesian_policies import BayesianLogisticThompsonSampling
from .metadata_cli import project_version
from .offer_api_schemas import (
    CatalogOffer,
    ModelInfoResponse,
    OfferDecisionRequest,
    OfferDecisionResponse,
    OfferFeedbackRequest,
    OfferFeedbackResponse,
    OfferHealthResponse,
    OfferMetricsResponse,
    OfferPosteriorResponse,
    OfferRequestBase,
)
from .offer_service import (
    Clock,
    DuplicateFeedbackError,
    MissingContextError,
    OfferDecisionService,
    OfferServiceConfig,
    UnknownDecisionEventError,
    utc_now,
)

LATENCY_HEADER = "X-Process-Time-Ms"


def create_offer_app(
    config: OfferServiceConfig | None = None,
    policy: BayesianLogisticThompsonSampling | None = None,
    clock: Clock = utc_now,
) -> FastAPI:
    service = OfferDecisionService(config=config, policy=policy, clock=clock)
    app = FastAPI(
        title="Offer Recommendation Service",
        version=project_version(),
        description=(
            "Real-time offer decisions from contextual Thompson Sampling, with feedback "
            "updating the Bayesian posterior online."
        ),
    )
    app.state.offer_service = service

    @app.middleware("http")
    async def track_latency(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        response.headers[LATENCY_HEADER] = f"{elapsed_ms:.3f}"
        if request.url.path == "/decide" and response.status_code == status.HTTP_200_OK:
            service.record_request_latency(elapsed_ms)
        return response

    def _unprocessable(error: Exception) -> HTTPException:
        return HTTPException(status_code=422, detail=str(error))

    @app.get("/health", response_model=OfferHealthResponse)
    async def health() -> OfferHealthResponse:
        return OfferHealthResponse(
            status="ok",
            environment=service.config.environment,
            policy_name=service.policy.name,
            model_version=service.model_version,
            feature_store=service.feature_store is not None,
        )

    @app.get("/offers", response_model=list[CatalogOffer])
    async def offers() -> list[CatalogOffer]:
        return service.catalog_payload()

    @app.post("/decide", response_model=OfferDecisionResponse)
    async def decide(request: OfferDecisionRequest) -> OfferDecisionResponse:
        try:
            return service.decide(request)
        except (MissingContextError, ValueError) as error:
            raise _unprocessable(error) from error

    @app.post("/feedback", response_model=OfferFeedbackResponse)
    async def feedback(request: OfferFeedbackRequest) -> OfferFeedbackResponse:
        try:
            return service.feedback(request)
        except UnknownDecisionEventError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"unknown or expired decision event: {request.event_id}",
            ) from error
        except DuplicateFeedbackError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"feedback already recorded for event: {request.event_id}",
            ) from error

    @app.post("/posterior", response_model=OfferPosteriorResponse)
    async def posterior(request: OfferRequestBase) -> OfferPosteriorResponse:
        try:
            return service.posterior(request)
        except (MissingContextError, ValueError) as error:
            raise _unprocessable(error) from error

    @app.get("/metrics", response_model=OfferMetricsResponse)
    async def metrics() -> OfferMetricsResponse:
        return service.metrics()

    @app.get("/model", response_model=ModelInfoResponse)
    async def model() -> ModelInfoResponse:
        return service.model_info()

    @app.post("/model/reload", response_model=ModelInfoResponse)
    async def reload_model() -> ModelInfoResponse:
        try:
            service.reload_model()
        except FileNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        except ValueError as error:
            raise _unprocessable(error) from error
        return service.model_info()

    return app
