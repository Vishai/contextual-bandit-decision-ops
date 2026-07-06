from __future__ import annotations

from fastapi import FastAPI, HTTPException, status

from .api_schemas import (
    DecisionRequest,
    DecisionResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    MetricsResponse,
    PolicyMetadataResponse,
)
from .policies import ProbabilisticBanditPolicy
from .service import (
    Clock,
    DecisionService,
    DuplicateFeedbackError,
    ServiceConfig,
    UnknownDecisionEventError,
    utc_now,
)


def create_app(
    config: ServiceConfig | None = None,
    policy: ProbabilisticBanditPolicy | None = None,
    clock: Clock = utc_now,
) -> FastAPI:
    service = DecisionService(config=config, policy=policy, clock=clock)
    app = FastAPI(
        title="Contextual Bandit Decision Service",
        version="0.1.0",
        description="Local/staging-only contextual bandit decision and feedback API.",
    )
    app.state.decision_service = service

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            environment="local-staging",
            staging_only=service.config.staging_only,
        )

    @app.get("/policy", response_model=PolicyMetadataResponse)
    async def policy_metadata() -> PolicyMetadataResponse:
        return service.policy_metadata()

    @app.post("/decide", response_model=DecisionResponse)
    async def decide(request: DecisionRequest) -> DecisionResponse:
        return service.decide(request)

    @app.post("/feedback", response_model=FeedbackResponse)
    async def feedback(request: FeedbackRequest) -> FeedbackResponse:
        try:
            return service.feedback(request)
        except UnknownDecisionEventError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"unknown decision event: {error}",
            ) from error
        except DuplicateFeedbackError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"feedback already recorded for event: {error}",
            ) from error

    @app.get("/metrics", response_model=MetricsResponse)
    async def metrics() -> MetricsResponse:
        return service.metrics()

    return app
