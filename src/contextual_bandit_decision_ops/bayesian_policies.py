from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from .offer_features import OfferContext
from .offers import OfferCatalog

Objective = Literal["revenue", "conversion"]


class OfferPolicy(Protocol):
    @property
    def name(self) -> str: ...

    def choose(
        self,
        context: OfferContext,
        available_offers: Sequence[str],
        rng: np.random.Generator,
    ) -> str: ...


@runtime_checkable
class LearningOfferPolicy(OfferPolicy, Protocol):
    def update(self, context: OfferContext, offer_id: str, converted: bool) -> None: ...


@dataclass(frozen=True)
class ConversionPosterior:
    offer_id: str
    mean: float
    lower: float
    upper: float
    expected_revenue: float
    observations: int


def _validate_objective(objective: str) -> None:
    if objective not in ("revenue", "conversion"):
        raise ValueError("objective must be 'revenue' or 'conversion'")


def _argmax_offer(offers: Sequence[str], scores: np.ndarray) -> str:
    return offers[int(np.argmax(scores))]


def _monte_carlo_choice_probabilities(
    offers: Sequence[str],
    sampled_scores: np.ndarray,
) -> dict[str, float]:
    winners = np.argmax(sampled_scores, axis=1)
    counts = np.bincount(winners, minlength=len(offers))
    return {offer: float(count / len(winners)) for offer, count in zip(offers, counts, strict=True)}


@dataclass
class BetaBernoulliThompsonSampling:
    """Beta posterior on each offer's conversion rate, kept separately per segment.

    Segments are built from context attributes (device by default), so mobile and
    desktop users get their own conversion estimates. ``discount`` below 1 shrinks
    old evidence back toward the prior before every update, which lets the
    posterior follow conversion rates that move over time.
    """

    catalog: OfferCatalog
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    segment_by: tuple[str, ...] = ("device",)
    discount: float = 1.0
    objective: Objective = "revenue"
    alphas: dict[tuple[str, str], float] = field(init=False, default_factory=dict)
    betas: dict[tuple[str, str], float] = field(init=False, default_factory=dict)
    observations: dict[tuple[str, str], int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.prior_alpha <= 0.0 or self.prior_beta <= 0.0:
            raise ValueError("prior_alpha and prior_beta must be positive")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        _validate_objective(self.objective)
        self.segment_by = tuple(self.segment_by)

    @property
    def name(self) -> str:
        return "beta_thompson_sampling"

    def segment(self, context: OfferContext) -> str:
        if not self.segment_by:
            return "all"
        return "|".join(str(getattr(context, attribute)) for attribute in self.segment_by)

    def parameters(self, context: OfferContext, offer_id: str) -> tuple[float, float]:
        key = (self.segment(context), offer_id)
        return self.alphas.get(key, self.prior_alpha), self.betas.get(key, self.prior_beta)

    def _parameter_vectors(
        self,
        context: OfferContext,
        offers: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        pairs = [self.parameters(context, offer_id) for offer_id in offers]
        return np.array([a for a, _ in pairs]), np.array([b for _, b in pairs])

    def _weights(self, offers: Sequence[str]) -> np.ndarray:
        if self.objective == "conversion":
            return np.ones(len(offers))
        return self.catalog.revenue_vector(offers)

    def choose(
        self,
        context: OfferContext,
        available_offers: Sequence[str],
        rng: np.random.Generator,
    ) -> str:
        offers = self.catalog.validate_subset(available_offers)
        alphas, betas = self._parameter_vectors(context, offers)
        sampled_rates = rng.beta(alphas, betas)
        return _argmax_offer(offers, sampled_rates * self._weights(offers))

    def action_probabilities(
        self,
        context: OfferContext,
        available_offers: Sequence[str],
        rng: np.random.Generator,
        n_samples: int = 2_000,
    ) -> dict[str, float]:
        offers = self.catalog.validate_subset(available_offers)
        alphas, betas = self._parameter_vectors(context, offers)
        sampled_rates = rng.beta(alphas, betas, size=(n_samples, len(offers)))
        return _monte_carlo_choice_probabilities(offers, sampled_rates * self._weights(offers))

    def update(self, context: OfferContext, offer_id: str, converted: bool) -> None:
        if offer_id not in self.catalog:
            raise ValueError(f"unknown offer: {offer_id}")
        segment = self.segment(context)
        if self.discount < 1.0:
            for key in [key for key in self.alphas if key[0] == segment]:
                self.alphas[key] = self.prior_alpha + self.discount * (
                    self.alphas[key] - self.prior_alpha
                )
                self.betas[key] = self.prior_beta + self.discount * (
                    self.betas[key] - self.prior_beta
                )
        key = (segment, offer_id)
        alpha, beta = self.alphas.get(key, self.prior_alpha), self.betas.get(key, self.prior_beta)
        self.alphas[key] = alpha + float(converted)
        self.betas[key] = beta + float(not converted)
        self.observations[key] = self.observations.get(key, 0) + 1

    def posterior(self, context: OfferContext, offer_id: str) -> ConversionPosterior:
        alpha, beta = self.parameters(context, offer_id)
        draws = np.random.default_rng(0).beta(alpha, beta, size=4_000)
        mean = alpha / (alpha + beta)
        return ConversionPosterior(
            offer_id=offer_id,
            mean=float(mean),
            lower=float(np.quantile(draws, 0.05)),
            upper=float(np.quantile(draws, 0.95)),
            expected_revenue=float(mean * self.catalog.revenue(offer_id)),
            observations=self.observations.get((self.segment(context), offer_id), 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "beta_bernoulli",
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
            "segment_by": list(self.segment_by),
            "discount": self.discount,
            "objective": self.objective,
            "posteriors": [
                {
                    "segment": segment,
                    "offer_id": offer_id,
                    "alpha": self.alphas[(segment, offer_id)],
                    "beta": self.betas[(segment, offer_id)],
                    "observations": self.observations.get((segment, offer_id), 0),
                }
                for segment, offer_id in sorted(self.alphas)
            ],
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        catalog: OfferCatalog,
    ) -> BetaBernoulliThompsonSampling:
        if payload.get("model_type") != "beta_bernoulli":
            raise ValueError("payload is not a beta_bernoulli model")
        policy = cls(
            catalog=catalog,
            prior_alpha=float(payload["prior_alpha"]),
            prior_beta=float(payload["prior_beta"]),
            segment_by=tuple(payload["segment_by"]),
            discount=float(payload["discount"]),
            objective=payload["objective"],
        )
        for row in payload["posteriors"]:
            key = (row["segment"], row["offer_id"])
            policy.alphas[key] = float(row["alpha"])
            policy.betas[key] = float(row["beta"])
            policy.observations[key] = int(row["observations"])
        return policy
