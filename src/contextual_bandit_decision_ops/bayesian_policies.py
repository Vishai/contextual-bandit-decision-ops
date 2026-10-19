from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from .offer_features import FEATURE_NAMES, OfferContext, encode_context, encode_contexts
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


def sigmoid(values: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def _logit(probability: float) -> float:
    return float(np.log(probability / (1.0 - probability)))


@dataclass
class _OfferPosterior:
    mean: np.ndarray
    precision: np.ndarray
    covariance: np.ndarray
    updates: int = 0


@dataclass
class BayesianLogisticThompsonSampling:
    """Contextual Thompson Sampling with a Bayesian logistic model per offer.

    Each offer keeps a Gaussian posterior over the weights of a logistic
    conversion model fit on the encoded user features. Every piece of feedback
    triggers a Laplace style online update: the precision grows by
    p(1 - p) x x^T and the mean takes one Newton step toward the outcome.

    To pick an offer we draw a conversion logit for each offer from its
    posterior predictive, turn it into a probability, multiply by the offer's
    revenue and take the best. Offers the model is unsure about get sampled
    high some of the time, and that is where the exploration comes from.
    """

    catalog: OfferCatalog
    feature_names: tuple[str, ...] = FEATURE_NAMES
    prior_variance: float = 1.0
    prior_conversion_rate: float = 0.05
    exploration_scale: float = 1.0
    forgetting: float = 1.0
    objective: Objective = "revenue"
    posteriors: dict[str, _OfferPosterior] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.prior_variance <= 0.0:
            raise ValueError("prior_variance must be positive")
        if not 0.0 < self.prior_conversion_rate < 1.0:
            raise ValueError("prior_conversion_rate must be in (0, 1)")
        if self.exploration_scale < 0.0:
            raise ValueError("exploration_scale must be non-negative")
        if not 0.0 < self.forgetting <= 1.0:
            raise ValueError("forgetting must be in (0, 1]")
        _validate_objective(self.objective)
        self.feature_names = tuple(self.feature_names)
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("feature_names must match the offer feature encoder")
        for offer_id in self.catalog.offer_ids:
            self._posterior(offer_id)

    @property
    def name(self) -> str:
        return "thompson_sampling"

    @property
    def dimension(self) -> int:
        return len(self.feature_names)

    @property
    def prior_mean(self) -> np.ndarray:
        mean = np.zeros(self.dimension)
        mean[0] = _logit(self.prior_conversion_rate)
        return mean

    @property
    def prior_precision(self) -> np.ndarray:
        return np.eye(self.dimension) / self.prior_variance

    def _posterior(self, offer_id: str) -> _OfferPosterior:
        posterior = self.posteriors.get(offer_id)
        if posterior is None:
            if offer_id not in self.catalog:
                raise ValueError(f"unknown offer: {offer_id}")
            posterior = _OfferPosterior(
                mean=self.prior_mean,
                precision=self.prior_precision,
                covariance=np.eye(self.dimension) * self.prior_variance,
            )
            self.posteriors[offer_id] = posterior
        return posterior

    def logit_distribution(
        self,
        context: OfferContext,
        offers: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Posterior predictive mean and standard deviation of each offer's logit."""
        features = encode_context(context)
        means = np.empty(len(offers))
        stds = np.empty(len(offers))
        for index, offer_id in enumerate(offers):
            posterior = self._posterior(offer_id)
            means[index] = posterior.mean @ features
            stds[index] = np.sqrt(max(features @ posterior.covariance @ features, 0.0))
        return means, stds

    def _weights(self, offers: Sequence[str]) -> np.ndarray:
        if self.objective == "conversion":
            return np.ones(len(offers))
        return self.catalog.revenue_vector(offers)

    def _scores(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        weights: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        sampled_logits = means + self.exploration_scale * stds * rng.standard_normal(len(means))
        return sigmoid(sampled_logits) * weights

    def choose(
        self,
        context: OfferContext,
        available_offers: Sequence[str],
        rng: np.random.Generator,
    ) -> str:
        offers = self.catalog.validate_subset(available_offers)
        means, stds = self.logit_distribution(context, offers)
        return _argmax_offer(offers, self._scores(means, stds, self._weights(offers), rng))

    def action_probabilities(
        self,
        context: OfferContext,
        available_offers: Sequence[str],
        rng: np.random.Generator,
        n_samples: int = 2_000,
    ) -> dict[str, float]:
        offers = self.catalog.validate_subset(available_offers)
        means, stds = self.logit_distribution(context, offers)
        noise = rng.standard_normal((n_samples, len(offers)))
        sampled = sigmoid(means + self.exploration_scale * stds * noise) * self._weights(offers)
        return _monte_carlo_choice_probabilities(offers, sampled)

    def _apply_forgetting(self) -> None:
        prior_precision = self.prior_precision
        for posterior in self.posteriors.values():
            posterior.precision = (
                self.forgetting * posterior.precision + (1.0 - self.forgetting) * prior_precision
            )
            posterior.covariance = np.linalg.inv(posterior.precision)

    def update(self, context: OfferContext, offer_id: str, converted: bool) -> None:
        posterior = self._posterior(offer_id)
        if self.forgetting < 1.0:
            self._apply_forgetting()
        features = encode_context(context)
        probability = float(sigmoid(posterior.mean @ features))
        weight = max(probability * (1.0 - probability), 1e-6)

        covariance_features = posterior.covariance @ features
        denominator = 1.0 + weight * (features @ covariance_features)
        covariance = posterior.covariance - (
            weight * np.outer(covariance_features, covariance_features) / denominator
        )
        posterior.covariance = 0.5 * (covariance + covariance.T)
        posterior.precision = posterior.precision + weight * np.outer(features, features)
        posterior.mean = posterior.mean + posterior.covariance @ features * (
            float(converted) - probability
        )
        posterior.updates += 1

    def fit(
        self,
        contexts: Sequence[OfferContext],
        offer_ids: Sequence[str],
        converted: Sequence[bool],
        max_iterations: int = 50,
        tolerance: float = 1e-8,
    ) -> None:
        """Refit every offer from scratch: MAP weights by Newton's method, then the
        Laplace covariance at the optimum. Retraining uses this batch path."""
        if not len(contexts) == len(offer_ids) == len(converted):
            raise ValueError("contexts, offer_ids and converted must have equal length")
        features = encode_contexts(list(contexts))
        outcomes = np.asarray(converted, dtype=float)
        offers = np.asarray(offer_ids, dtype=object)
        prior_mean = self.prior_mean
        prior_precision = self.prior_precision
        fitted: dict[str, _OfferPosterior] = {}
        for offer_id in self.catalog.offer_ids:
            rows = offers == offer_id
            x, y = features[rows], outcomes[rows]
            weights = prior_mean.copy()
            for _ in range(max_iterations):
                probabilities = sigmoid(x @ weights)
                gradient = x.T @ (y - probabilities) - prior_precision @ (weights - prior_mean)
                curvature = probabilities * (1.0 - probabilities)
                hessian = prior_precision + (x * curvature[:, None]).T @ x
                step = np.linalg.solve(hessian, gradient)
                weights = weights + step
                if float(np.max(np.abs(step))) < tolerance:
                    break
            probabilities = sigmoid(x @ weights)
            curvature = probabilities * (1.0 - probabilities)
            hessian = prior_precision + (x * curvature[:, None]).T @ x
            fitted[offer_id] = _OfferPosterior(
                mean=weights,
                precision=hessian,
                covariance=np.linalg.inv(hessian),
                updates=int(rows.sum()),
            )
        self.posteriors = fitted

    def posterior(self, context: OfferContext, offer_id: str) -> ConversionPosterior:
        mean_logit, std_logit = self.logit_distribution(context, [offer_id])
        mu, sd = float(mean_logit[0]), float(std_logit[0])
        expected_rate = float(sigmoid(mu / np.sqrt(1.0 + np.pi * sd**2 / 8.0)))
        return ConversionPosterior(
            offer_id=offer_id,
            mean=expected_rate,
            lower=float(sigmoid(mu - 1.645 * sd)),
            upper=float(sigmoid(mu + 1.645 * sd)),
            expected_revenue=expected_rate * self.catalog.revenue(offer_id),
            observations=self._posterior(offer_id).updates,
        )

    def posterior_summary(
        self,
        context: OfferContext,
        available_offers: Sequence[str] | None = None,
    ) -> list[ConversionPosterior]:
        offers = self.catalog.validate_subset(available_offers or self.catalog.offer_ids)
        return [self.posterior(context, offer_id) for offer_id in offers]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "bayesian_logistic",
            "feature_names": list(self.feature_names),
            "prior_variance": self.prior_variance,
            "prior_conversion_rate": self.prior_conversion_rate,
            "exploration_scale": self.exploration_scale,
            "forgetting": self.forgetting,
            "objective": self.objective,
            "offers": {
                offer_id: {
                    "mean": posterior.mean.tolist(),
                    "precision": posterior.precision.tolist(),
                    "updates": posterior.updates,
                }
                for offer_id, posterior in sorted(self.posteriors.items())
            },
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        catalog: OfferCatalog,
    ) -> BayesianLogisticThompsonSampling:
        if payload.get("model_type") != "bayesian_logistic":
            raise ValueError("payload is not a bayesian_logistic model")
        if tuple(payload["feature_names"]) != FEATURE_NAMES:
            raise ValueError("saved model was trained on a different feature set")
        policy = cls(
            catalog=catalog,
            prior_variance=float(payload["prior_variance"]),
            prior_conversion_rate=float(payload["prior_conversion_rate"]),
            exploration_scale=float(payload["exploration_scale"]),
            forgetting=float(payload["forgetting"]),
            objective=payload["objective"],
        )
        for offer_id, state in payload["offers"].items():
            if offer_id not in catalog:
                continue
            precision = np.asarray(state["precision"], dtype=float)
            policy.posteriors[offer_id] = _OfferPosterior(
                mean=np.asarray(state["mean"], dtype=float),
                precision=precision,
                covariance=np.linalg.inv(precision),
                updates=int(state["updates"]),
            )
        return policy
