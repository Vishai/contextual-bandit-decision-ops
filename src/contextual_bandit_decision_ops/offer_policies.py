from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .bayesian_policies import BayesianLogisticThompsonSampling, OfferPolicy, sigmoid
from .offer_features import OfferContext
from .offer_simulation import ConversionModel
from .offers import OfferCatalog


def _one_hot(offers: Sequence[str], chosen: str) -> dict[str, float]:
    return {offer_id: float(offer_id == chosen) for offer_id in offers}


@dataclass(frozen=True)
class RandomOfferPolicy:
    catalog: OfferCatalog

    @property
    def name(self) -> str:
        return "random"

    def choose(
        self,
        context: OfferContext,
        available_offers: Sequence[str],
        rng: np.random.Generator,
    ) -> str:
        offers = self.catalog.validate_subset(available_offers)
        return offers[int(rng.integers(0, len(offers)))]

    def action_probabilities(
        self,
        context: OfferContext,
        available_offers: Sequence[str],
        rng: np.random.Generator | None = None,
    ) -> dict[str, float]:
        offers = self.catalog.validate_subset(available_offers)
        return {offer_id: 1.0 / len(offers) for offer_id in offers}


@dataclass(frozen=True)
class RuleBasedOfferPolicy:
    """The hand-written targeting rules used before the bandit.

    First-time buyers get the discount, mobile gets free shipping, frequent
    buyers get the bundle, engaged big spenders get the premium trial, and
    everyone else gets loyalty points. If the preferred offer is unavailable
    we fall back to the next rule that matches.
    """

    catalog: OfferCatalog
    frequent_buyer_purchases: int = 4
    big_spender_order_value: float = 100.0
    engaged_score: float = 0.6

    @property
    def name(self) -> str:
        return "rule_based"

    def preferences(self, context: OfferContext) -> list[str]:
        ranked: list[str] = []
        if not context.has_purchased:
            ranked.append("pct10_discount")
        if context.device == "mobile":
            ranked.append("free_shipping")
        if context.purchases_90d >= self.frequent_buyer_purchases:
            ranked.append("bundle_deal")
        if (
            context.avg_order_value >= self.big_spender_order_value
            and context.engagement_score >= self.engaged_score
        ):
            ranked.append("premium_trial")
        ranked.append("loyalty_points")
        return ranked

    def choose(
        self,
        context: OfferContext,
        available_offers: Sequence[str],
        rng: np.random.Generator,
    ) -> str:
        offers = self.catalog.validate_subset(available_offers)
        for offer_id in self.preferences(context):
            if offer_id in offers:
                return offer_id
        return offers[0]

    def action_probabilities(
        self,
        context: OfferContext,
        available_offers: Sequence[str],
        rng: np.random.Generator | None = None,
    ) -> dict[str, float]:
        offers = self.catalog.validate_subset(available_offers)
        return _one_hot(offers, self.choose(context, offers, np.random.default_rng(0)))


class GreedyOfferPolicy(BayesianLogisticThompsonSampling):
    """Same Bayesian logistic model as Thompson Sampling, but always exploits.

    It picks the offer with the best posterior mean and never explores on
    purpose, which is the classic way a learner gets stuck on an early guess.
    """

    @property
    def name(self) -> str:
        return "greedy"

    def _scores(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        weights: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return sigmoid(means) * weights

    def action_probabilities(
        self,
        context: OfferContext,
        available_offers: Sequence[str],
        rng: np.random.Generator,
        n_samples: int = 2_000,
    ) -> dict[str, float]:
        offers = self.catalog.validate_subset(available_offers)
        return _one_hot(offers, self.choose(context, offers, rng))


@dataclass
class UCBOfferPolicy(BayesianLogisticThompsonSampling):
    """Same Bayesian logistic model, scored by an upper confidence bound.

    Each offer is scored as if its conversion logit were ``ucb_width`` standard
    deviations above the posterior mean. Exploration is deterministic: the most
    uncertain promising offer always gets the bonus.
    """

    ucb_width: float = 1.0

    def __post_init__(self) -> None:
        if self.ucb_width < 0.0:
            raise ValueError("ucb_width must be non-negative")
        super().__post_init__()

    @property
    def name(self) -> str:
        return "ucb"

    def _scores(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        weights: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return sigmoid(means + self.ucb_width * stds) * weights

    def action_probabilities(
        self,
        context: OfferContext,
        available_offers: Sequence[str],
        rng: np.random.Generator,
        n_samples: int = 2_000,
    ) -> dict[str, float]:
        offers = self.catalog.validate_subset(available_offers)
        return _one_hot(offers, self.choose(context, offers, rng))


@dataclass(frozen=True)
class OracleOfferPolicy:
    """Knows the simulator's true conversion rates. Only for measuring regret."""

    model: ConversionModel

    @property
    def name(self) -> str:
        return "oracle"

    def choose(
        self,
        context: OfferContext,
        available_offers: Sequence[str],
        rng: np.random.Generator,
    ) -> str:
        return self.model.best_offer(context, available_offers)


@dataclass(frozen=True)
class EpsilonExplorationPolicy:
    """Wraps another policy and swaps in a uniform random offer ``epsilon`` of the time.

    The wrapped policy has to report its own choice probabilities, so the
    combined propensity of every logged offer is known exactly. That's what
    makes the logs usable for off-policy evaluation later.
    """

    base: OfferPolicy
    catalog: OfferCatalog
    epsilon: float = 0.2

    def __post_init__(self) -> None:
        if not 0.0 < self.epsilon <= 1.0:
            raise ValueError("epsilon must be in (0, 1]")
        if not hasattr(self.base, "action_probabilities"):
            raise ValueError("the base policy must expose action_probabilities")

    @property
    def name(self) -> str:
        return f"{self.base.name}_eps{self.epsilon:.2f}"

    def choose(
        self,
        context: OfferContext,
        available_offers: Sequence[str],
        rng: np.random.Generator,
    ) -> str:
        offers = self.catalog.validate_subset(available_offers)
        if rng.random() < self.epsilon:
            return offers[int(rng.integers(0, len(offers)))]
        return self.base.choose(context, offers, rng)

    def action_probabilities(
        self,
        context: OfferContext,
        available_offers: Sequence[str],
        rng: np.random.Generator | None = None,
    ) -> dict[str, float]:
        offers = self.catalog.validate_subset(available_offers)
        base = self.base.action_probabilities(  # type: ignore[attr-defined]
            context, offers, rng or np.random.default_rng(0)
        )
        uniform = self.epsilon / len(offers)
        return {
            offer_id: uniform + (1.0 - self.epsilon) * base.get(offer_id, 0.0)
            for offer_id in offers
        }
