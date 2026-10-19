from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .bayesian_policies import sigmoid
from .offer_features import FEATURE_DIMENSION, FEATURE_NAMES, OfferContext, encode_context
from .offers import Offer, OfferCatalog, default_offer_catalog

# Ground truth for the simulator only. Policies never see these numbers; they
# have to learn them from conversions. Each offer appeals to a different kind
# of user, so no single offer wins everywhere.
DEFAULT_TRUE_COEFFICIENTS: dict[str, dict[str, float]] = {
    "pct10_discount": {
        "bias": -2.9,
        "engagement_score": -0.7,
        "never_purchased": 1.0,
        "purchase_recency": -0.4,
        "avg_order_value_log": -0.6,
        "region_south": 0.3,
    },
    "free_shipping": {
        "bias": -3.3,
        "device_mobile": 0.8,
        "visit_recency": 0.5,
        "avg_order_value_log": 0.4,
        "region_west": 0.35,
        "region_east": -0.25,
    },
    "bundle_deal": {
        "bias": -4.1,
        "purchases_90d_log": 1.5,
        "purchase_recency": 0.6,
        "never_purchased": -0.8,
        "device_desktop": 0.3,
        "engagement_score": 0.3,
    },
    "premium_trial": {
        "bias": -5.4,
        "engagement_score": 1.8,
        "avg_order_value_log": 1.2,
        "sessions_30d_log": 0.6,
        "device_desktop": 0.5,
        "device_mobile": -0.4,
        "tenure_log": 0.3,
    },
    "loyalty_points": {
        "bias": -2.8,
        "tenure_log": 0.8,
        "offer_response_rate_30d": 1.5,
        "purchases_90d_log": 0.4,
        "region_north": 0.3,
        "engagement_score": 0.2,
    },
}


def coefficient_vector(coefficients: Mapping[str, float]) -> np.ndarray:
    unknown = set(coefficients) - set(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"unknown features: {', '.join(sorted(unknown))}")
    return np.array([coefficients.get(name, 0.0) for name in FEATURE_NAMES], dtype=float)


@dataclass(frozen=True)
class ConversionModel:
    """The simulator's true conversion probability for every offer and user."""

    catalog: OfferCatalog
    weights: Mapping[str, np.ndarray] = field(repr=False)

    def __post_init__(self) -> None:
        missing = [offer_id for offer_id in self.catalog.offer_ids if offer_id not in self.weights]
        if missing:
            raise ValueError(f"missing true weights for: {', '.join(missing)}")
        for offer_id, vector in self.weights.items():
            if np.shape(vector) != (FEATURE_DIMENSION,):
                raise ValueError(f"weights for {offer_id} have the wrong shape")
        object.__setattr__(
            self, "weights", {k: np.asarray(v, dtype=float) for k, v in self.weights.items()}
        )

    def conversion_probabilities(
        self,
        context: OfferContext,
        offers: Sequence[str] | None = None,
    ) -> np.ndarray:
        offer_ids = self.catalog.offer_ids if offers is None else tuple(offers)
        features = encode_context(context)
        logits = np.array([self.weights[offer_id] @ features for offer_id in offer_ids])
        return sigmoid(logits)

    def conversion_probability(self, context: OfferContext, offer_id: str) -> float:
        return float(self.conversion_probabilities(context, [offer_id])[0])

    def expected_revenue(
        self,
        context: OfferContext,
        offers: Sequence[str] | None = None,
    ) -> np.ndarray:
        offer_ids = self.catalog.offer_ids if offers is None else tuple(offers)
        return self.conversion_probabilities(context, offer_ids) * self.catalog.revenue_vector(
            offer_ids
        )

    def best_offer(self, context: OfferContext, offers: Sequence[str] | None = None) -> str:
        offer_ids = self.catalog.offer_ids if offers is None else tuple(offers)
        return offer_ids[int(np.argmax(self.expected_revenue(context, offer_ids)))]

    def with_logit_shift(self, shifts: Mapping[str, float]) -> ConversionModel:
        """Move offers' conversion rates up or down, used for drift scenarios."""
        weights = {offer_id: vector.copy() for offer_id, vector in self.weights.items()}
        for offer_id, shift in shifts.items():
            weights[offer_id][0] += shift
        return ConversionModel(self.catalog, weights)

    def with_offer(self, offer: Offer, coefficients: Mapping[str, float]) -> ConversionModel:
        """Add a new offer to the world, used for cold-start launch scenarios."""
        return ConversionModel(
            self.catalog.with_offer(offer),
            {**self.weights, offer.offer_id: coefficient_vector(coefficients)},
        )


def default_conversion_model(catalog: OfferCatalog | None = None) -> ConversionModel:
    offer_catalog = catalog or default_offer_catalog()
    return ConversionModel(
        offer_catalog,
        {
            offer_id: coefficient_vector(DEFAULT_TRUE_COEFFICIENTS[offer_id])
            for offer_id in offer_catalog.offer_ids
        },
    )
