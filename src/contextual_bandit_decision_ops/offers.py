from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Offer:
    offer_id: str
    name: str
    revenue_per_conversion: float

    def __post_init__(self) -> None:
        if not self.offer_id:
            raise ValueError("offer_id must not be empty")
        if self.revenue_per_conversion <= 0.0:
            raise ValueError("revenue_per_conversion must be positive")


@dataclass(frozen=True)
class OfferCatalog:
    offers: tuple[Offer, ...]
    _index: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        offers = tuple(self.offers)
        if not offers:
            raise ValueError("an offer catalog needs at least one offer")
        ids = [offer.offer_id for offer in offers]
        if len(set(ids)) != len(ids):
            raise ValueError("offer ids must be unique")
        object.__setattr__(self, "offers", offers)
        object.__setattr__(self, "_index", {offer_id: i for i, offer_id in enumerate(ids)})

    def __len__(self) -> int:
        return len(self.offers)

    def __contains__(self, offer_id: object) -> bool:
        return offer_id in self._index

    @property
    def offer_ids(self) -> tuple[str, ...]:
        return tuple(offer.offer_id for offer in self.offers)

    def get(self, offer_id: str) -> Offer:
        try:
            return self.offers[self._index[offer_id]]
        except KeyError as error:
            raise KeyError(f"unknown offer: {offer_id}") from error

    def index(self, offer_id: str) -> int:
        try:
            return self._index[offer_id]
        except KeyError as error:
            raise KeyError(f"unknown offer: {offer_id}") from error

    def revenue(self, offer_id: str) -> float:
        return self.get(offer_id).revenue_per_conversion

    def revenue_vector(self, offer_ids: Sequence[str] | None = None) -> np.ndarray:
        ids = self.offer_ids if offer_ids is None else offer_ids
        return np.array([self.revenue(offer_id) for offer_id in ids], dtype=float)

    def validate_subset(self, offer_ids: Iterable[str]) -> tuple[str, ...]:
        subset = tuple(offer_ids)
        if not subset:
            raise ValueError("at least one offer must be available")
        unknown = [offer_id for offer_id in subset if offer_id not in self]
        if unknown:
            raise ValueError(f"unknown offers: {', '.join(unknown)}")
        if len(set(subset)) != len(subset):
            raise ValueError("available offers must be unique")
        return subset

    def with_offer(self, offer: Offer) -> OfferCatalog:
        return OfferCatalog((*self.offers, offer))


DEFAULT_OFFERS = (
    Offer("pct10_discount", "10% off the next order", 32.0),
    Offer("free_shipping", "Free shipping on the next order", 38.0),
    Offer("bundle_deal", "Buy two, save 20%", 55.0),
    Offer("premium_trial", "30-day premium membership trial", 70.0),
    Offer("loyalty_points", "Double loyalty points this week", 26.0),
)


def default_offer_catalog() -> OfferCatalog:
    return OfferCatalog(DEFAULT_OFFERS)
