import math

import numpy as np
import pytest

from contextual_bandit_decision_ops.offer_features import (
    CONTEXT_FIELDS,
    FEATURE_DIMENSION,
    FEATURE_NAMES,
    OfferContext,
    context_from_mapping,
    encode_context,
    encode_contexts,
    new_user_context,
)
from contextual_bandit_decision_ops.offers import Offer, OfferCatalog, default_offer_catalog


def returning_buyer(**overrides: object) -> OfferContext:
    values: dict[str, object] = {
        "user_id": "user-1",
        "device": "desktop",
        "region": "north",
        "engagement_score": 0.7,
        "sessions_7d": 4,
        "sessions_30d": 12,
        "days_since_last_visit": 1.5,
        "tenure_days": 400.0,
        "purchases_90d": 3,
        "revenue_90d": 240.0,
        "avg_order_value": 80.0,
        "days_since_last_purchase": 12.0,
        "offer_impressions_30d": 5,
        "offer_conversions_30d": 1,
    }
    values.update(overrides)
    return OfferContext(**values)  # type: ignore[arg-type]


def test_default_catalog_has_unique_offers_with_revenue() -> None:
    catalog = default_offer_catalog()
    assert len(catalog) == 5
    assert len(set(catalog.offer_ids)) == 5
    assert np.all(catalog.revenue_vector() > 0)
    assert catalog.index("bundle_deal") == 2
    assert catalog.revenue("premium_trial") == 70.0


def test_catalog_rejects_duplicates_and_unknown_offers() -> None:
    with pytest.raises(ValueError, match="unique"):
        OfferCatalog((Offer("a", "A", 1.0), Offer("a", "A again", 2.0)))
    with pytest.raises(ValueError, match="at least one"):
        OfferCatalog(())
    catalog = default_offer_catalog()
    with pytest.raises(KeyError):
        catalog.get("missing")
    with pytest.raises(ValueError, match="unknown offers"):
        catalog.validate_subset(["pct10_discount", "missing"])
    with pytest.raises(ValueError, match="positive"):
        Offer("bad", "Bad", 0.0)


def test_catalog_can_grow_for_new_offer_launches() -> None:
    catalog = default_offer_catalog().with_offer(Offer("gift_card", "Gift card", 45.0))
    assert catalog.offer_ids[-1] == "gift_card"
    assert len(catalog) == 6


def test_encoder_produces_named_bounded_features() -> None:
    vector = encode_context(returning_buyer())
    assert vector.shape == (FEATURE_DIMENSION,)
    assert len(FEATURE_NAMES) == FEATURE_DIMENSION
    assert np.all(np.isfinite(vector))
    assert np.all(vector >= 0.0)
    assert np.all(vector <= 1.5)
    named = dict(zip(FEATURE_NAMES, vector, strict=True))
    assert named["bias"] == 1.0
    assert named["device_desktop"] == 1.0
    assert named["device_mobile"] == 0.0
    assert named["region_north"] == 1.0
    assert named["never_purchased"] == 0.0
    assert named["purchase_recency"] == pytest.approx(math.exp(-12.0 / 30.0))
    assert named["offer_response_rate_30d"] == pytest.approx(1.0 / 6.0)


def test_one_hot_groups_sum_to_one() -> None:
    for device in ("mobile", "desktop", "tablet"):
        for region in ("north", "south", "east", "west"):
            named = dict(
                zip(
                    FEATURE_NAMES,
                    encode_context(returning_buyer(device=device, region=region)),
                    strict=True,
                )
            )
            assert sum(v for k, v in named.items() if k.startswith("device_")) == 1.0
            assert sum(v for k, v in named.items() if k.startswith("region_")) == 1.0


def test_new_user_is_encoded_as_cold_start() -> None:
    named = dict(
        zip(FEATURE_NAMES, encode_context(new_user_context("new", "mobile", "west")), strict=True)
    )
    assert named["never_purchased"] == 1.0
    assert named["purchase_recency"] == 0.0
    assert named["tenure_log"] == 0.0
    assert named["sessions_30d_log"] == 0.0


def test_more_activity_means_larger_features() -> None:
    light = dict(zip(FEATURE_NAMES, encode_context(returning_buyer(sessions_30d=4, sessions_7d=1))))
    heavy = dict(zip(FEATURE_NAMES, encode_context(returning_buyer(sessions_30d=40, sessions_7d=9))))
    assert heavy["sessions_30d_log"] > light["sessions_30d_log"]
    assert heavy["sessions_7d_log"] > light["sessions_7d_log"]
    stale = dict(zip(FEATURE_NAMES, encode_context(returning_buyer(days_since_last_visit=30.0))))
    fresh = dict(zip(FEATURE_NAMES, encode_context(returning_buyer(days_since_last_visit=0.5))))
    assert fresh["visit_recency"] > stale["visit_recency"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("device", "smart_tv", "device"),
        ("region", "mars", "region"),
        ("engagement_score", 1.2, "engagement_score"),
        ("sessions_30d", -1, "sessions_30d"),
        ("sessions_7d", 20, "sessions_7d cannot exceed"),
        ("offer_conversions_30d", 9, "offer_conversions_30d cannot exceed"),
        ("days_since_last_purchase", -2.0, "days_since_last_purchase"),
        ("avg_order_value", float("nan"), "avg_order_value"),
    ],
)
def test_context_validation(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        returning_buyer(**{field: value})


def test_context_round_trips_through_mapping() -> None:
    original = returning_buyer(days_since_last_purchase=None)
    payload = {
        key: str(value) if isinstance(value, int) else value
        for key, value in original.to_dict().items()
    }
    restored = context_from_mapping(payload)
    assert restored == original
    assert set(original.to_dict()) == set(CONTEXT_FIELDS)
    with pytest.raises(ValueError, match="missing context fields"):
        context_from_mapping({"user_id": "x"})


def test_encode_contexts_stacks_rows() -> None:
    matrix = encode_contexts([returning_buyer(), new_user_context("n", "tablet", "south")])
    assert matrix.shape == (2, FEATURE_DIMENSION)
    assert encode_contexts([]).shape == (0, FEATURE_DIMENSION)
