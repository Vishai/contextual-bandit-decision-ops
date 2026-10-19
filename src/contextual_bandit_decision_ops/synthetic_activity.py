from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from .feature_store import FeatureStore, PurchaseRecord, SessionRecord, UserRecord
from .offer_features import DEVICES, GEOGRAPHIES, OfferContext, new_user_context

DEVICE_SHARES = (0.55, 0.35, 0.10)


@dataclass(frozen=True)
class ActivityConfig:
    n_users: int = 2_000
    seed: int = 11
    start: datetime = datetime(2025, 3, 1, tzinfo=UTC)
    history_days: int = 90
    simulation_days: int = 28
    new_user_share: float = 0.12

    def __post_init__(self) -> None:
        if self.n_users <= 0:
            raise ValueError("n_users must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.start.tzinfo is None:
            raise ValueError("start must be timezone aware")
        if self.history_days <= 0 or self.simulation_days <= 0:
            raise ValueError("history_days and simulation_days must be positive")
        if not 0.0 <= self.new_user_share < 1.0:
            raise ValueError("new_user_share must be in [0, 1)")

    @property
    def end(self) -> datetime:
        return self.start + timedelta(days=self.simulation_days)

    def snapshot_times(self) -> list[datetime]:
        return [self.start + timedelta(days=day) for day in range(self.simulation_days)]


@dataclass(frozen=True)
class DecisionPoint:
    """A visit during the simulation window where we get to show an offer."""

    timestamp: datetime
    session_id: str
    context: OfferContext
    is_new_user: bool


@dataclass(frozen=True)
class SyntheticPopulation:
    config: ActivityConfig
    users: tuple[UserRecord, ...]
    sessions: tuple[SessionRecord, ...]
    purchases: tuple[PurchaseRecord, ...]

    def decision_sessions(self) -> list[SessionRecord]:
        window = [
            session
            for session in self.sessions
            if self.config.start <= session.started_at < self.config.end
        ]
        return sorted(window, key=lambda session: (session.started_at, session.session_id))


def _seconds(value: datetime) -> datetime:
    return value.replace(microsecond=0)


def generate_population(config: ActivityConfig) -> SyntheticPopulation:
    """Organic browsing and buying, independent of any offer we might show."""
    rng = np.random.default_rng(config.seed)
    history_start = config.start - timedelta(days=config.history_days)
    users: list[UserRecord] = []
    sessions: list[SessionRecord] = []
    purchases: list[PurchaseRecord] = []

    for user_index in range(config.n_users):
        user_id = f"user-{user_index:06d}"
        device = DEVICES[int(rng.choice(len(DEVICES), p=DEVICE_SHARES))]
        region = GEOGRAPHIES[int(rng.integers(0, len(GEOGRAPHIES)))]
        if rng.random() < config.new_user_share:
            signup_at = config.start + timedelta(days=float(rng.uniform(0.0, config.simulation_days)))
        else:
            signup_at = config.start - timedelta(days=float(rng.uniform(1.0, 720.0)))
        signup_at = _seconds(signup_at)
        users.append(UserRecord(user_id, signup_at, device, region))

        visits_per_day = float(np.clip(rng.lognormal(math.log(0.25), 0.8), 0.02, 3.0))
        purchase_probability = float(rng.beta(1.2, 10.0))
        basket_mean = float(rng.lognormal(math.log(70.0), 0.5))
        minutes_mean = float(rng.gamma(2.0, 6.0))

        active_from = max(signup_at, history_start)
        active_days = (config.end - active_from).total_seconds() / 86_400.0
        n_sessions = int(rng.poisson(visits_per_day * active_days))
        offsets = np.sort(rng.uniform(0.0, active_days, size=n_sessions))

        for session_index, offset in enumerate(offsets):
            started_at = _seconds(active_from + timedelta(days=float(offset)))
            duration = float(rng.gamma(2.0, minutes_mean / 2.0))
            page_views = 1 + int(rng.poisson(duration / 2.0))
            session_id = f"{user_id}-s{session_index:05d}"
            sessions.append(
                SessionRecord(session_id, user_id, started_at, round(duration, 2), page_views)
            )
            if rng.random() < purchase_probability:
                amount = float(rng.lognormal(math.log(basket_mean), 0.35))
                purchased_at = _seconds(
                    started_at + timedelta(minutes=float(rng.uniform(0.0, duration)))
                )
                purchases.append(
                    PurchaseRecord(f"{session_id}-p", user_id, purchased_at, round(amount, 2))
                )

    return SyntheticPopulation(config, tuple(users), tuple(sessions), tuple(purchases))


def build_feature_store(
    config: ActivityConfig,
    path: Path | str = ":memory:",
) -> tuple[FeatureStore, SyntheticPopulation]:
    """Load synthetic activity into SQLite and materialize one snapshot per day."""
    population = generate_population(config)
    store = FeatureStore(path)
    store.add_users(population.users)
    store.add_sessions(population.sessions)
    store.add_purchases(population.purchases)
    for snapshot_time in config.snapshot_times():
        store.materialize_features(snapshot_time)
    return store, population


def decision_points(
    store: FeatureStore,
    population: SyntheticPopulation,
) -> list[DecisionPoint]:
    """Pair every visit in the simulation window with that morning's feature snapshot.

    Users who signed up after the snapshot was taken have no row yet, so they get
    a cold-start context, the same thing the live service does.
    """
    config = population.config
    device_by_user = {user.user_id: (user.device, user.region) for user in population.users}
    snapshots = {time: store.snapshot(time) for time in config.snapshot_times()}
    points: list[DecisionPoint] = []
    for session in population.decision_sessions():
        day = (session.started_at - config.start).days
        snapshot_time = config.start + timedelta(days=day)
        context = snapshots[snapshot_time].get(session.user_id)
        is_new_user = context is None
        if context is None:
            device, region = device_by_user[session.user_id]
            context = new_user_context(session.user_id, device, region)
        points.append(DecisionPoint(session.started_at, session.session_id, context, is_new_user))
    return points
