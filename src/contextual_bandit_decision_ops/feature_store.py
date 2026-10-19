from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self

from .offer_features import OfferContext

SQL_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    signup_at TEXT NOT NULL,
    device TEXT NOT NULL,
    region TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    started_at TEXT NOT NULL,
    duration_minutes REAL NOT NULL CHECK (duration_minutes >= 0),
    page_views INTEGER NOT NULL CHECK (page_views >= 0)
);
CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_sessions_user_time ON sessions(user_id, started_at);

CREATE TABLE IF NOT EXISTS purchases (
    purchase_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    purchased_at TEXT NOT NULL,
    amount REAL NOT NULL CHECK (amount >= 0)
);
CREATE INDEX IF NOT EXISTS idx_purchases_user_time ON purchases(user_id, purchased_at);

CREATE TABLE IF NOT EXISTS offer_events (
    event_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    offer_id TEXT NOT NULL,
    policy_name TEXT NOT NULL,
    propensity REAL NOT NULL CHECK (propensity > 0 AND propensity <= 1),
    context_json TEXT NOT NULL,
    converted INTEGER CHECK (converted IN (0, 1)),
    revenue REAL,
    feedback_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_offer_events_decided_at ON offer_events(decided_at);
CREATE INDEX IF NOT EXISTS idx_offer_events_user_time ON offer_events(user_id, decided_at);

CREATE TABLE IF NOT EXISTS user_features (
    user_id TEXT NOT NULL,
    as_of TEXT NOT NULL,
    device TEXT NOT NULL,
    region TEXT NOT NULL,
    engagement_score REAL NOT NULL,
    sessions_7d INTEGER NOT NULL,
    sessions_30d INTEGER NOT NULL,
    days_since_last_visit REAL NOT NULL,
    tenure_days REAL NOT NULL,
    purchases_90d INTEGER NOT NULL,
    revenue_90d REAL NOT NULL,
    avg_order_value REAL NOT NULL,
    days_since_last_purchase REAL,
    offer_impressions_30d INTEGER NOT NULL,
    offer_conversions_30d INTEGER NOT NULL,
    PRIMARY KEY (user_id, as_of)
);
CREATE INDEX IF NOT EXISTS idx_user_features_as_of ON user_features(as_of);
"""


def to_sql_timestamp(value: datetime) -> str:
    """Store every timestamp as naive UTC text so string order matches time order."""
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone aware")
    return value.astimezone(UTC).strftime(SQL_TIMESTAMP_FORMAT)


def from_sql_timestamp(value: str) -> datetime:
    return datetime.strptime(value, SQL_TIMESTAMP_FORMAT).replace(tzinfo=UTC)


@dataclass(frozen=True)
class UserRecord:
    user_id: str
    signup_at: datetime
    device: str
    region: str


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    user_id: str
    started_at: datetime
    duration_minutes: float
    page_views: int


@dataclass(frozen=True)
class PurchaseRecord:
    purchase_id: str
    user_id: str
    purchased_at: datetime
    amount: float


class FeatureStore:
    """SQLite store for raw user activity, logged offer events and feature snapshots."""

    def __init__(self, path: Path | str = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self.connection.execute("PRAGMA journal_mode = WAL")
            self.connection.execute("PRAGMA synchronous = NORMAL")
        self.create_schema()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def create_schema(self) -> None:
        self.connection.executescript(SCHEMA_SQL)
        self.connection.commit()

    def add_users(self, users: Iterable[UserRecord]) -> int:
        rows = [
            (user.user_id, to_sql_timestamp(user.signup_at), user.device, user.region)
            for user in users
        ]
        self.connection.executemany(
            "INSERT INTO users (user_id, signup_at, device, region) VALUES (?, ?, ?, ?)",
            rows,
        )
        self.connection.commit()
        return len(rows)

    def add_sessions(self, sessions: Iterable[SessionRecord]) -> int:
        rows = [
            (
                session.session_id,
                session.user_id,
                to_sql_timestamp(session.started_at),
                session.duration_minutes,
                session.page_views,
            )
            for session in sessions
        ]
        self.connection.executemany(
            "INSERT INTO sessions (session_id, user_id, started_at, duration_minutes, page_views) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self.connection.commit()
        return len(rows)

    def add_purchases(self, purchases: Iterable[PurchaseRecord]) -> int:
        rows = [
            (
                purchase.purchase_id,
                purchase.user_id,
                to_sql_timestamp(purchase.purchased_at),
                purchase.amount,
            )
            for purchase in purchases
        ]
        self.connection.executemany(
            "INSERT INTO purchases (purchase_id, user_id, purchased_at, amount) VALUES (?, ?, ?, ?)",
            rows,
        )
        self.connection.commit()
        return len(rows)

    def record_offer_decision(
        self,
        *,
        event_id: str,
        context: OfferContext,
        decided_at: datetime,
        offer_id: str,
        policy_name: str,
        propensity: float,
        commit: bool = True,
    ) -> None:
        self.connection.execute(
            "INSERT INTO offer_events "
            "(event_id, user_id, decided_at, offer_id, policy_name, propensity, context_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                context.user_id,
                to_sql_timestamp(decided_at),
                offer_id,
                policy_name,
                propensity,
                json.dumps(context.to_dict(), sort_keys=True),
            ),
        )
        if commit:
            self.connection.commit()

    def record_offer_feedback(
        self,
        *,
        event_id: str,
        converted: bool,
        revenue: float,
        feedback_at: datetime,
        commit: bool = True,
    ) -> None:
        cursor = self.connection.execute(
            "UPDATE offer_events SET converted = ?, revenue = ?, feedback_at = ? "
            "WHERE event_id = ? AND converted IS NULL",
            (int(converted), revenue, to_sql_timestamp(feedback_at), event_id),
        )
        if cursor.rowcount != 1:
            raise KeyError(f"no open offer event with id {event_id}")
        if commit:
            self.connection.commit()

    def commit(self) -> None:
        self.connection.commit()

    def table_counts(self) -> dict[str, int]:
        return {
            table: int(self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("users", "sessions", "purchases", "offer_events", "user_features")
        }
