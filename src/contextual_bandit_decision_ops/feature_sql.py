"""SQL that builds per-user offer features as of a point in time.

Every aggregate only looks at rows strictly before ``:as_of``. Offer conversions
only count once their feedback has arrived before ``:as_of`` too, so a snapshot
never leaks an outcome the model could not have seen yet.
"""

FEATURE_COLUMNS = (
    "user_id",
    "device",
    "region",
    "engagement_score",
    "sessions_7d",
    "sessions_30d",
    "days_since_last_visit",
    "tenure_days",
    "purchases_90d",
    "revenue_90d",
    "avg_order_value",
    "days_since_last_purchase",
    "offer_impressions_30d",
    "offer_conversions_30d",
)

FEATURE_QUERY = """
WITH
session_window AS (
    SELECT
        user_id,
        SUM(CASE WHEN started_at >= datetime(:as_of, '-7 days') THEN 1 ELSE 0 END)
            AS sessions_7d,
        COUNT(*) AS sessions_30d,
        AVG(duration_minutes) AS avg_session_minutes_30d
    FROM sessions
    WHERE started_at < :as_of
      AND started_at >= datetime(:as_of, '-30 days')
    GROUP BY user_id
),
last_session AS (
    SELECT user_id, MAX(started_at) AS last_session_at
    FROM sessions
    WHERE started_at < :as_of
    GROUP BY user_id
),
purchase_history AS (
    SELECT
        user_id,
        SUM(CASE WHEN purchased_at >= datetime(:as_of, '-90 days') THEN 1 ELSE 0 END)
            AS purchases_90d,
        SUM(CASE WHEN purchased_at >= datetime(:as_of, '-90 days') THEN amount ELSE 0 END)
            AS revenue_90d,
        AVG(amount) AS avg_order_value,
        MAX(purchased_at) AS last_purchase_at
    FROM purchases
    WHERE purchased_at < :as_of
    GROUP BY user_id
),
offer_history AS (
    SELECT
        user_id,
        COUNT(*) AS offer_impressions_30d,
        SUM(CASE WHEN converted = 1 AND feedback_at < :as_of THEN 1 ELSE 0 END)
            AS offer_conversions_30d
    FROM offer_events
    WHERE decided_at < :as_of
      AND decided_at >= datetime(:as_of, '-30 days')
    GROUP BY user_id
)
SELECT
    u.user_id,
    u.device,
    u.region,
    MIN(
        1.0,
        0.6 * MIN(COALESCE(sw.sessions_30d, 0), 20) / 20.0
        + 0.4 * MIN(COALESCE(sw.avg_session_minutes_30d, 0.0), 30.0) / 30.0
    ) AS engagement_score,
    COALESCE(sw.sessions_7d, 0) AS sessions_7d,
    COALESCE(sw.sessions_30d, 0) AS sessions_30d,
    julianday(:as_of) - julianday(COALESCE(ls.last_session_at, u.signup_at))
        AS days_since_last_visit,
    julianday(:as_of) - julianday(u.signup_at) AS tenure_days,
    COALESCE(ph.purchases_90d, 0) AS purchases_90d,
    COALESCE(ph.revenue_90d, 0.0) AS revenue_90d,
    COALESCE(ph.avg_order_value, 0.0) AS avg_order_value,
    CASE
        WHEN ph.last_purchase_at IS NULL THEN NULL
        ELSE julianday(:as_of) - julianday(ph.last_purchase_at)
    END AS days_since_last_purchase,
    COALESCE(oh.offer_impressions_30d, 0) AS offer_impressions_30d,
    COALESCE(oh.offer_conversions_30d, 0) AS offer_conversions_30d
FROM users AS u
LEFT JOIN session_window AS sw ON sw.user_id = u.user_id
LEFT JOIN last_session AS ls ON ls.user_id = u.user_id
LEFT JOIN purchase_history AS ph ON ph.user_id = u.user_id
LEFT JOIN offer_history AS oh ON oh.user_id = u.user_id
WHERE u.signup_at < :as_of
"""

MATERIALIZE_QUERY = f"""
INSERT OR REPLACE INTO user_features (as_of, {", ".join(FEATURE_COLUMNS)})
SELECT :as_of, {", ".join(FEATURE_COLUMNS)}
FROM ({FEATURE_QUERY})
"""

SNAPSHOT_QUERY = f"""
SELECT {", ".join(FEATURE_COLUMNS)}
FROM user_features
WHERE as_of = :as_of
ORDER BY user_id
"""

LATEST_USER_FEATURES_QUERY = f"""
SELECT as_of, {", ".join(FEATURE_COLUMNS)}
FROM user_features
WHERE user_id = :user_id AND as_of <= :as_of
ORDER BY as_of DESC
LIMIT 1
"""
