# Feature Pipeline Summary

Synthetic raw activity is loaded into SQLite, and one point-in-time feature snapshot
is built each simulated morning with the SQL in `feature_sql.py`. Every visit in the
simulation window becomes a decision point that reads that morning's snapshot.

## Run

| Setting | Value |
| --- | ---: |
| Users | 2000 |
| Seed | 11 |
| History window (days) | 90 |
| Simulation window (days) | 28 |
| Simulation start | 2025-03-01 |

## Tables

| Table | Rows |
| --- | ---: |
| `users` | 2000 |
| `sessions` | 67588 |
| `purchases` | 7081 |
| `offer_events` | 0 |
| `user_features` | 52613 |

## Decision points

| Metric | Value |
| --- | ---: |
| Decision points | 17931 |
| Unique users | 1923 |
| Cold-start decisions (no snapshot row yet) | 39 |
| Decisions for users who never purchased | 3873 |
| Mean sessions in last 30 days | 16.37 |
| Mean purchases in last 90 days | 4.76 |

| Device | Share |
| --- | ---: |
| mobile | 53.9% |
| desktop | 35.7% |
| tablet | 10.4% |

| Region | Share |
| --- | ---: |
| north | 24.1% |
| south | 26.7% |
| east | 24.1% |
| west | 25.0% |

## Encoded features

| Feature | Mean | Min | Max |
| --- | ---: | ---: | ---: |
| `bias` | 1.000 | 1.000 | 1.000 |
| `engagement_score` | 0.528 | 0.000 | 1.000 |
| `sessions_7d_log` | 0.500 | 0.000 | 1.203 |
| `sessions_30d_log` | 0.621 | 0.000 | 1.141 |
| `visit_recency` | 0.738 | 0.000 | 1.000 |
| `tenure_log` | 0.822 | 0.000 | 1.003 |
| `purchases_90d_log` | 0.493 | 0.000 | 1.533 |
| `avg_order_value_log` | 0.559 | 0.000 | 0.980 |
| `purchase_recency` | 0.472 | 0.000 | 1.000 |
| `never_purchased` | 0.216 | 0.000 | 1.000 |
| `offer_response_rate_30d` | 0.000 | 0.000 | 0.000 |
| `device_mobile` | 0.539 | 0.000 | 1.000 |
| `device_desktop` | 0.357 | 0.000 | 1.000 |
| `device_tablet` | 0.104 | 0.000 | 1.000 |
| `region_north` | 0.241 | 0.000 | 1.000 |
| `region_south` | 0.267 | 0.000 | 1.000 |
| `region_east` | 0.241 | 0.000 | 1.000 |
| `region_west` | 0.250 | 0.000 | 1.000 |
