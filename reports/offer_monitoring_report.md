# Offer Service Monitoring

**Overall status: ALERT.** Source: simulated traffic through the API: a normal week, then a mobile campaign with new users and shifted conversion rates.

## Checks

| Check | Status | Detail |
| --- | --- | --- |
| latency_p99 | ok | p99 1.10 ms against a 100 ms budget |
| latency_budget_rate | ok | 100.00% of decisions inside the budget |
| feature_drift | ALERT | 4 of 14 features shifted (engagement_score, sessions_30d, tenure_days, device); largest is device at TVD 0.201 |
| allocation_shift | WARN | total variation 0.191 between the offer mixes |
| offer_starvation | ok | every offer still gets traffic |
| propensity_health | ok | min propensity 0.0100, 0.00% under 0.005 |
| feedback_coverage | ok | 100.0% of decisions have an outcome |
| conversion_change | ok | 11.83% to 12.29% (+0.46 pts) |
| revenue_change | WARN | 5.401 to 4.798 per decision (-11%) |
| estimated_regret | ok | 0.738 to 0.410 per decision (-44%) |
| true_regret_simulation | WARN | 0.583 to 1.839 per decision (+215%); only measurable in simulation |

## Windows

| Metric | Reference | Current |
| --- | ---: | ---: |
| From | 2025-03-15T00:00:45+00:00 | 2025-03-22T00:02:35+00:00 |
| To | 2025-03-21T02:58:52+00:00 | 2025-03-28T23:57:18+00:00 |
| Decisions | 3,000 | 2,872 |
| Unique users | 1,071 | 1,020 |
| Conversion | 11.83% | 12.29% |
| Revenue / decision | 5.401 | 4.798 |
| Feedback coverage | 100.0% | 100.0% |
| Exploration rate | 5.0% | 5.0% |
| Decision latency p50 ms | 0.39 | 0.39 |
| Decision latency p95 ms | 0.65 | 0.72 |
| Decision latency p99 ms | 0.84 | 1.10 |
| Minimum propensity | 0.0100 | 0.0100 |
| Regret / decision, model estimate | 0.738 | 0.410 |
| Regret / decision, true (simulation only) | 0.583 | 1.839 |

The model estimate scores each logged decision against the best offer under the
served model's own posterior. It shows how much the policy is still exploring, but it
can't see where the model itself is wrong. True regret needs the simulator.

## Offer allocation

| Offer | Reference | Current | Conversion (reference) | Conversion (current) |
| --- | ---: | ---: | ---: | ---: |
| pct10_discount | 7.6% | 18.5% | 9.21% | 13.94% |
| free_shipping | 31.4% | 28.3% | 13.50% | 8.23% |
| bundle_deal | 22.7% | 14.8% | 12.94% | 3.99% |
| premium_trial | 24.5% | 16.2% | 8.58% | 13.98% |
| loyalty_points | 13.9% | 22.1% | 13.43% | 20.44% |

### Allocation over time

Both windows in order, split into equal slices of decisions.

| Slice | pct10_discount | free_shipping | bundle_deal | premium_trial | loyalty_points |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.2% | 29.6% | 23.4% | 30.0% | 11.9% |
| 2 | 7.2% | 26.6% | 25.3% | 25.3% | 15.5% |
| 3 | 7.9% | 36.1% | 20.7% | 22.6% | 12.7% |
| 4 | 9.7% | 32.2% | 21.5% | 21.8% | 14.9% |
| 5 | 9.8% | 34.7% | 19.5% | 15.4% | 20.6% |
| 6 | 16.5% | 29.6% | 16.5% | 17.0% | 20.4% |
| 7 | 21.8% | 25.5% | 13.9% | 15.3% | 23.6% |
| 8 | 25.3% | 24.9% | 9.8% | 15.9% | 24.0% |

## Data drift

PSI for numeric features (under 0.1 stable, 0.1 to 0.25 moderate, over 0.25 large),
total variation distance for categorical ones. PSI runs high on small windows, so with
only a few hundred decisions per window a value near 0.1 can be noise.

| Feature | Statistic | Value | Status |
| --- | --- | ---: | --- |
| device | TVD | 0.201 | ALERT |
| sessions_30d | PSI | 0.174 | WARN |
| tenure_days | PSI | 0.165 | WARN |
| engagement_score | PSI | 0.115 | WARN |
| has_purchased | TVD | 0.090 | ok |
| revenue_90d | PSI | 0.078 | ok |
| purchases_90d | PSI | 0.074 | ok |
| sessions_7d | PSI | 0.070 | ok |
| avg_order_value | PSI | 0.065 | ok |
| region | TVD | 0.065 | ok |
| days_since_last_visit | PSI | 0.017 | ok |
| offer_impressions_30d | PSI | 0.000 | ok |
| offer_conversions_30d | PSI | 0.000 | ok |
| feature_source | TVD | 0.000 | ok |
