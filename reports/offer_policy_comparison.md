# Offer Policy Comparison

Every policy sees the same stream of visits built from the SQL feature store and the
same random draws for conversion. Learning policies start with no data and update
after each outcome. Conversions come from a synthetic ground truth model, so these
numbers show how the policies behave relative to each other, not real business results.

## Setup

| Setting | Value |
| --- | ---: |
| Decisions | 17931 |
| Unique users | 1923 |
| Simulated days | 28 |
| Activity seed | 11 |
| Policy seed | 21 |

## Results

| Policy | Conversion | Revenue / user | Revenue / decision | Expected revenue / decision | Cumulative regret | Second-half regret | p50 ms | p99 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 8.53% | 31.69 | 3.399 | 3.332 | 47,763 | 22,442 | 0.026 | 0.071 |
| rule_based | 13.65% | 48.80 | 5.233 | 5.200 | 14,272 | 6,631 | 0.024 | 0.069 |
| greedy | 12.22% | 50.44 | 5.410 | 5.312 | 12,263 | 5,056 | 0.101 | 0.292 |
| ucb | 12.14% | 52.80 | 5.662 | 5.573 | 7,592 | 1,601 | 0.101 | 0.265 |
| thompson_sampling | 12.49% | 52.56 | 5.637 | 5.474 | 9,365 | 2,070 | 0.106 | 0.286 |
| beta_thompson_sampling | 10.78% | 47.06 | 5.047 | 4.976 | 18,297 | 7,973 | 0.052 | 0.143 |
| oracle | 13.78% | 56.75 | 6.086 | 5.996 | 0 | 0 | 0.041 | 0.112 |

## Against the rule-based strategy

| Policy | Conversion change (pts) | Revenue / user lift | Regret change |
| --- | ---: | ---: | ---: |
| random | -5.12 | -35.0% | +234.7% |
| greedy | -1.43 | +3.4% | -14.1% |
| ucb | -1.51 | +8.2% | -46.8% |
| thompson_sampling | -1.17 | +7.7% | -34.4% |
| beta_thompson_sampling | -2.87 | -3.6% | +28.2% |
| oracle | +0.12 | +16.3% | -100.0% |

## Allocation

| Policy | pct10_discount | free_shipping | bundle_deal | premium_trial | loyalty_points |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 19.9% | 19.9% | 20.0% | 20.0% | 20.2% |
| rule_based | 21.6% | 41.3% | 18.4% | 1.5% | 17.3% |
| greedy | 9.5% | 49.5% | 40.8% | 0.2% | 0.0% |
| ucb | 12.5% | 31.6% | 19.8% | 29.0% | 7.1% |
| thompson_sampling | 9.6% | 34.2% | 20.7% | 25.3% | 10.2% |
| beta_thompson_sampling | 1.0% | 52.6% | 8.8% | 35.6% | 2.0% |
| oracle | 10.9% | 34.9% | 21.8% | 19.5% | 12.9% |

## Cumulative regret over time

| Checkpoint | random | rule_based | greedy | ucb | thompson_sampling | beta_thompson_sampling | oracle |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25% | 10,165 | 2,988 | 3,549 | 3,644 | 4,742 | 4,590 | 0 |
| 50% | 22,863 | 6,885 | 6,660 | 5,750 | 6,918 | 9,439 | 0 |
| 75% | 35,358 | 10,583 | 9,414 | 6,702 | 8,362 | 13,927 | 0 |
| 100% | 47,763 | 14,272 | 12,263 | 7,592 | 9,365 | 18,297 | 0 |

## Notes

- Regret is the expected revenue lost against the best offer for each visit under
  the true model. It is the least noisy way to compare the learners.
- `greedy`, `ucb` and `thompson_sampling` share one Bayesian logistic model, so
  they differ only in how they explore.
- `beta_thompson_sampling` only splits by device, which shows how much the full
  user context is worth.
- `oracle` knows the true conversion rates and is only a reference point.
- Latency is the in-process time for one choose plus one update.
