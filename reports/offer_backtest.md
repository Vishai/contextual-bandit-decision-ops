# Offer Backtest

The old production system (`rule_based_eps0.20`: rule-based targeting with
20% uniform exploration) is replayed over 42 days of
synthetic traffic, and every decision goes into the SQLite `offer_events` table with
its propensity. Policies are trained on the first
28 days of that history and evaluated on the rest.

Outcomes come from a synthetic ground truth model. That is what lets this report
check each offline estimate against the true value, but it also means the numbers
describe the simulator, not a live business.

## Historical data

| Metric | Value |
| --- | ---: |
| Training events | 17,854 |
| Held-out events | 9,494 |
| Held-out users | 1,804 |
| Logged conversion rate (all history) | 12.47% |
| Logged revenue per decision (all history) | 4.904 |

## Offline policy evaluation on the held-out window

Each trained policy is frozen and scored on held-out logs it never saw.

| Policy | Conversion IPS | Conversion SNIPS | Conversion DR | True conversion | Revenue/decision SNIPS | Revenue/decision DR | True revenue/decision | ESS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 8.37% | 8.41% | 8.42% | 8.96% | 3.399 | 3.451 | 3.552 | 2,338 |
| rule_based | 13.42% | 13.40% | 13.42% | 13.86% | 5.242 | 5.249 | 5.416 | 7,984 |
| greedy | 14.26% | 14.84% | 14.60% | 14.74% | 5.898 | 5.942 | 6.070 | 563 |
| ucb | 13.00% | 13.48% | 13.17% | 13.79% | 5.634 | 5.723 | 5.933 | 496 |
| thompson_sampling | 13.49% | 13.94% | 13.70% | 13.85% | 5.765 | 5.814 | 5.831 | 870 |

### Lift over the rule-based strategy (doubly robust)

Intervals are 95% paired bootstrap intervals that resample held-out users, so a user's repeat visits count as one draw.

| Policy | Conversion change (pts) | 95% interval | Revenue / user | Lift | 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | -5.00 | -6.12 to -3.83 | 18.16 | -34.2% | -42.9% to -25.8% |
| rule_based | +0.00 | baseline | 27.62 | +0.0% | baseline |
| greedy | +1.18 | -1.44 to +4.41 | 31.27 | +13.2% | -7.9% to +41.0% |
| ucb | -0.24 | -2.79 to +2.59 | 30.12 | +9.0% | -14.1% to +31.4% |
| thompson_sampling | +0.28 | -2.20 to +2.60 | 30.60 | +10.8% | -9.1% to +31.1% |

## Forward simulation after warm start

Each policy is trained on the same history, then run on the held-out visits while
it keeps learning from every outcome. Outcomes come from the true model.

| Policy | Conversion | Change vs rules (pts) | Revenue / user | Lift vs rules | Cumulative regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 8.52% | -4.98 | 17.66 | -36.2% | 26,841 |
| rule_based | 13.50% | +0.00 | 27.67 | +0.0% | 8,962 |
| greedy | 14.54% | +1.03 | 31.88 | +15.3% | 1,657 |
| ucb | 13.88% | +0.38 | 32.16 | +16.2% | 1,954 |
| thompson_sampling | 13.97% | +0.46 | 31.10 | +12.4% | 3,216 |

## Replay of learning from scratch

Every learner starts empty and walks the full history in order. An event counts
only when the learner picks the logged offer and passes a propensity-based
acceptance draw, so the kept events behave like uniformly logged data. It's
unbiased but uses only a few percent of the logs.

| Policy | Accepted events | Conversion | Revenue / decision |
| --- | ---: | ---: | ---: |
| random | 1,047 | 9.07% | 3.181 |
| rule_based | 1,135 | 13.66% | 5.377 |
| greedy | 1,064 | 9.77% | 4.657 |
| ucb | 1,043 | 10.26% | 5.125 |
| thompson_sampling | 1,071 | 11.11% | 4.214 |

## Reading the results

- IPS and SNIPS only use logged outcomes and propensities. Doubly robust adds a
  logistic reward model fit on the training window, which cuts variance.
- ESS is the effective sample size of the importance weights. Policies that
  disagree with the rules a lot only match a small slice of the logs, so their
  intervals are wide.
- Revenue per user is revenue per decision times the held-out decisions per user.
- The replay starts every learner cold on about 4% of the logs, which is why
  model-based learners trail the rules there. The warm-started runs above show
  how they do with the full history.
