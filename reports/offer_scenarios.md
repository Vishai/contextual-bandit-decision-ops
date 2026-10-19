# Offer Bandit Scenario Simulations

Each scenario replays the same 17,931 visits from the SQL feature store
(28 days, 2,000 users). All learners start with no
data. Regret is expected revenue lost against the best available offer under the
true model at that moment. The change point is halfway through. In the delay
scenario nothing changes there, so the two halves just show learning progress.
Ground truth is synthetic, and the main tables are single runs, so differences of
10 to 15% between similar learners can come down to the seed.

## Cold start offer launch

A new flash-sale offer launches halfway through. It is the best offer for engaged mobile users, but nobody has data on it. New users with no history also keep arriving the whole time.

| Policy | Conversion | Revenue / user | Total regret | Regret before change | Regret after change | Last-quarter regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rule_based | 13.17% | 47.34 | 21,983 | 7,243 | 14,740 | 7,364 |
| greedy | 11.29% | 42.81 | 30,914 | 12,918 | 17,996 | 9,126 |
| ucb | 12.05% | 51.98 | 13,158 | 4,786 | 8,373 | 3,159 |
| thompson_sampling | 11.72% | 50.89 | 14,013 | 8,940 | 5,073 | 2,052 |
| thompson_sampling_forgetting | 11.78% | 49.20 | 17,330 | 8,665 | 8,665 | 3,994 |

The oracle would send 37.7% of last-quarter traffic to the new offer.

| Policy | New-offer share (last quarter) | Regret per decision, users under 7 days | Regret per decision, returning users |
| --- | ---: | ---: | ---: |
| rule_based | 0.0% | 0.341 | 1.254 |
| greedy | 0.0% | 2.615 | 1.696 |
| ucb | 23.5% | 1.006 | 0.725 |
| thompson_sampling | 33.5% | 0.576 | 0.788 |
| thompson_sampling_forgetting | 33.5% | 0.982 | 0.966 |

## Delayed rewards

Conversions are reported after an exponential delay averaging 24 hours, so policies decide while lots of outcomes are still pending.

| Policy | Conversion | Revenue / user | Total regret | Regret before change | Regret after change | Last-quarter regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rule_based | 13.17% | 47.34 | 14,272 | 7,243 | 7,028 | 3,528 |
| greedy | 8.59% | 42.58 | 21,706 | 13,214 | 8,492 | 4,208 |
| ucb | 11.63% | 48.46 | 11,397 | 8,828 | 2,570 | 1,106 |
| thompson_sampling | 11.66% | 48.24 | 12,151 | 8,574 | 3,577 | 1,584 |
| thompson_sampling_forgetting | 11.33% | 46.43 | 16,074 | 10,587 | 5,486 | 2,732 |

By the end, 17,248 of 17,931 outcomes had reached the learners. The rest were still in flight.

### Thompson Sampling vs feedback delay

Thompson Sampling only, averaged over 3 seeds. The spread column is the standard
deviation of total regret between seeds.

| Mean delay | Total regret | Spread | Regret after change | Last-quarter regret | Revenue / user |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 h | 11,573 | 1,331 | 3,285 | 1,331 | 49.16 |
| 6 h | 12,229 | 468 | 3,359 | 1,387 | 48.68 |
| 24 h | 12,443 | 1,528 | 3,514 | 1,489 | 48.72 |
| 72 h | 14,206 | 523 | 3,886 | 1,621 | 47.64 |

## Shifting conversion rates

Halfway through, the bundle deal and free shipping stop working as well and the premium trial and loyalty points get better, like a competitor launch or a seasonal change.

| Policy | Conversion | Revenue / user | Total regret | Regret before change | Regret after change | Last-quarter regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rule_based | 11.85% | 41.27 | 46,989 | 7,243 | 39,745 | 19,730 |
| greedy | 13.84% | 42.94 | 42,746 | 12,918 | 29,828 | 13,525 |
| ucb | 13.79% | 58.36 | 13,156 | 4,786 | 8,370 | 2,259 |
| thompson_sampling | 12.22% | 55.48 | 18,407 | 8,940 | 9,467 | 2,300 |
| thompson_sampling_forgetting | 12.95% | 54.68 | 19,146 | 8,665 | 10,480 | 2,571 |

### Thompson Sampling vs forgetting factor

Thompson Sampling only, averaged over 3 seeds. The spread column is the standard
deviation of total regret between seeds.

| Forgetting | Total regret | Spread | Regret after change | Last-quarter regret | Revenue / user |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 20,000 | 1,382 | 11,712 | 3,110 | 55.02 |
| 0.9998 | 19,120 | 359 | 10,189 | 3,112 | 55.46 |
| 0.9995 | 21,262 | 554 | 11,238 | 4,535 | 54.49 |
| 0.999 | 23,102 | 377 | 11,965 | 5,439 | 53.59 |

## Takeaways

- Cold start: Thompson Sampling sent 34% of last-quarter traffic to the new offer against an oracle share of 38% (rule_based 0%, greedy 0%, ucb 24%, thompson_sampling_forgetting 34%). A wide posterior on an unseen offer is what gets it tried; greedy has no reason to, and the rules don't know the offer exists.
- Delayed rewards: with outcomes arriving late, Thompson Sampling finished with 12,151 regret against 14,272 for the rules. Going from 0 h to 72 h of average delay changed its regret by +23% in the sweep.
- Shifting rates: the rules can't adapt and lost 39,745 after the shift. The best learner here, ucb, lost 8,370. In the forgetting sweep the lowest after-change regret came from `0.9998`. Seed-to-seed spread is about 668, so treat small gaps between settings as noise.
