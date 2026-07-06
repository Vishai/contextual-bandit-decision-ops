# Contextual Learning Policy Comparison

## Run configuration

| Setting | Value |
| --- | ---: |
| Events | 5000 |
| Seed | 42 |
| Actions | 3 |
| Epsilon | 0.10 |
| LinUCB alpha | 0.50 |
| Thompson scale | 0.25 |
| Regularization | 1.00 |

## Policy descriptions

| Policy | Description |
| --- | --- |
| random_uniform | Chooses every available action with equal probability. |
| online_epsilon_greedy_0.10 | Learns per-action reward averages and explores with probability epsilon. |
| linucb | Fits one ridge-linear reward model per action and adds a confidence bonus. |
| linear_thompson_sampling | Samples coefficients from each action's approximate linear posterior. |
| greedy_oracle | Selects the simulator's highest-probability action as an upper bound. |

## Learning results

| Policy | Total reward | Average reward | Cumulative regret |
| --- | ---: | ---: | ---: |
| random_uniform | 1392 | 0.278 | 261.68 |
| online_epsilon_greedy_0.10 | 1470 | 0.294 | 208.36 |
| linucb | 1624 | 0.325 | 68.42 |
| linear_thompson_sampling | 1566 | 0.313 | 120.96 |
| greedy_oracle | 1685 | 0.337 | 0.00 |

## Final action distribution

| Policy | Action | Count | Share |
| --- | ---: | ---: | ---: |
| random_uniform | 0 | 1630 | 32.60% |
| random_uniform | 1 | 1699 | 33.98% |
| random_uniform | 2 | 1671 | 33.42% |
| online_epsilon_greedy_0.10 | 0 | 284 | 5.68% |
| online_epsilon_greedy_0.10 | 1 | 4560 | 91.20% |
| online_epsilon_greedy_0.10 | 2 | 156 | 3.12% |
| linucb | 0 | 673 | 13.46% |
| linucb | 1 | 3193 | 63.86% |
| linucb | 2 | 1134 | 22.68% |
| linear_thompson_sampling | 0 | 2013 | 40.26% |
| linear_thompson_sampling | 1 | 713 | 14.26% |
| linear_thompson_sampling | 2 | 2274 | 45.48% |
| greedy_oracle | 0 | 1273 | 25.46% |
| greedy_oracle | 1 | 2506 | 50.12% |
| greedy_oracle | 2 | 1221 | 24.42% |

## Interpretation

`linucb` had the highest realized average reward among non-oracle policies (0.325).
LinUCB and Thompson Sampling improve by updating action-specific linear models from observed rewards; their regret includes early exploration.
The greedy oracle sees the simulator reward probabilities and is only an upper bound. This is deterministic online-style simulation, not a production online learner or off-policy evaluation.
