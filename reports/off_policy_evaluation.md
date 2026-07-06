# Off-Policy Evaluation

## Run configuration

| Setting | Value |
| --- | ---: |
| Logged events | 5000 |
| Seed | 42 |
| Actions | 3 |
| Behavior policy | random_uniform (propensity=0.3333) |
| Observed behavior reward | 0.2906 |
| Expected behavior reward from simulator | 0.2908 |
| Reward-model regularization | 1.00 |

## Target policies

- `random_uniform` — uniform probability over available actions
- `fixed_action_0` — always chooses the configured action
- `epsilon_greedy_0.10` — simulator-oracle greedy action with uniform epsilon exploration
- `greedy_oracle` — simulator-only upper-bound policy

## Estimator results

| Target policy | Estimator | Estimated value | ESS | Matched rows |
| --- | --- | ---: | ---: | ---: |
| random_uniform | Direct logged average | 0.2906 | 5000.0 | N/A |
| random_uniform | Replay / matching | 0.2882 | N/A | 1676 |
| random_uniform | IPS | 0.2906 | 5000.0 | N/A |
| random_uniform | SNIPS | 0.2906 | 5000.0 | N/A |
| random_uniform | Doubly robust | 0.2899 | 5000.0 | N/A |
| random_uniform | Simulator truth (reference) | 0.2906 | N/A | N/A |
| fixed_action_0 | Direct logged average | 0.2906 | 5000.0 | N/A |
| fixed_action_0 | Replay / matching | 0.2758 | N/A | 1657 |
| fixed_action_0 | IPS | 0.2742 | 1657.0 | N/A |
| fixed_action_0 | SNIPS | 0.2758 | 1657.0 | N/A |
| fixed_action_0 | Doubly robust | 0.2746 | 1657.0 | N/A |
| fixed_action_0 | Simulator truth (reference) | 0.2733 | N/A | N/A |
| epsilon_greedy_0.10 | Direct logged average | 0.2906 | 5000.0 | N/A |
| epsilon_greedy_0.10 | Replay / matching | 0.3349 | N/A | 1678 |
| epsilon_greedy_0.10 | IPS | 0.3309 | 1902.3 | N/A |
| epsilon_greedy_0.10 | SNIPS | 0.3321 | 1902.3 | N/A |
| epsilon_greedy_0.10 | Doubly robust | 0.3344 | 1902.3 | N/A |
| epsilon_greedy_0.10 | Simulator truth (reference) | 0.3385 | N/A | N/A |
| greedy_oracle | Direct logged average | 0.2906 | 5000.0 | N/A |
| greedy_oracle | Replay / matching | 0.3367 | N/A | 1660 |
| greedy_oracle | IPS | 0.3354 | 1660.0 | N/A |
| greedy_oracle | SNIPS | 0.3367 | 1660.0 | N/A |
| greedy_oracle | Doubly robust | 0.3394 | 1660.0 | N/A |
| greedy_oracle | Simulator truth (reference) | 0.3438 | N/A | N/A |

## Why weighting can be unstable

IPS divides each target-policy probability by the logged propensity. Rare logged actions therefore create large weights, high variance, and a small effective sample size. SNIPS normalizes the weights, which controls their overall scale but does not repair weak action support or eliminate variance.

## Interpretation

The direct logged average measures the behavior policy and is repeated only as a deliberately naive target-policy comparison. Replay is intuitive but discards non-matching rows.
In this synthetic run, the behavior policy has known uniform propensities and positive support for every action, so IPS and SNIPS are meaningful. The doubly robust estimate combines a fitted linear reward prediction with an IPS residual correction and is generally the most informative estimate when either component is reasonably specified.
Simulator truth is shown only for validation. These estimates do not establish real production policy quality.
