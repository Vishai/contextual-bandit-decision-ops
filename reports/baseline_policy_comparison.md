# Baseline Policy Comparison

## Run configuration

| Setting | Value |
| --- | ---: |
| Events | 1000 |
| Seed | 42 |
| Actions | 3 |
| Fixed action | 0 |
| Epsilon | 0.10 |

## Policy results

| Policy | Total reward | Average reward | Regret estimate |
| --- | ---: | ---: | ---: |
| random_uniform | 282 | 0.282 | 0.0520 |
| fixed_action_0 | 278 | 0.278 | 0.0683 |
| greedy_oracle | 341 | 0.341 | 0.0000 |
| epsilon_greedy_0.10 | 337 | 0.337 | 0.0056 |

## Action distribution

| Policy | Action | Count | Share |
| --- | ---: | ---: | ---: |
| random_uniform | 0 | 307 | 30.70% |
| random_uniform | 1 | 345 | 34.50% |
| random_uniform | 2 | 348 | 34.80% |
| fixed_action_0 | 0 | 1000 | 100.00% |
| fixed_action_0 | 1 | 0 | 0.00% |
| fixed_action_0 | 2 | 0 | 0.00% |
| greedy_oracle | 0 | 261 | 26.10% |
| greedy_oracle | 1 | 505 | 50.50% |
| greedy_oracle | 2 | 234 | 23.40% |
| epsilon_greedy_0.10 | 0 | 274 | 27.40% |
| epsilon_greedy_0.10 | 1 | 483 | 48.30% |
| epsilon_greedy_0.10 | 2 | 243 | 24.30% |

## Interpretation

`greedy_oracle` had the highest realized average reward (0.341) on this deterministic replay.
The greedy oracle can inspect the simulator reward model, so it is an upper-bound baseline rather than a deployable online policy.
Regret is the average expected reward-probability gap to that oracle. These results describe an offline simulator comparison, not real online learning.
