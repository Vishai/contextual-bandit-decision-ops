# Policy Promotion Gate

**Final decision: HOLD**

Hold all candidates; none passed every configured check.

## Run configuration

| Setting | Value |
| --- | ---: |
| Simulation events | 5000 |
| OPE logged events | 5000 |
| Seed | 42 |
| Actions | 3 |
| Evidence source | deterministic synthetic simulation and OPE |

## Constraints

| Constraint | Requirement |
| --- | ---: |
| Blocked actions | none |
| Maximum action share | 70.00% |
| Minimum exploration rate | 10.00% |
| Minimum count per allowed action | 10 |
| Minimum OPE effective sample size | 1000.0 |
| Minimum replay matches | 1000 |
| Minimum estimated improvement | 0.0100 |
| Maximum average simulation regret | 0.0300 |
| External evidence required | yes |

## Candidate policies

- `fixed_action_0` — capacity-stressing deterministic baseline
- `epsilon_greedy_0.10` — simulator-oracle-assisted baseline with uniform exploration

## Metric summary

| Candidate | DR value | Improvement | OPE ESS | Replay matches | Max share | Exploration | Avg regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_action_0 | 0.2746 | -0.0153 | 1657.0 | 1657 | 100.00% | 0.00% | 0.0692 |
| epsilon_greedy_0.10 | 0.3344 | 0.0445 | 1902.3 | 1678 | 47.80% | 52.20% | 0.0057 |

## Pass/fail checklist

| Candidate | Check | Status | Observed | Requirement |
| --- | --- | --- | ---: | ---: |
| fixed_action_0 | blocked_actions | PASS | 0 | 0 selections |
| fixed_action_0 | ope_effective_sample_size | PASS | 1657.0 | >= 1000.0 |
| fixed_action_0 | replay_matched_count | PASS | 1657 | >= 1000 |
| fixed_action_0 | max_action_share | FAIL | 100.00% | <= 70.00% |
| fixed_action_0 | minimum_exploration_rate | FAIL | 0.00% | >= 10.00% |
| fixed_action_0 | action_coverage | FAIL | 0 | >= 10 per allowed action |
| fixed_action_0 | estimated_improvement | FAIL | -0.0153 | >= 0.0100 |
| fixed_action_0 | simulation_regret | FAIL | 0.0692 | <= 0.0300 |
| fixed_action_0 | non_synthetic_evidence | FAIL | synthetic | external validation required |
| epsilon_greedy_0.10 | blocked_actions | PASS | 0 | 0 selections |
| epsilon_greedy_0.10 | max_action_share | PASS | 47.80% | <= 70.00% |
| epsilon_greedy_0.10 | minimum_exploration_rate | PASS | 52.20% | >= 10.00% |
| epsilon_greedy_0.10 | action_coverage | PASS | 1273 | >= 10 per allowed action |
| epsilon_greedy_0.10 | ope_effective_sample_size | PASS | 1902.3 | >= 1000.0 |
| epsilon_greedy_0.10 | replay_matched_count | PASS | 1678 | >= 1000 |
| epsilon_greedy_0.10 | estimated_improvement | PASS | 0.0445 | >= 0.0100 |
| epsilon_greedy_0.10 | simulation_regret | PASS | 0.0057 | <= 0.0300 |
| epsilon_greedy_0.10 | non_synthetic_evidence | FAIL | synthetic | external validation required |

## Warnings

- All evidence is synthetic and does not establish real-world safety.
- Observed action concentration exceeds the configured capacity cap.
- This epsilon-greedy candidate uses simulator-oracle scores and is not deployable.

## Interpretation

A higher estimated reward is not sufficient for promotion. The fixed-action candidate fails concentration, exploration, coverage, improvement, and regret checks. The epsilon-greedy candidate has stronger synthetic metrics, but it uses simulator-oracle information and lacks external evidence.
The gate therefore holds every candidate. This demonstrates deterministic simulated readiness logic; it is not a real-world safety certification or launch authorization.
