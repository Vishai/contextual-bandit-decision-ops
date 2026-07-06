# Staging Observability Report

## Run configuration

| Setting | Value |
| --- | ---: |
| Source | deterministic staging fixture logs |
| Reference events | 1000 |
| Current events | 1000 |
| Seed | 42 |
| Actions | 3 |
| Action TV warning threshold | 0.1000 |
| Reward-rate warning threshold | 0.0300 |
| Minimum healthy propensity | 0.0500 |
| Maximum missing feedback | 20.00% |
| Minimum exploration | 10.00% |

## Log summary

| Window | Decisions | Feedback | Missing feedback | Avg reward | Exploration | Min propensity | Max propensity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reference | 1000 | 1000 | 0.00% | 0.2810 | 65.20% | 0.3333 | 0.3333 |
| current | 1000 | 750 | 25.00% | 0.3627 | 50.00% | 0.0333 | 0.9333 |

## Action distribution

| Window | Action | Count | Share |
| --- | ---: | ---: | ---: |
| reference | 0 | 307 | 30.70% |
| reference | 1 | 345 | 34.50% |
| reference | 2 | 348 | 34.80% |
| current | 0 | 180 | 18.00% |
| current | 1 | 500 | 50.00% |
| current | 2 | 320 | 32.00% |

## Metric table

| Metric | Reference | Current | Shift |
| --- | ---: | ---: | ---: |
| Action distribution (TV) | — | — | 0.1550 |
| Reward rate | 0.2810 | 0.3627 | 0.0817 |
| Context age mean / std | 44.55 / 14.90 | 47.65 / 14.89 | mean_z=0.208 |
| Context engagement mean / std | 0.487 / 0.287 | 0.641 / 0.277 | mean_z=0.536 |
| Context region distribution (TV) | — | — | 0.0170 |

## Passed checks

| Check | Observed | Threshold | Explanation |
| --- | ---: | ---: | --- |
| context_region_shift | 0.0170 | <= 0.1000 | Total variation distance compares region distributions. |
| propensity_validity | 0 | 0 invalid values | Propensities must be finite and in (0, 1]. |
| low_exploration | 50.00% | >= 10.00% | Exploration is one minus the largest current action share. |
| service_counters | decisions=1000, feedback=750 | informational | Local API decision and feedback counters were summarized. |

## Warnings

| Check | Observed | Threshold | Explanation |
| --- | ---: | ---: | --- |
| action_distribution_shift | 0.1550 | <= 0.1000 | Total variation distance compares reference and current action shares. |
| reward_rate_shift | 0.0817 | <= 0.0300 | Absolute reward-rate change compares feedback windows. |
| context_age_shift | mean_z=0.208, std_rel=0.001 | mean_z <= 0.200, std_rel <= 0.250 | Age drift compares standardized mean and relative standard deviation. |
| context_engagement_shift | mean_z=0.536, std_rel=0.034 | mean_z <= 0.200, std_rel <= 0.250 | Engagement drift compares standardized mean and relative standard deviation. |
| low_propensity | 59 (5.90%) | 0 below 0.0500 | Very small propensities can make importance weights unstable. |
| missing_feedback_rate | 25.00% | <= 20.00% | Missing feedback is the share of decision ids without an outcome. |

## Failed checks

- None

## Interpretation

Warnings identify transparent staging changes rather than proving an incident. The deterministic fixture intentionally shifts context, policy behavior, and feedback coverage so the checks remain easy to inspect.
Invalid propensities are treated as failures because they break logged-policy reasoning; small but valid propensities remain warnings because they can make importance-weighted estimates unstable.
This report is local/staging diagnostics only. It is not production observability, alerting, or a launch guarantee.
