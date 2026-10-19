# Retraining Run

**Status: promoted.** Candidate passed every gate.

| Item | Value |
| --- | --- |
| New model version | offer-ts-20261019T193312Z |
| Previous model version | none |
| Data window | 2025-02-28T23:58:43+00:00 to 2025-03-28T23:58:43+00:00 |
| Training events | 13,448 |
| Holdout events | 4,483 |
| Service told to reload | no URL given |

## Log ingestion

| Decisions read | Decisions added | Feedback read | Feedback applied | Feedback without a decision |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 | 0 |

## Holdout evaluation (doubly robust)

| Policy | Conversion | Revenue / decision | Effective sample size |
| --- | ---: | ---: | ---: |
| candidate | 12.14% | 4.826 | 464 |
| rule_based | 13.50% | 5.088 | 3,752 |

## Promotion gate

A candidate only fails a comparison when the 95% interval shows it is worse by
more than the allowed margin.

| Check | Result | Detail |
| --- | --- | --- |
| enough_training_data | pass | 13,448 training events (minimum 1,000) |
| holdout_support | pass | effective sample size 464 (minimum 100) |
| revenue_vs_rules | pass | lift -5.1% (95% interval -25.9% to +29.4%), 4.826 vs 5.088 |
| conversion_vs_rules | pass | change -1.36 pts (95% interval -4.06 to +1.55), 12.14% vs 13.50% |

## Recent registry entries

| Recorded at | Status | Version | Reason |
| --- | --- | --- | --- |
| 2026-10-19T19:33:13.106372+00:00 | promoted | offer-ts-20261019T193312Z |  |
