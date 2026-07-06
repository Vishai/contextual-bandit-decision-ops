# Policy Card

## Release

- Project: Contextual Bandit Decision Ops
- Version: 0.1.0
- Status: local/staging demonstration
- Promotion decision: **HOLD**

## Intended use

This project demonstrates how contextual decision policies can be simulated,
compared, evaluated from logged data, constrained, served, and monitored in a
reproducible development environment. It is suitable for education, portfolio
review, implementation discussion, and local experimentation.

## Non-goals

It is not a validated policy for real users, a production recommendation service,
or evidence of causal impact in a real domain. The abstract actions must not be
mapped to consequential interventions without domain review, experimentation,
privacy controls, and production validation.

## Inputs and available actions

The simulator uses age, engagement, and region context features. Available actions
are the abstract integer labels `0`, `1`, and `2`. They represent alternative
treatments only inside the synthetic environment; no real product meaning is
assigned to them.

## Policy candidates

| Policy | Role | Uses observed feedback? |
|---|---|---:|
| Random uniform | Exploration and evaluation baseline | No |
| Fixed action | Simple control and OPE support test | No |
| Epsilon-greedy | Exploit/explore baseline; oracle-assisted in default reports | No |
| LinUCB | Optimistic linear contextual learner | Yes |
| Linear Thompson Sampling | Posterior-sampling linear contextual learner | Yes |
| Greedy oracle | Simulation upper reference, not deployable | No |

The default epsilon-greedy implementation uses simulator reward probabilities to
identify its greedy action. It must therefore be treated as an oracle-assisted
benchmark, not as an independently learned launch candidate.

## Evaluation evidence

All values below come from deterministic synthetic runs.

| Evidence | Result |
|---|---|
| Baseline simulation, 1,000 events | Random average reward `0.282`; oracle `0.341`; epsilon-greedy `0.337` |
| Sequential learning, 5,000 events | Random `0.278`; LinUCB `0.325`; Thompson Sampling `0.313`; oracle `0.337` |
| LinUCB cumulative regret | `68.42` versus `261.68` for random |
| OPE, 5,000 uniform-policy events | Oracle DR estimate `0.3394`; simulator value `0.3438`; ESS `1,660` |
| Default promotion gate | `HOLD` because evidence remains synthetic |

The numbers show that the contextual learners recover useful structure in this
simulator. They do not establish generalization to a real population.

## Safety constraints

The project can enforce or evaluate:

- blocked actions;
- maximum action-share caps;
- minimum exploration rate;
- minimum action coverage;
- minimum OPE effective sample size;
- minimum replay matched count;
- minimum estimated improvement;
- maximum simulation regret versus the oracle.

The constrained policy wrapper removes blocked actions and redirects choices when
an action-share cap would be exceeded. Promotion checks report passes, failures,
and warnings separately so weak support is visible rather than hidden in one score.

## Promotion decision

The release decision is **HOLD**. The default epsilon-greedy candidate satisfies
the configured numerical checks, but it relies on simulator oracle probabilities
and has no non-synthetic evidence. The fixed candidate also fails concentration,
exploration, coverage, improvement, and regret checks.

A future promotion would require a deployable learned candidate, representative
logged data with reliable propensities, sufficient overlap and effective sample
size, domain-specific capacity constraints, and staged evidence from a controlled
experiment.

## Limitations and risks

- The reward model is linear and intentionally simple.
- Context features are synthetic and do not model real drift, missingness, privacy,
  fairness, or adversarial behavior.
- Rewards are immediate; delayed and censored outcomes are not represented.
- OPE can have high variance when behavior propensities are small or support is
  weak.
- Safety checks are examples, not a substitute for domain risk assessment.
- Local JSONL logging and in-memory counters are not durable production systems.
- No calibration, confidence interval, human review workflow, or rollback
  automation is provided.
