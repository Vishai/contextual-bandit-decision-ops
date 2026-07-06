# Evaluation Methodology

## Questions answered

The evaluation stack separates three questions:

1. Does the simulator generate valid, reproducible events?
2. Can a policy learn useful context/action structure during sequential simulation?
3. Given fixed logs from a behavior policy, what can be estimated about a different
   target policy?

Keeping these questions separate prevents an oracle-assisted simulation result from
being mistaken for deployable offline evidence.

## Deterministic experimental design

Every command accepts or uses a fixed seed. Context generation, policy randomness,
reward sampling, IDs, and timestamps are reproducible. Policies in a comparison see
the same sequence of contexts and underlying reward randomness where applicable,
which reduces noise in pairwise comparisons.

The default simulator has three actions and context-dependent Bernoulli reward
probabilities. Because the simulator exposes the true expected reward for every
context/action pair, tests can measure regret and compare estimator output with
known synthetic truth.

## Online-style simulation

For each event, the evaluator:

1. observes a context;
2. asks the policy for an action;
3. samples the selected action's reward;
4. updates a learning policy with that observation;
5. records reward, action share, and regret.

Expected instantaneous regret is the oracle action's reward probability minus the
selected action's probability. Cumulative regret is the sum across events. This
uses privileged simulator information only for measurement.

## Baseline and learner interpretation

Random uniform is the exploration floor. Fixed action exposes context-insensitive
behavior. LinUCB adds an uncertainty bonus to per-action linear reward estimates;
Linear Thompson Sampling samples plausible linear parameters. The greedy oracle is
an upper reference under the simulator, not a feasible real policy.

On the default 5,000-event run, LinUCB averages `0.325` reward and Linear Thompson
Sampling `0.313`, compared with `0.278` for random and `0.337` for the oracle.
These values are meaningful only for this seeded environment and configuration.

## Off-policy estimators

Let `a_i` and `r_i` be the logged action and reward, `μ(a_i|x_i)` the logged
behavior propensity, and `π(a_i|x_i)` the target-policy probability.

- **Direct logged average** is the raw mean of observed rewards. It describes the
  behavior policy, not an arbitrary target.
- **Replay/matching** averages rewards among rows where a sampled or deterministic
  target action matches the logged action. Its matched count exposes data loss.
- **IPS** uses
  `n⁻¹ Σ [π(a_i|x_i) / μ(a_i|x_i)] r_i`.
- **SNIPS** divides the weighted reward sum by the sum of importance weights.
- **Doubly robust (DR)** combines a fitted reward-model prediction for all target
  actions with an importance-weighted correction on the logged action.

Effective sample size is `(Σw)² / Σw²`. It falls when a few large weights dominate.
Rows with zero target probability receive zero weight safely. A non-positive logged
propensity is invalid because the importance ratio is undefined.

## Reward model

The DR implementation fits a small linear reward model using context and action
features. Cross-fitting is outside this milestone, so its estimates can still
reflect model-fit bias. DR is useful here because the simulator makes its two
sources of information inspectable: model predictions supply a baseline and
importance weighting corrects observed residuals.

## Reliability checks

Evaluation output should be read alongside:

- action coverage and replay matched counts;
- importance-weight effective sample size;
- minimum and maximum propensity;
- action concentration and exploration rate;
- the gap between estimates and simulator truth;
- seed and event count.

IPS and SNIPS become unstable when behavior propensities are very small, target and
behavior policies have poor overlap, or a few samples carry most of the weight.
SNIPS controls scale but does not create missing support. DR reduces dependence on
either weighting or modeling alone, but it is not immune to both being wrong.

## What is not established

The reports do not provide confidence intervals, causal identification for
observational production data, long-term effects, fairness analysis, or evidence
that the synthetic reward function matches a real domain. A real launch would need
pre-registered metrics, representative logs, overlap analysis, uncertainty
quantification, and a controlled staged experiment.
