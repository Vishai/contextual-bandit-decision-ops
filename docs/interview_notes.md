# Interview Notes

## Two-minute explanation

This project is an end-to-end contextual bandit reference built around one rule:
better estimated reward is not enough to launch. It starts with a deterministic
three-action simulator, compares simple baselines, learns with LinUCB and Linear
Thompson Sampling, evaluates target policies from logged behavior-policy data, and
applies explicit promotion constraints. The selected policy can then be exercised
through a local FastAPI service and checked with transparent drift metrics.

The default release still says **HOLD**. Its strongest evidence comes from a
synthetic reward function, and one high-performing baseline has privileged oracle
information. That conservative boundary is part of the design, not an unfinished
claim.

## Design choices worth discussing

- **Determinism:** seeded contexts, actions, rewards, IDs, and timestamps make
  regressions and report diffs easy to audit.
- **Common interfaces:** simple `choose_action` and `update` behavior keeps
  baselines and learners compatible without a deep class hierarchy.
- **Known propensities:** logs record behavior probabilities so IPS/SNIPS/DR are
  meaningful rather than reconstructed after the fact.
- **Transparent safety:** promotion checks return individual passes, failures, and
  warnings instead of compressing evidence into one opaque score.
- **Deliberately small serving layer:** local JSONL and process counters demonstrate
  the contract while making production gaps obvious.

## Algorithm intuition

LinUCB fits a linear reward model for each action and adds a confidence bonus. It
prefers actions that look rewarding or remain uncertain. Thompson Sampling draws a
plausible parameter vector from each action's posterior approximation and chooses
the best sample, producing exploration through uncertainty.

The simulator oracle knows every action's true reward probability. It is useful for
measuring regret and checking whether learners discover structure, but it cannot be
served in a real environment.

## OPE talking points

Naively averaging logged rewards evaluates the behavior policy. Replay uses only
matching actions and can discard much of the data. IPS corrects action-selection
bias through target-to-behavior probability ratios but becomes noisy with weak
overlap. SNIPS normalizes weights. Doubly robust estimation combines a reward model
with a weighted residual correction; it is more resilient, not magical. ESS and
matched counts are therefore first-class outputs.

## Results to remember

- In 5,000 sequential synthetic events, LinUCB averages `0.325` reward versus
  `0.278` for random and `0.337` for the oracle.
- Linear Thompson Sampling averages `0.313`.
- The default OPE oracle DR estimate is `0.3394` versus simulator value `0.3438`,
  with ESS `1,660`.
- The promotion result is `HOLD` despite strong synthetic performance.
- The monitoring fixture raises six warnings and no hard failure, demonstrating
  graded operational signals.

## Five-minute reviewer path

```bash
make release-check
make demo
contextual-bandit-info
```

Then read the policy promotion report, policy card, and contextual learning report.
Use `make smoke` to exercise the API without starting an external server.

## Likely extension questions

**How would this move toward production?** Replace synthetic action semantics and
reward assumptions with a reviewed domain contract; add durable event joins,
privacy controls, calibrated uncertainty, representative OPE, staged experiments,
rollback rules, authentication, and production telemetry.

**Why not choose the highest synthetic performer?** Oracle-assisted performance can
leak unavailable information, and reward alone ignores support, capacity,
exploration, reliability, and operational constraints.

**What would you improve statistically?** Cross-fitted reward models, confidence
intervals or bootstrap uncertainty, clipping sensitivity, richer overlap
diagnostics, delayed-feedback handling, and evaluation across multiple seeds and
shift scenarios.

**What is the biggest modeling limitation?** A simple linear Bernoulli environment
makes the learners easy to inspect but understates misspecification, non-stationary
behavior, and real-world outcome complexity.
