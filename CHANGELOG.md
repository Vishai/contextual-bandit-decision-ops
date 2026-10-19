# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-10-19

### Added

- Offer catalog and an offer context with user history, device, geography,
  engagement and purchase features, encoded into 18 model features.
- SQLite feature store with point-in-time SQL feature pipelines and daily
  snapshots, plus a synthetic population of users, sessions and purchases.
- Bayesian Thompson Sampling: a contextual Bayesian logistic model per offer
  with online Laplace updates, a batch refit, forgetting and Monte Carlo
  propensities, plus a Beta-Bernoulli variant.
- Random, rule-based, greedy and UCB offer policies, and an online simulator
  that reports conversion, revenue per user, regret, allocation and latency.
- Historical logging, offline policy evaluation (IPS, SNIPS, doubly robust)
  with bootstrap intervals, replay, and warm-started forward simulation.
- Cold start, delayed reward and shifting conversion rate scenarios, with
  delay and forgetting sweeps.
- FastAPI offer service with online posterior updates, feature store lookups,
  cold-start contexts, an exploration floor, latency tracking, model hot
  reload and polling, and handling for feedback that reaches another replica.
- Latency benchmark against the 100 ms target, in process or against a URL.
- Automated retraining with log ingestion, a do-no-harm promotion gate, a
  versioned model registry and a run schedule.
- Monitoring of regret, offer allocation, latency, data drift (PSI and total
  variation), outcomes and propensities from service logs.
- Docker entrypoint configured by environment variables, and Terraform for
  ECR, ECS Fargate, ALB, EFS, EventBridge retraining and CloudWatch alarms.
- Offer recommendation design doc and a script that regenerates every report.

### Changed

- The Docker image now runs the offer service.
- Offline evaluation works without simulator truth and takes any baseline.

## [0.1.0] - 2026-07-06

### Added

- Deterministic contextual bandit simulator, typed event schema, CSV log, and
  synthetic-data report.
- Random, fixed, epsilon-greedy, and greedy-oracle baseline comparisons.
- LinUCB and Linear Thompson Sampling with sequential learning evaluation.
- Replay, IPS, SNIPS, and doubly robust off-policy estimators.
- Safe policy wrapper and conservative promotion gates.
- Local/staging FastAPI decision and feedback service.
- Local observability checks for actions, rewards, features, propensities,
  feedback coverage, exploration, and service counters.
- CPU-only Docker image, GitHub Actions quality checks, smoke scripts, and
  deterministic demo generation.
- Architecture, policy card, evaluation methodology, operations, release
  checklist, interview notes, and portfolio release summary.
