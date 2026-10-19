#!/usr/bin/env bash
# Regenerate every offer recommendation report. Takes a few minutes, since the
# backtest and scenario runs simulate tens of thousands of decisions.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

required_commands=(
  contextual-bandit-features
  contextual-bandit-offers
  contextual-bandit-backtest
  contextual-bandit-scenarios
  contextual-bandit-latency
  contextual-bandit-retrain
  contextual-bandit-monitor
)
for command_name in "${required_commands[@]}"; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing ${command_name}; run 'make install' first." >&2
    exit 1
  fi
done

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/offer-reports.XXXXXX")"
trap 'rm -rf "${work_dir}"' EXIT

contextual-bandit-features
contextual-bandit-offers
contextual-bandit-backtest
contextual-bandit-scenarios
contextual-bandit-latency
contextual-bandit-retrain \
  --store "${work_dir}/store.sqlite" \
  --bootstrap-demo-store \
  --model-dir "${work_dir}/models"
contextual-bandit-monitor

echo "Regenerated offer reports:"
for report in \
  feature_pipeline_summary \
  offer_policy_comparison \
  offer_backtest \
  offer_scenarios \
  latency_benchmark \
  retraining_run \
  offer_monitoring_report; do
  echo "  reports/${report}.md"
done
