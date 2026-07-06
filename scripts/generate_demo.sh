#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

temporary_artifacts="$(mktemp -d "${TMPDIR:-/tmp}/contextual-bandit-demo.XXXXXX")"
trap 'rm -rf "${temporary_artifacts}"' EXIT

required_commands=(
  contextual-bandit-generate
  contextual-bandit-compare
  contextual-bandit-learn
  contextual-bandit-ope
  contextual-bandit-gate
  contextual-bandit-observe
)

for command_name in "${required_commands[@]}"; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing ${command_name}; run 'make install' first." >&2
    exit 1
  fi
done

contextual-bandit-generate
contextual-bandit-compare \
  --artifact-json "${temporary_artifacts}/baseline-policy-comparison.json"
contextual-bandit-learn \
  --artifact-json "${temporary_artifacts}/contextual-learning-comparison.json"
contextual-bandit-ope \
  --artifact-json "${temporary_artifacts}/off-policy-evaluation.json"
contextual-bandit-gate \
  --artifact-json "${temporary_artifacts}/policy-promotion-gate.json"
contextual-bandit-observe \
  --artifact-json "${temporary_artifacts}/staging-observability.json"

echo "Regenerated deterministic reports:"
echo "  reports/synthetic_bandit_log_summary.md"
echo "  reports/baseline_policy_comparison.md"
echo "  reports/contextual_learning_policy_comparison.md"
echo "  reports/off_policy_evaluation.md"
echo "  reports/policy_promotion_gate.md"
echo "  reports/staging_observability_report.md"
