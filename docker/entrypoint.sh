#!/bin/sh
# Start the offer service with settings from environment variables, which is
# how ECS task definitions and `docker run -e` pass configuration.
set -eu

log_dir="${LOG_DIR:-/tmp/offer-logs}"
mkdir -p "${log_dir}"

set -- \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --environment "${SERVICE_ENVIRONMENT:-container}" \
  --decision-log "${log_dir}/decisions.jsonl" \
  --feedback-log "${log_dir}/feedback.jsonl" \
  --exploration-floor "${EXPLORATION_FLOOR:-0.05}"

if [ -n "${MODEL_ARTIFACT:-}" ]; then
  set -- "$@" --model-artifact "${MODEL_ARTIFACT}"
fi
if [ -n "${FEATURE_STORE:-}" ]; then
  set -- "$@" --feature-store "${FEATURE_STORE}"
fi

exec contextual-bandit-offer-service "$@"
