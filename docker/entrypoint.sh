#!/bin/sh
# Start the offer service with settings from environment variables, which is
# how ECS task definitions and `docker run -e` pass configuration.
set -eu

# Every replica writes its own log folder so shared storage (EFS) never sees two
# processes appending to the same file. Retraining reads the whole parent folder.
replica_id="${HOSTNAME:-$(cat /proc/sys/kernel/random/uuid)}"
log_dir="${LOG_DIR:-/tmp/offer-logs}/${replica_id}"
mkdir -p "${log_dir}"

set -- \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --environment "${SERVICE_ENVIRONMENT:-container}" \
  --decision-log "${log_dir}/decisions.jsonl" \
  --feedback-log "${log_dir}/feedback.jsonl" \
  --exploration-floor "${EXPLORATION_FLOOR:-0.05}" \
  --model-poll-seconds "${MODEL_POLL_SECONDS:-0}"

if [ -n "${MODEL_ARTIFACT:-}" ]; then
  set -- "$@" --model-artifact "${MODEL_ARTIFACT}"
fi
if [ -n "${FEATURE_STORE:-}" ]; then
  set -- "$@" --feature-store "${FEATURE_STORE}"
fi
if [ "${LOG_UNMATCHED_FEEDBACK:-false}" = "true" ]; then
  set -- "$@" --log-unmatched-feedback
fi

exec contextual-bandit-offer-service "$@"
