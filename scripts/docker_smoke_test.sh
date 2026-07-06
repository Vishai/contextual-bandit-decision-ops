#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

docker_command="${DOCKER:-docker}"
python_command="${PYTHON:-python3}"
image_name="${IMAGE_NAME:-contextual-bandit-decision-ops:local}"
container_name="${CONTAINER_NAME:-contextual-bandit-smoke-$$}"
host_port="${HOST_PORT:-18000}"

if ! command -v "${docker_command}" >/dev/null 2>&1; then
  echo "Docker is unavailable; skipping the container smoke test." >&2
  exit 2
fi
if ! "${docker_command}" info >/dev/null 2>&1; then
  echo "Docker is installed but its daemon is unavailable; skipping the container smoke test." >&2
  exit 2
fi
if ! command -v "${python_command}" >/dev/null 2>&1; then
  echo "Python is required to check the container health endpoint." >&2
  exit 2
fi

cleanup() {
  "${docker_command}" rm --force "${container_name}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

"${docker_command}" build --tag "${image_name}" .
"${docker_command}" run \
  --detach \
  --rm \
  --name "${container_name}" \
  --publish "127.0.0.1:${host_port}:8000" \
  "${image_name}" >/dev/null

health_url="http://127.0.0.1:${host_port}/health"
for ((attempt = 1; attempt <= 30; attempt += 1)); do
  if "${python_command}" -c \
    "import json, sys, urllib.request; payload = json.load(urllib.request.urlopen(sys.argv[1], timeout=2)); assert payload['status'] == 'ok'" \
    "${health_url}" >/dev/null 2>&1; then
    echo "Docker smoke test passed: ${health_url}"
    exit 0
  fi
  sleep 1
done

echo "Container did not become healthy." >&2
"${docker_command}" logs "${container_name}" >&2 || true
exit 1
