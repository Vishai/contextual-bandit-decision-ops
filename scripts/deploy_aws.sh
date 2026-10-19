#!/usr/bin/env bash
# Build the image, push it to ECR and roll it out to ECS with Terraform.
#
#   scripts/deploy_aws.sh [image-tag]
#
# The tag defaults to the current commit. Set AUTO_APPROVE=1 to skip the
# Terraform confirmation prompt on the main apply.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
infra_dir="${repo_root}/infra/aws"

for tool in aws docker terraform curl; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "${tool} is required to deploy." >&2
    exit 2
  fi
done
if [ ! -f "${infra_dir}/terraform.tfvars" ]; then
  echo "Copy infra/aws/terraform.tfvars.example to infra/aws/terraform.tfvars first." >&2
  exit 2
fi

tag="${1:-$(git -C "${repo_root}" rev-parse --short HEAD)}"
approve=()
if [ "${AUTO_APPROVE:-0}" = "1" ]; then
  approve=(-auto-approve)
fi

terraform -chdir="${infra_dir}" init -input=false

# The repository has to exist before there is anywhere to push the image.
terraform -chdir="${infra_dir}" apply -input=false -auto-approve \
  -target=aws_ecr_repository.service \
  -target=aws_ecr_lifecycle_policy.service

repository_url="$(terraform -chdir="${infra_dir}" output -raw ecr_repository_url)"
registry="${repository_url%%/*}"
region="$(echo "${registry}" | cut -d. -f4)"

aws ecr get-login-password --region "${region}" \
  | docker login --username AWS --password-stdin "${registry}"
docker build --platform linux/amd64 --tag "${repository_url}:${tag}" "${repo_root}"
docker push "${repository_url}:${tag}"

terraform -chdir="${infra_dir}" apply -input=false ${approve[@]+"${approve[@]}"} \
  -var "image_tag=${tag}"

service_url="$(terraform -chdir="${infra_dir}" output -raw service_url)"
echo "Waiting for ${service_url}/health"
for ((attempt = 1; attempt <= 60; attempt += 1)); do
  if curl -fsS "${service_url}/health" >/dev/null 2>&1; then
    echo "Service is healthy at ${service_url}"
    if command -v contextual-bandit-latency >/dev/null 2>&1; then
      contextual-bandit-latency --url "${service_url}" --requests 300 \
        --report-md "${repo_root}/reports/latency_benchmark_aws.md" \
        --artifact-json "${repo_root}/artifacts/latency_benchmark_aws.json" || true
    fi
    exit 0
  fi
  sleep 10
done

echo "The service did not become healthy. Check the ECS service events and CloudWatch logs." >&2
exit 1
