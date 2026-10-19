#!/usr/bin/env bash
# Start the retraining task right away instead of waiting for the schedule.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
infra_dir="${repo_root}/infra/aws"

output() {
  terraform -chdir="${infra_dir}" output -raw "$1"
}

cluster="$(output cluster_name)"
task_definition="$(output retrain_task_definition)"
subnets="$(output task_subnets)"
security_group="$(output task_security_group)"
region="$(output region)"

task_arn="$(aws ecs run-task \
  --region "${region}" \
  --cluster "${cluster}" \
  --task-definition "${task_definition}" \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[${subnets}],securityGroups=[${security_group}],assignPublicIp=ENABLED}" \
  --query 'tasks[0].taskArn' \
  --output text)"

echo "Started ${task_arn}"
echo "Follow it with: aws logs tail /ecs/$(terraform -chdir="${infra_dir}" output -raw cluster_name) --follow --region ${region}"
