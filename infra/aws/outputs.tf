output "region" {
  value = var.region
}

output "service_url" {
  description = "Base URL of the offer service."
  value       = "${local.https_enabled ? "https" : "http"}://${aws_lb.service.dns_name}"
}

output "ecr_repository_url" {
  value = aws_ecr_repository.service.repository_url
}

output "cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "service_name" {
  value = aws_ecs_service.service.name
}

output "retrain_task_definition" {
  value = aws_ecs_task_definition.retrain.family
}

output "task_subnets" {
  value = join(",", var.subnet_ids)
}

output "task_security_group" {
  value = aws_security_group.tasks.id
}

output "efs_file_system_id" {
  value = aws_efs_file_system.data.id
}
