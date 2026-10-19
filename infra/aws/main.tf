locals {
  name           = var.project
  container_name = "offer-service"
  container_port = 8000
  data_path      = "/data"
  image          = "${aws_ecr_repository.service.repository_url}:${var.image_tag}"
  https_enabled  = var.certificate_arn != ""

  # Shared by the service and the retraining job, both on the EFS volume.
  shared_environment = [
    { name = "FEATURE_STORE_JOURNAL_MODE", value = "DELETE" },
  ]
}

resource "aws_ecr_repository" "service" {
  name                 = local.name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "service" {
  repository = aws_ecr_repository.service.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the 20 most recent images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 20
      }
      action = { type = "expire" }
    }]
  })
}

resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${local.name}"
  retention_in_days = var.log_retention_days
}
