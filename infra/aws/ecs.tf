resource "aws_ecs_cluster" "main" {
  name = local.name

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

locals {
  efs_volume = {
    name = "data"
    efs = {
      file_system_id  = aws_efs_file_system.data.id
      access_point_id = aws_efs_access_point.data.id
    }
  }

  log_configuration = {
    logDriver = "awslogs"
    options = {
      awslogs-group         = aws_cloudwatch_log_group.ecs.name
      awslogs-region        = var.region
      awslogs-stream-prefix = "ecs"
    }
  }

  retrain_command = concat(
    [
      "contextual-bandit-retrain",
      "--store", "${local.data_path}/feature_store.sqlite",
      "--model-dir", "${local.data_path}/models",
      "--decision-log", "${local.data_path}/logs",
      "--feedback-log", "${local.data_path}/logs",
      "--report-md", "${local.data_path}/reports/retraining_run.md",
    ],
    var.bootstrap_demo_history ? ["--bootstrap-demo-store"] : [],
  )
}

resource "aws_ecs_task_definition" "service" {
  family                   = "${local.name}-service"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.service_cpu
  memory                   = var.service_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  volume {
    name = local.efs_volume.name

    efs_volume_configuration {
      file_system_id     = local.efs_volume.efs.file_system_id
      transit_encryption = "ENABLED"

      authorization_config {
        access_point_id = local.efs_volume.efs.access_point_id
        iam             = "ENABLED"
      }
    }
  }

  container_definitions = jsonencode([
    {
      name      = local.container_name
      image     = local.image
      essential = true
      portMappings = [{
        containerPort = local.container_port
        protocol      = "tcp"
      }]
      environment = concat(local.shared_environment, [
        { name = "PORT", value = tostring(local.container_port) },
        { name = "SERVICE_ENVIRONMENT", value = "aws" },
        { name = "FEATURE_STORE", value = "${local.data_path}/feature_store.sqlite" },
        { name = "MODEL_ARTIFACT", value = "${local.data_path}/models/latest.json" },
        { name = "LOG_DIR", value = "${local.data_path}/logs" },
        { name = "MODEL_POLL_SECONDS", value = tostring(var.model_poll_seconds) },
        { name = "EXPLORATION_FLOOR", value = tostring(var.exploration_floor) },
        { name = "LOG_UNMATCHED_FEEDBACK", value = "true" },
      ])
      mountPoints = [{
        sourceVolume  = local.efs_volume.name
        containerPath = local.data_path
        readOnly      = false
      }]
      healthCheck = {
        command = [
          "CMD-SHELL",
          "python -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:${local.container_port}/health', timeout=2)\"",
        ]
        interval    = 15
        timeout     = 5
        retries     = 3
        startPeriod = 20
      }
      logConfiguration = local.log_configuration
    }
  ])
}

resource "aws_ecs_task_definition" "retrain" {
  family                   = "${local.name}-retrain"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.retrain_cpu
  memory                   = var.retrain_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  volume {
    name = local.efs_volume.name

    efs_volume_configuration {
      file_system_id     = local.efs_volume.efs.file_system_id
      transit_encryption = "ENABLED"

      authorization_config {
        access_point_id = local.efs_volume.efs.access_point_id
        iam             = "ENABLED"
      }
    }
  }

  container_definitions = jsonencode([
    {
      name        = "retrain"
      image       = local.image
      essential   = true
      command     = local.retrain_command
      environment = local.shared_environment
      mountPoints = [{
        sourceVolume  = local.efs_volume.name
        containerPath = local.data_path
        readOnly      = false
      }]
      logConfiguration = local.log_configuration
    }
  ])
}

resource "aws_ecs_service" "service" {
  name                              = "${local.name}-service"
  cluster                           = aws_ecs_cluster.main.id
  task_definition                   = aws_ecs_task_definition.service.arn
  desired_count                     = var.desired_count
  launch_type                       = "FARGATE"
  health_check_grace_period_seconds = 30

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [aws_security_group.tasks.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.service.arn
    container_name   = local.container_name
    container_port   = local.container_port
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  lifecycle {
    ignore_changes = [desired_count]
  }

  depends_on = [aws_lb_listener.http, aws_efs_mount_target.data]
}

resource "aws_appautoscaling_target" "service" {
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.service.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  min_capacity       = var.min_count
  max_capacity       = var.max_count
}

resource "aws_appautoscaling_policy" "cpu" {
  name               = "${local.name}-cpu"
  policy_type        = "TargetTrackingScaling"
  service_namespace  = aws_appautoscaling_target.service.service_namespace
  resource_id        = aws_appautoscaling_target.service.resource_id
  scalable_dimension = aws_appautoscaling_target.service.scalable_dimension

  target_tracking_scaling_policy_configuration {
    target_value       = 60
    scale_in_cooldown  = 120
    scale_out_cooldown = 60

    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
  }
}
