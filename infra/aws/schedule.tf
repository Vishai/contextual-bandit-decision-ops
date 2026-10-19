# Automated retraining: EventBridge starts the retrain task on a schedule. It
# ingests every replica's logs from EFS, refits the posterior, runs the
# promotion gate and, if the candidate passes, writes a new latest.json that the
# service tasks pick up on their next poll.
resource "aws_cloudwatch_event_rule" "retrain" {
  name                = "${local.name}-retrain"
  description         = "Retrain the offer posterior from logged feedback"
  schedule_expression = var.retrain_schedule
}

resource "aws_cloudwatch_event_target" "retrain" {
  rule     = aws_cloudwatch_event_rule.retrain.name
  arn      = aws_ecs_cluster.main.arn
  role_arn = aws_iam_role.scheduler.arn

  ecs_target {
    task_definition_arn = aws_ecs_task_definition.retrain.arn
    launch_type         = "FARGATE"
    task_count          = 1

    network_configuration {
      subnets          = var.subnet_ids
      security_groups  = [aws_security_group.tasks.id]
      assign_public_ip = true
    }
  }
}

# Lets EventBridge start the scheduled retraining task.
data "aws_iam_policy_document" "events_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "scheduler" {
  name_prefix        = "${local.name}-sched-"
  assume_role_policy = data.aws_iam_policy_document.events_assume.json
}

data "aws_iam_policy_document" "run_retrain" {
  statement {
    actions   = ["ecs:RunTask"]
    resources = ["${aws_ecs_task_definition.retrain.arn_without_revision}:*"]

    condition {
      test     = "ArnEquals"
      variable = "ecs:cluster"
      values   = [aws_ecs_cluster.main.arn]
    }
  }

  statement {
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.execution.arn, aws_iam_role.task.arn]
  }
}

resource "aws_iam_role_policy" "scheduler" {
  name   = "run-retraining"
  role   = aws_iam_role.scheduler.id
  policy = data.aws_iam_policy_document.run_retrain.json
}
