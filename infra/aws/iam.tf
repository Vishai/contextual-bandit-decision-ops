data "aws_iam_policy_document" "ecs_tasks_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# Lets ECS pull the image and ship container logs.
resource "aws_iam_role" "execution" {
  name_prefix        = "${local.name}-exec-"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# What the application itself may do: mount and write the shared EFS volume
# through its access point, and nothing else.
resource "aws_iam_role" "task" {
  name_prefix        = "${local.name}-task-"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

data "aws_iam_policy_document" "efs_access" {
  statement {
    actions = [
      "elasticfilesystem:ClientMount",
      "elasticfilesystem:ClientWrite",
    ]
    resources = [aws_efs_file_system.data.arn]

    condition {
      test     = "StringEquals"
      variable = "elasticfilesystem:AccessPointArn"
      values   = [aws_efs_access_point.data.arn]
    }
  }
}

resource "aws_iam_role_policy" "task_efs" {
  name   = "efs-access"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.efs_access.json
}
