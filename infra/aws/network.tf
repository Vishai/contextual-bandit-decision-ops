resource "aws_security_group" "alb" {
  name_prefix = "${local.name}-alb-"
  description = "Public traffic to the offer service load balancer"
  vpc_id      = var.vpc_id

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidrs
  }

  dynamic "ingress" {
    for_each = local.https_enabled ? [443] : []
    content {
      description = "HTTPS"
      from_port   = ingress.value
      to_port     = ingress.value
      protocol    = "tcp"
      cidr_blocks = var.allowed_cidrs
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_security_group" "tasks" {
  name_prefix = "${local.name}-tasks-"
  description = "Offer service and retraining tasks"
  vpc_id      = var.vpc_id

  ingress {
    description     = "Service port from the load balancer"
    from_port       = local.container_port
    to_port         = local.container_port
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    description = "ECR, CloudWatch and EFS"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_security_group" "efs" {
  name_prefix = "${local.name}-efs-"
  description = "NFS from the offer service and retraining tasks"
  vpc_id      = var.vpc_id

  ingress {
    description     = "NFS"
    from_port       = 2049
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [aws_security_group.tasks.id]
  }

  lifecycle {
    create_before_destroy = true
  }
}
