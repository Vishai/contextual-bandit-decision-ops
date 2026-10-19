# One encrypted EFS volume holds everything the service and retraining share:
# the SQLite feature store, every replica's decision and feedback logs, and the
# model registry (versions/, latest.json, registry.jsonl).
resource "aws_efs_file_system" "data" {
  creation_token = "${local.name}-data"
  encrypted      = true

  lifecycle_policy {
    transition_to_ia = "AFTER_30_DAYS"
  }

  tags = {
    Name = "${local.name}-data"
  }
}

resource "aws_efs_mount_target" "data" {
  for_each = toset(var.subnet_ids)

  file_system_id  = aws_efs_file_system.data.id
  subnet_id       = each.value
  security_groups = [aws_security_group.efs.id]
}

resource "aws_efs_access_point" "data" {
  file_system_id = aws_efs_file_system.data.id

  posix_user {
    uid = 1000
    gid = 1000
  }

  root_directory {
    path = "/${local.name}"

    creation_info {
      owner_uid   = 1000
      owner_gid   = 1000
      permissions = "0755"
    }
  }
}
