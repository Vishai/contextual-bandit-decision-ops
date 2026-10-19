variable "region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Name prefix for every resource."
  type        = string
  default     = "offer-bandit"
}

variable "vpc_id" {
  description = "VPC for the load balancer, service tasks and EFS."
  type        = string
}

variable "subnet_ids" {
  description = "Public subnets in at least two availability zones."
  type        = list(string)

  validation {
    condition     = length(var.subnet_ids) >= 2
    error_message = "The load balancer needs subnets in at least two availability zones."
  }
}

variable "allowed_cidrs" {
  description = "CIDR ranges allowed to reach the load balancer."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "certificate_arn" {
  description = "ACM certificate for HTTPS. Leave empty to serve plain HTTP."
  type        = string
  default     = ""
}

variable "image_tag" {
  description = "Image tag in the ECR repository to deploy."
  type        = string
  default     = "latest"
}

variable "service_cpu" {
  description = "Fargate CPU units for each service task."
  type        = number
  default     = 512
}

variable "service_memory" {
  description = "Fargate memory (MiB) for each service task."
  type        = number
  default     = 1024
}

variable "desired_count" {
  description = "Service tasks to start with."
  type        = number
  default     = 2
}

variable "min_count" {
  description = "Fewest service tasks autoscaling may run."
  type        = number
  default     = 2
}

variable "max_count" {
  description = "Most service tasks autoscaling may run."
  type        = number
  default     = 6
}

variable "exploration_floor" {
  description = "Share of decisions that pick a uniformly random offer."
  type        = number
  default     = 0.05
}

variable "model_poll_seconds" {
  description = "How often each service task checks for a newly promoted model."
  type        = number
  default     = 60
}

variable "retrain_schedule" {
  description = "EventBridge schedule for the retraining task."
  type        = string
  default     = "cron(0 3 * * ? *)"
}

variable "retrain_cpu" {
  description = "Fargate CPU units for the retraining task."
  type        = number
  default     = 1024
}

variable "retrain_memory" {
  description = "Fargate memory (MiB) for the retraining task."
  type        = number
  default     = 2048
}

variable "bootstrap_demo_history" {
  description = "Seed an empty feature store with synthetic rule-based history on the first retrain."
  type        = bool
  default     = true
}

variable "latency_alarm_seconds" {
  description = "Alarm when the load balancer's p99 target response time goes above this."
  type        = number
  default     = 0.1
}

variable "alarm_email" {
  description = "Email address for alarm notifications. Leave empty to skip the subscription."
  type        = string
  default     = ""
}

variable "log_retention_days" {
  description = "CloudWatch log retention."
  type        = number
  default     = 30
}
