variable "vpc_id" {
  type        = string
  description = "The VPC ID"
}

variable "subnet_ids" {
  type        = list(string)
  description = "List of private subnet IDs for RDS"
}

variable "vpc_cidr" {
  type        = string
  description = "The CIDR block of the VPC"
}

variable "environment" {
  type        = string
  description = "Deployment environment"
}

variable "db_name" {
  type        = string
  default     = "idip_metadata"
  description = "Database name"
}

variable "db_user" {
  type        = string
  default     = "idip_admin"
  description = "Database administrator username"
}

variable "db_password" {
  type        = string
  sensitive   = true
  description = "Database password"
}

variable "instance_class" {
  type        = string
  default     = "db.t4g.large"
  description = "RDS instance class"
}

variable "allocated_storage" {
  type        = number
  default     = 20
  description = "Allocated storage size in gigabytes"
}

variable "multi_az" {
  type        = bool
  default     = true
  description = "Enable Multi-AZ deployment"
}
