variable "aws_region" {
  type        = string
  default     = "us-west-2"
  description = "AWS region to deploy resources into"
}

variable "environment" {
  type        = string
  default     = "production"
  description = "Target deployment environment"
}

# VPC variables
variable "vpc_cidr" {
  type        = string
  default     = "10.0.0.0/16"
  description = "CIDR block for VPC"
}

variable "public_subnets" {
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
  description = "Public subnets list"
}

variable "private_subnets" {
  type        = list(string)
  default     = ["10.0.10.0/24", "10.0.11.0/24"]
  description = "Private subnets list"
}

variable "availability_zones" {
  type        = list(string)
  default     = ["us-west-2a", "us-west-2b"]
  description = "Availability zones list"
}

variable "enable_nat_gateway" {
  type        = bool
  default     = true
  description = "Enable NAT gateway for private subnets to pull internet traffic"
}

# EKS worker node group configuration
variable "cpu_instance_types" {
  type        = list(string)
  default     = ["m5.large"]
  description = "EC2 instances for general CPU nodes"
}

variable "gpu_instance_types" {
  type        = list(string)
  default     = ["g4dn.xlarge"]
  description = "EC2 instances for GPU model inference worker nodes"
}

variable "cpu_desired_size" {
  type    = number
  default = 3
}

variable "cpu_min_size" {
  type    = number
  default = 2
}

variable "cpu_max_size" {
  type    = number
  default = 5
}

variable "gpu_desired_size" {
  type    = number
  default = 2
}

variable "gpu_min_size" {
  type    = number
  default = 1
}

variable "gpu_max_size" {
  type    = number
  default = 4
}

# RDS database variables
variable "db_name" {
  type        = string
  default     = "idip_metadata"
  description = "PostgreSQL DB name"
}

variable "db_user" {
  type        = string
  default     = "idip_admin"
  description = "RDS master username"
}

variable "db_password" {
  type        = string
  sensitive   = true
  description = "RDS master password"
}

variable "db_instance_class" {
  type        = string
  default     = "db.t4g.large"
  description = "RDS instance class"
}

variable "db_allocated_storage" {
  type        = number
  default     = 20
  description = "RDS allocated storage size in GB"
}

variable "db_multi_az" {
  type        = bool
  default     = true
  description = "Enable RDS PostgreSQL Multi-AZ"
}

# ElastiCache Redis variables
variable "redis_node_type" {
  type        = string
  default     = "cache.t4g.medium"
  description = "ElastiCache Redis node type"
}

variable "redis_num_cache_clusters" {
  type        = number
  default     = 2
  description = "Number of cache nodes in the replication group"
}
