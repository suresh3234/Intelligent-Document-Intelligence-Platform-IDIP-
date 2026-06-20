variable "vpc_id" {
  type        = string
  description = "The VPC ID"
}

variable "subnet_ids" {
  type        = list(string)
  description = "List of private subnet IDs for Redis"
}

variable "vpc_cidr" {
  type        = string
  description = "The CIDR block of the VPC"
}

variable "environment" {
  type        = string
  description = "Deployment environment"
}

variable "node_type" {
  type        = string
  default     = "cache.t4g.medium"
  description = "Elasticache node instance type"
}

variable "num_cache_clusters" {
  type        = number
  default     = 2
  description = "Number of cache clusters in the replication group"
}
