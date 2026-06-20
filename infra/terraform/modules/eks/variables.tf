variable "cluster_name" {
  type        = string
  description = "The name of the EKS cluster"
}

variable "cluster_role_arn" {
  type        = string
  description = "IAM role ARN for the EKS cluster plane"
}

variable "node_role_arn" {
  type        = string
  description = "IAM role ARN for EKS worker nodes"
}

variable "subnet_ids" {
  type        = list(string)
  description = "Subnet IDs for EKS node groups"
}

variable "environment" {
  type        = string
  description = "Environment name"
}

variable "cpu_instance_types" {
  type        = list(string)
  default     = ["m5.large"]
  description = "EC2 instance types for CPU node group"
}

variable "gpu_instance_types" {
  type        = list(string)
  default     = ["g4dn.xlarge"]
  description = "EC2 instance types for GPU node group"
}

variable "cpu_desired_size" {
  type        = number
  default     = 3
  description = "Desired number of CPU worker nodes"
}

variable "cpu_min_size" {
  type        = number
  default     = 2
  description = "Minimum number of CPU worker nodes"
}

variable "cpu_max_size" {
  type        = number
  default     = 5
  description = "Maximum number of CPU worker nodes"
}

variable "gpu_desired_size" {
  type        = number
  default     = 2
  description = "Desired number of GPU worker nodes"
}

variable "gpu_min_size" {
  type        = number
  default     = 1
  description = "Minimum number of GPU worker nodes"
}

variable "gpu_max_size" {
  type        = number
  default     = 4
  description = "Maximum number of GPU worker nodes"
}
