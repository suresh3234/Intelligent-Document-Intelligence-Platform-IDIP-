variable "vpc_cidr" {
  type        = string
  description = "CIDR block for the VPC"
  default     = "10.0.0.0/16"
}

variable "public_subnets" {
  type        = list(string)
  description = "List of public subnet CIDR blocks"
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "private_subnets" {
  type        = list(string)
  description = "List of private subnet CIDR blocks"
  default     = ["10.0.10.0/24", "10.0.11.0/24"]
}

variable "availability_zones" {
  type        = list(string)
  description = "Availability zones to launch resources in"
  default     = ["us-west-2a", "us-west-2b"]
}

variable "enable_nat_gateway" {
  type        = bool
  description = "Flag to enable NAT gateway"
  default     = true
}

variable "environment" {
  type        = string
  description = "Deployment environment name"
}
