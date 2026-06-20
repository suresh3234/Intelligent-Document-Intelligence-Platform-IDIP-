variable "environment" {
  type        = string
  description = "Deployment environment name"
}

variable "raw_bucket_arn" {
  type        = string
  description = "ARN of raw bucket"
}

variable "processed_bucket_arn" {
  type        = string
  description = "ARN of processed bucket"
}

variable "models_bucket_arn" {
  type        = string
  description = "ARN of models bucket"
}

variable "dvc_bucket_arn" {
  type        = string
  description = "ARN of DVC bucket"
}
