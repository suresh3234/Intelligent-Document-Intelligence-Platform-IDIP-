output "api_repo_url" {
  value       = aws_ecr_repository.api.repository_url
  description = "The repository URL for the API image registry"
}

output "worker_repo_url" {
  value       = aws_ecr_repository.worker.repository_url
  description = "The repository URL for the worker image registry"
}

output "base_repo_url" {
  value       = aws_ecr_repository.base.repository_url
  description = "The repository URL for the shared base image registry"
}

output "api_repo_arn" {
  value       = aws_ecr_repository.api.arn
  description = "The ARN of the ECR repository for the API service"
}

output "worker_repo_arn" {
  value       = aws_ecr_repository.worker.arn
  description = "The ARN of the ECR repository for the worker service"
}

output "base_repo_arn" {
  value       = aws_ecr_repository.base.arn
  description = "The ARN of the ECR repository for the base image"
}
