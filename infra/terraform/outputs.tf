# Root outputs
output "vpc_id" {
  value       = module.vpc.vpc_id
  description = "The VPC ID"
}

output "eks_cluster_name" {
  value       = module.eks.cluster_name
  description = "EKS Cluster Name"
}

output "eks_cluster_endpoint" {
  value       = module.eks.cluster_endpoint
  description = "Endpoint URL for EKS control plane"
}

output "rds_db_endpoint" {
  value       = module.rds.db_endpoint
  description = "Connection endpoint address of RDS PostgreSQL instance"
}

output "redis_primary_endpoint" {
  value       = module.elasticache.redis_primary_endpoint
  description = "Primary primary endpoint of ElastiCache Redis"
}

output "ecr_api_url" {
  value       = module.ecr.api_repo_url
  description = "ECR Repository URL for API container image"
}

output "ecr_worker_url" {
  value       = module.ecr.worker_repo_url
  description = "ECR Repository URL for Worker container image"
}

output "ecr_base_url" {
  value       = module.ecr.base_repo_url
  description = "ECR Repository URL for Base container image"
}

output "s3_raw_bucket" {
  value       = module.s3.raw_bucket_name
  description = "Raw document S3 storage bucket"
}

output "s3_processed_bucket" {
  value       = module.s3.processed_bucket_name
  description = "Processed document/features parquet S3 storage bucket"
}

output "s3_models_bucket" {
  value       = module.s3.models_bucket_name
  description = "Model weights storage S3 bucket"
}

output "s3_dvc_bucket" {
  value       = module.s3.dvc_bucket_name
  description = "DVC registry storage S3 bucket"
}
