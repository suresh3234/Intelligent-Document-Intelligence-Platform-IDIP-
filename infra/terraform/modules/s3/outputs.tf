output "raw_bucket_name" {
  value       = aws_s3_bucket.raw.id
  description = "Name of raw data S3 bucket"
}

output "processed_bucket_name" {
  value       = aws_s3_bucket.processed.id
  description = "Name of processed data S3 bucket"
}

output "models_bucket_name" {
  value       = aws_s3_bucket.models.id
  description = "Name of model artifacts S3 bucket"
}

output "dvc_bucket_name" {
  value       = aws_s3_bucket.dvc.id
  description = "Name of DVC store S3 bucket"
}

output "raw_bucket_arn" {
  value       = aws_s3_bucket.raw.arn
  description = "ARN of raw data S3 bucket"
}

output "processed_bucket_arn" {
  value       = aws_s3_bucket.processed.arn
  description = "ARN of processed data S3 bucket"
}

output "models_bucket_arn" {
  value       = aws_s3_bucket.models.arn
  description = "ARN of model artifacts S3 bucket"
}

output "dvc_bucket_arn" {
  value       = aws_s3_bucket.dvc.arn
  description = "ARN of DVC store S3 bucket"
}
