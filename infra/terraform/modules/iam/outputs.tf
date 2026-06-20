output "cluster_role_arn" {
  value       = aws_iam_role.cluster.arn
  description = "EKS control plane cluster role ARN"
}

output "node_role_arn" {
  value       = aws_iam_role.nodes.arn
  description = "EKS worker node role ARN"
}

output "s3_access_policy_arn" {
  value       = aws_iam_policy.s3_access.arn
  description = "Scoped S3 access policy ARN"
}
