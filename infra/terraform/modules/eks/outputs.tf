output "cluster_name" {
  value       = aws_eks_cluster.this.name
  description = "EKS Cluster Name"
}

output "cluster_endpoint" {
  value       = aws_eks_cluster.this.endpoint
  description = "Endpoint for EKS Control Plane"
}

output "cluster_certificate_authority_data" {
  value       = aws_eks_cluster.this.certificate_authority[0].data
  description = "EKS Control Plane certificate authority data"
}

output "cluster_id" {
  value       = aws_eks_cluster.this.id
  description = "EKS Cluster ID"
}
