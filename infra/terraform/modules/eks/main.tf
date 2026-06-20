# EKS Cluster and Node Groups Configuration
resource "aws_eks_cluster" "this" {
  name     = var.cluster_name
  role_arn = var.cluster_role_arn

  vpc_config {
    subnet_ids              = var.subnet_ids
    endpoint_private_access = true
    endpoint_public_access  = true
  }

  tags = {
    Name        = var.cluster_name
    Environment = var.environment
  }
}

resource "aws_eks_node_group" "cpu" {
  cluster_name    = aws_eks_cluster.this.name
  node_group_name = "cpu-node-group"
  node_role_arn   = var.node_role_arn
  subnet_ids      = var.subnet_ids
  instance_types  = var.cpu_instance_types

  scaling_config {
    desired_size = var.cpu_desired_size
    max_size     = var.cpu_max_size
    min_size     = var.cpu_min_size
  }

  update_config {
    max_unavailable = 1
  }

  labels = {
    role        = "general"
    Environment = var.environment
  }

  tags = {
    Name        = "idip-${var.environment}-cpu-node"
    Environment = var.environment
  }
}

resource "aws_eks_node_group" "gpu" {
  cluster_name    = aws_eks_cluster.this.name
  node_group_name = "gpu-node-group"
  node_role_arn   = var.node_role_arn
  subnet_ids      = var.subnet_ids
  instance_types  = var.gpu_instance_types

  scaling_config {
    desired_size = var.gpu_desired_size
    max_size     = var.gpu_max_size
    min_size     = var.gpu_min_size
  }

  update_config {
    max_unavailable = 1
  }

  labels = {
    role        = "gpu"
    accelerator = "nvidia-gpu"
    Environment = var.environment
  }

  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  tags = {
    Name        = "idip-${var.environment}-gpu-node"
    Environment = var.environment
  }
}
