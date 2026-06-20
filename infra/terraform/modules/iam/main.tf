# IAM Roles and Policies Configuration

# 1. IAM Role for EKS Cluster Control Plane
resource "aws_iam_role" "cluster" {
  name = "idip-${var.environment}-eks-cluster-role"

  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "eks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

  tags = {
    Name        = "idip-${var.environment}-eks-cluster-role"
    Environment = var.environment
  }
}

resource "aws_iam_role_policy_attachment" "cluster_AmazonEKSClusterPolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.cluster.name
}

# 2. IAM Role for EKS Worker Nodes
resource "aws_iam_role" "nodes" {
  name = "idip-${var.environment}-eks-node-role"

  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

  tags = {
    Name        = "idip-${var.environment}-eks-node-role"
    Environment = var.environment
  }
}

resource "aws_iam_role_policy_attachment" "nodes_AmazonEKSWorkerNodePolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.nodes.name
}

resource "aws_iam_role_policy_attachment" "nodes_AmazonEKS_CNI_Policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.nodes.name
}

resource "aws_iam_role_policy_attachment" "nodes_AmazonEC2ContainerRegistryReadOnly" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.nodes.name
}

# 3. Least-Privilege S3 Access Policy for IDIP workloads (Ingestion & DVC)
resource "aws_iam_policy" "s3_access" {
  name        = "idip-${var.environment}-s3-access-policy"
  description = "Scoped read/write S3 access for IDIP pods"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          var.raw_bucket_arn,
          "${var.raw_bucket_arn}/*",
          var.processed_bucket_arn,
          "${var.processed_bucket_arn}/*",
          var.models_bucket_arn,
          "${var.models_bucket_arn}/*",
          var.dvc_bucket_arn,
          "${var.dvc_bucket_arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "nodes_S3Access" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess" # Can also use the scoped policy directly
  role       = aws_iam_role.nodes.name
}

resource "aws_iam_role_policy_attachment" "nodes_custom_S3Access" {
  policy_arn = aws_iam_policy.s3_access.arn
  role       = aws_iam_role.nodes.name
}
