# ElastiCache Redis Cluster Configuration
resource "aws_elasticache_subnet_group" "this" {
  name       = "idip-${var.environment}-redis-subnet-group"
  subnet_ids = var.subnet_ids

  tags = {
    Name        = "idip-${var.environment}-redis-subnet-group"
    Environment = var.environment
  }
}

resource "aws_security_group" "redis" {
  name        = "idip-${var.environment}-redis-sg"
  description = "Access to ElastiCache Redis from EKS nodes"
  vpc_id      = var.vpc_id

  ingress {
    description = "Redis access from VPC"
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }

  tags = {
    Name        = "idip-${var.environment}-redis-sg"
    Environment = var.environment
  }
}

resource "aws_elasticache_replication_group" "this" {
  replication_group_id        = "idip-${var.environment}-redis"
  description                 = "Redis cluster replication group for IDIP semantic caching and Celery broker"
  node_type                   = var.node_type
  port                        = 6379
  parameter_group_name        = "default.redis7"
  automatic_failover_enabled  = true
  multi_az_enabled            = true
  num_cache_clusters          = var.num_cache_clusters
  subnet_group_name           = aws_elasticache_subnet_group.this.name
  security_group_ids          = [aws_security_group.redis.id]
  at_rest_encryption_enabled = true
  transit_encryption_enabled  = true

  tags = {
    Name        = "idip-${var.environment}-redis-cluster"
    Environment = var.environment
  }
}
