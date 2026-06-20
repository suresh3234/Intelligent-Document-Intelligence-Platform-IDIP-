output "redis_primary_endpoint" {
  value       = aws_elasticache_replication_group.this.primary_endpoint_address
  description = "Primary connection endpoint for Redis replication group"
}

output "redis_port" {
  value       = 6379
  description = "Redis service port"
}
