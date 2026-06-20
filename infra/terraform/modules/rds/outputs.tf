output "db_endpoint" {
  value       = aws_db_instance.this.endpoint
  description = "The database endpoint connection string"
}

output "db_address" {
  value       = aws_db_instance.this.address
  description = "The database host address"
}

output "db_port" {
  value       = aws_db_instance.this.port
  description = "The database port"
}

output "db_name" {
  value       = aws_db_instance.this.db_name
  description = "The database name"
}

output "db_user" {
  value       = aws_db_instance.this.username
  description = "The database username"
}
