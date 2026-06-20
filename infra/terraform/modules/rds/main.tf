# RDS PostgreSQL 15 Configuration
resource "aws_db_subnet_group" "this" {
  name       = "idip-${var.environment}-rds-subnet-group"
  subnet_ids = var.subnet_ids

  tags = {
    Name        = "idip-${var.environment}-rds-subnet-group"
    Environment = var.environment
  }
}

resource "aws_security_group" "rds" {
  name        = "idip-${var.environment}-rds-sg"
  description = "Access to RDS PostgreSQL from EKS nodes"
  vpc_id      = var.vpc_id

  ingress {
    description = "PostgreSQL access from VPC"
    from_port   = 5432
    to_port     = 5432
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
    Name        = "idip-${var.environment}-rds-sg"
    Environment = var.environment
  }
}

resource "aws_db_instance" "this" {
  identifier             = "idip-${var.environment}-db"
  engine                 = "postgres"
  engine_version         = "15"
  instance_class         = var.instance_class
  allocated_storage      = var.allocated_storage
  max_allocated_storage  = 100
  db_name                = var.db_name
  username               = var.db_user
  password               = var.db_password
  db_subnet_group_name   = aws_db_subnet_group.this.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  multi_az               = var.multi_az
  skip_final_snapshot    = var.environment == "production" ? false : true
  final_snapshot_identifier = "idip-${var.environment}-db-final-snapshot"
  storage_encrypted      = true

  tags = {
    Name        = "idip-${var.environment}-db"
    Environment = var.environment
  }
}
