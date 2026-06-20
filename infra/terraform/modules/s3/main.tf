# S3 Buckets Configuration
resource "aws_kms_key" "s3_key" {
  description             = "KMS key for IDIP S3 buckets encryption"
  deletion_window_in_days = 10
  enable_key_rotation     = true

  tags = {
    Name        = "idip-${var.environment}-s3-key"
    Environment = var.environment
  }
}

# 1. Raw Data Bucket
resource "aws_s3_bucket" "raw" {
  bucket        = "idip-${var.environment}-raw-data"
  force_destroy = var.environment == "production" ? false : true

  tags = {
    Name        = "idip-${var.environment}-raw-data"
    Environment = var.environment
  }
}

# 2. Processed Data Bucket
resource "aws_s3_bucket" "processed" {
  bucket        = "idip-${var.environment}-processed-data"
  force_destroy = var.environment == "production" ? false : true

  tags = {
    Name        = "idip-${var.environment}-processed-data"
    Environment = var.environment
  }
}

# 3. Model Artifacts Bucket
resource "aws_s3_bucket" "models" {
  bucket        = "idip-${var.environment}-model-artifacts"
  force_destroy = var.environment == "production" ? false : true

  tags = {
    Name        = "idip-${var.environment}-model-artifacts"
    Environment = var.environment
  }
}

# 4. DVC Store Bucket
resource "aws_s3_bucket" "dvc" {
  bucket        = "idip-${var.environment}-dvc-store"
  force_destroy = var.environment == "production" ? false : true

  tags = {
    Name        = "idip-${var.environment}-dvc-store"
    Environment = var.environment
  }
}

# Set Default SSE-KMS Encryption for all buckets
resource "aws_s3_bucket_server_side_encryption_configuration" "raw" {
  bucket = aws_s3_bucket.raw.id
  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.s3_key.arn
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "processed" {
  bucket = aws_s3_bucket.processed.id
  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.s3_key.arn
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id
  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.s3_key.arn
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "dvc" {
  bucket = aws_s3_bucket.dvc.id
  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.s3_key.arn
      sse_algorithm     = "aws:kms"
    }
  }
}

# Enable Public Access Block for all buckets
resource "aws_s3_bucket_public_access_block" "raw" {
  bucket                  = aws_s3_bucket.raw.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "processed" {
  bucket                  = aws_s3_bucket.processed.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket                  = aws_s3_bucket.models.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "dvc" {
  bucket                  = aws_s3_bucket.dvc.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
