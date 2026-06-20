terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# 1. VPC Module
module "vpc" {
  source             = "./modules/vpc"
  environment        = var.environment
  vpc_cidr           = var.vpc_cidr
  public_subnets     = var.public_subnets
  private_subnets    = var.private_subnets
  availability_zones = var.availability_zones
  enable_nat_gateway = var.enable_nat_gateway
}

# 2. S3 Buckets Module
module "s3" {
  source      = "./modules/s3"
  environment = var.environment
}

# 3. ECR Repositories Module
module "ecr" {
  source      = "./modules/ecr"
  environment = var.environment
}

# 4. IAM Roles & Policies Module
module "iam" {
  source               = "./modules/iam"
  environment          = var.environment
  raw_bucket_arn       = module.s3.raw_bucket_arn
  processed_bucket_arn = module.s3.processed_bucket_arn
  models_bucket_arn    = module.s3.models_bucket_arn
  dvc_bucket_arn       = module.s3.dvc_bucket_arn
}

# 5. EKS Cluster Module
module "eks" {
  source             = "./modules/eks"
  environment        = var.environment
  cluster_name       = "idip-${var.environment}"
  cluster_role_arn   = module.iam.cluster_role_arn
  node_role_arn      = module.iam.node_role_arn
  subnet_ids         = module.vpc.private_subnet_ids
  cpu_instance_types = var.cpu_instance_types
  gpu_instance_types = var.gpu_instance_types
  cpu_desired_size   = var.cpu_desired_size
  cpu_min_size       = var.cpu_min_size
  cpu_max_size       = var.cpu_max_size
  gpu_desired_size   = var.gpu_desired_size
  gpu_min_size       = var.gpu_min_size
  gpu_max_size       = var.gpu_max_size
}

# 6. RDS PostgreSQL Database Module
module "rds" {
  source            = "./modules/rds"
  environment       = var.environment
  vpc_id            = module.vpc.vpc_id
  subnet_ids        = module.vpc.private_subnet_ids
  vpc_cidr          = module.vpc.vpc_cidr_block
  db_name           = var.db_name
  db_user           = var.db_user
  db_password       = var.db_password
  instance_class    = var.db_instance_class
  allocated_storage = var.db_allocated_storage
  multi_az          = var.environment == "production" ? true : var.db_multi_az
}

# 7. ElastiCache Redis Cluster Module
module "elasticache" {
  source             = "./modules/elasticache"
  environment        = var.environment
  vpc_id             = module.vpc.vpc_id
  subnet_ids         = module.vpc.private_subnet_ids
  vpc_cidr           = module.vpc.vpc_cidr_block
  node_type          = var.redis_node_type
  num_cache_clusters = var.redis_num_cache_clusters
}
