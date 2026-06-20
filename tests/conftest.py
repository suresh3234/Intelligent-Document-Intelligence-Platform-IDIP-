import os
import sys
import pytest
import boto3
from unittest.mock import MagicMock
from sqlalchemy import create_engine

# Disable Ryuk for offline/restricted environments to avoid pulling additional cleanup containers
os.environ["TESTCONTAINERS_RYUK_DISABLED"] = "true"

from sqlalchemy.orm import sessionmaker
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer
from testcontainers.core.container import DockerContainer
from moto import mock_aws

# Ensure the root of the project is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings

# --- 1. Testcontainers Fixtures ---

@pytest.fixture(scope="session")
def postgres_service():
    """Starts a PostgreSQL container and returns the connection string."""
    print("\nStarting Postgres Testcontainer...")
    # Use pgvector/pgvector:pg16 which is locally available on the host machine
    with PostgresContainer("pgvector/pgvector:pg16") as postgres:
        db_url = postgres.get_connection_url()
        
        # Override settings for the application
        settings.POSTGRES_HOST = postgres.get_container_host_ip()
        settings.POSTGRES_PORT = int(postgres.get_exposed_port(5432))
        settings.POSTGRES_USER = postgres.username
        settings.POSTGRES_PASSWORD = postgres.password
        settings.POSTGRES_DB = postgres.dbname
        
        # Patch serving dependencies engine
        import serving.dependencies
        serving.dependencies.engine = create_engine(db_url)
        serving.dependencies.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=serving.dependencies.engine
        )
        
        # Patch serving tasks engine
        import serving.tasks
        serving.tasks.engine = serving.dependencies.engine
        
        # Patch drift monitoring engine as well
        import monitoring.drift
        monitoring.drift.engine = serving.dependencies.engine
        
        # Patch ab_testing engine
        import mlops.ab_testing
        mlops.ab_testing.engine = serving.dependencies.engine
        
        yield db_url

@pytest.fixture(scope="session")
def redis_service():
    """Starts a Redis container and overrides settings with host/port."""
    print("Starting Redis Testcontainer...")
    with RedisContainer("redis:7-alpine") as redis_container:
        host = redis_container.get_container_host_ip()
        port = int(redis_container.get_exposed_port(6379))
        
        settings.REDIS_HOST = host
        settings.REDIS_PORT = port
        
        redis_url = f"redis://{host}:{port}/0"
        yield redis_container

@pytest.fixture(scope="session")
def weaviate_service():
    """Starts a Weaviate container for vector search tests."""
    print("Starting Weaviate Testcontainer...")
    weaviate_container = DockerContainer("semitechnologies/weaviate:1.24.1")
    weaviate_container.with_exposed_ports(8080)
    weaviate_container.with_env("AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED", "true")
    weaviate_container.with_env("PERSISTENCE_DATA_PATH", "/var/lib/weaviate")
    
    try:
        with weaviate_container:
            host = weaviate_container.get_container_host_ip()
            port = weaviate_container.get_exposed_port(8080)
            weaviate_url = f"http://{host}:{port}"
            os.environ["WEAVIATE_URL"] = weaviate_url
            yield weaviate_url
    except Exception as e:
        print(f"\n[Warning] Weaviate testcontainer start failed (offline mode): {e}. Falling back to FAISS/Mock.")
        os.environ["WEAVIATE_URL"] = "http://localhost:8080"
        yield "http://localhost:8080"

# --- 2. AWS Moto S3 Fixtures ---

@pytest.fixture(scope="session")
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-west-2"

@pytest.fixture(scope="session")
def s3_client(aws_credentials):
    """Sets up a mocked S3 environment with all project buckets pre-created."""
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-west-2")
        
        buckets = [
            "idip-raw-data",
            "idip-processed-data",
            "idip-model-artifacts",
            "idip-dvc-store"
        ]
        for bucket in buckets:
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={'LocationConstraint': 'us-west-2'}
            )
        yield s3
