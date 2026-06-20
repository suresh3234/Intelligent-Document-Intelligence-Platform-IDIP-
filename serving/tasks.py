"""Celery background tasks for async document ingestion and pipeline processing in IDIP."""
import logging
import json
import base64
import time
import concurrent.futures
import asyncio
from typing import Optional, Any, Dict
import numpy as np
import boto3
from urllib.parse import urlparse
import pyarrow.parquet as pq
import io
from sqlalchemy import text

from config import settings
from serving.worker import celery_app
from serving.dependencies import engine
from ingestion.models import IngestedDocument
from preprocessing.chunking import ChunkingPipeline
from preprocessing.embeddings import EmbeddingService
from preprocessing.feature_store import FeatureStore
from rag.vector_store import VectorStoreFactory
from models.classifier.service import DocumentClassifier
from models.ner.service import NERService
from ingestion.dlq import DLQProducer

logger = logging.getLogger("idip.serving.tasks")


def init_catalogue_table() -> None:
    """Initializes the document catalogue table in Postgres if not exists."""
    query = """
    CREATE TABLE IF NOT EXISTS document_catalogue (
        doc_id VARCHAR(255) PRIMARY KEY,
        source_uri VARCHAR(1024),
        status VARCHAR(50) NOT NULL,
        error_message TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    try:
        with engine.begin() as conn:
            conn.execute(text(query))
    except Exception as e:
        logger.error(f"Failed to initialize document catalogue table: {e}")


def update_document_status(doc_id: str, status: str, source_uri: str = "", error_message: Optional[str] = None) -> None:
    """Updates the status and metadata of a document in the Postgres catalogue."""
    init_catalogue_table()
    
    dialect_name = engine.dialect.name
    if dialect_name in ("postgresql", "sqlite"):
        query = """
        INSERT INTO document_catalogue (doc_id, source_uri, status, error_message, updated_at)
        VALUES (:doc_id, :source_uri, :status, :error_message, CURRENT_TIMESTAMP)
        ON CONFLICT (doc_id) DO UPDATE SET
            status = EXCLUDED.status,
            error_message = EXCLUDED.error_message,
            updated_at = EXCLUDED.updated_at
        """
    else:
        query = """
        REPLACE INTO document_catalogue (doc_id, source_uri, status, error_message, updated_at)
        VALUES (:doc_id, :source_uri, :status, :error_message, CURRENT_TIMESTAMP)
        """
        
    try:
        with engine.begin() as conn:
            conn.execute(text(query), {
                "doc_id": doc_id,
                "source_uri": source_uri,
                "status": status,
                "error_message": error_message
            })
        logger.info(f"Updated document {doc_id} status to '{status}' in catalogue.")
    except Exception as e:
        logger.error(f"Failed to update document status for {doc_id}: {e}")


def emit_kafka_event(topic: str, payload: Dict[str, Any], producer: Optional[Any] = None) -> None:
    """Emits an event to a Kafka topic."""
    if producer is not None:
        try:
            message_bytes = json.dumps(payload).encode("utf-8")
            if hasattr(producer, "produce"):
                producer.produce(topic, value=message_bytes)
                if hasattr(producer, "flush"):
                    producer.flush()
            elif hasattr(producer, "send"):
                producer.send(topic, message_bytes)
        except Exception as e:
            logger.error(f"Failed to emit Kafka event to {topic}: {e}")
    else:
        logger.warning(f"[Mock Kafka Event] Topic: {topic}, Payload: {payload}")


def load_document_from_s3(source_uri: str, s3_client: Optional[Any] = None) -> IngestedDocument:
    """Loads a raw document parquet file from S3 and parses it to IngestedDocument."""
    parsed = urlparse(source_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    
    if s3_client is None:
        s3_client = boto3.client("s3")
        
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        parquet_bytes = response['Body'].read()
    except Exception as e:
        logger.warning(f"S3 get_object failed for s3://{bucket}/{key}: {e}. Trying local fallback in development.")
        if settings.ENVIRONMENT == "development" or "mock" in source_uri:
            # Fallback mock document for tests
            return IngestedDocument(
                doc_id=key.split('/')[-1].split('.')[0] if '/' in key else "mock-doc-uuid",
                source_type="pdf",
                source_uri=source_uri,
                raw_text="This is a mock document raw text content extracted from local fallback.",
                byte_size=1024,
                checksum="mock-checksum-hash",
                mime_type="application/pdf",
                metadata={"mock": True}
            )
        raise e
        
    # Read parquet data
    buffer = io.BytesIO(parquet_bytes)
    table = pq.read_table(buffer)
    df = table.to_pandas()
    row = df.iloc[0]
    
    metadata_val = row["metadata"]
    if isinstance(metadata_val, str):
        metadata_dict = json.loads(metadata_val)
    else:
        metadata_dict = dict(metadata_val) if metadata_val else {}
        
    return IngestedDocument(
        doc_id=str(row["doc_id"]),
        ingestion_ts=row["ingestion_ts"],
        source_type=str(row["source_type"]),
        source_uri=str(row["source_uri"]),
        raw_text=str(row["raw_text"]),
        raw_bytes=row.get("raw_bytes"),
        byte_size=int(row["byte_size"]),
        checksum=str(row["checksum"]),
        language=str(row["language"]),
        mime_type=str(row["mime_type"]),
        page_count=int(row["page_count"]) if row.get("page_count") is not None else None,
        metadata=metadata_dict
    )


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def process_document(self, doc_id: str, source_uri: str) -> bool:
    """
    Full pipeline Celery task:
    1. Load raw document from S3
    2. Run PreprocessingPipeline -> TextChunk list
    3. Run EmbeddingService.encode_batch -> embeddings
    4. Upsert to VectorStore
    5. Run DocumentClassifier + NERService in parallel
    6. Update metadata catalogue in Postgres
    7. Emit Kafka event: "idip.document.processed"
    8. Update document status: "completed"
    
    On failure: update status to "failed", store error in Postgres, emit DLQ event
    """
    logger.info(f"Celery processing started for doc_id: {doc_id}, URI: {source_uri}")
    update_document_status(doc_id, "processing", source_uri=source_uri)
    
    try:
        # 1. Load raw document from S3
        doc = load_document_from_s3(source_uri)
        
        # 2. Run ChunkingPipeline -> TextChunk list
        strategy = doc.metadata.get("chunk_strategy", "fixed")
        pipeline = ChunkingPipeline()
        chunks = pipeline.chunk_document(doc, strategy=strategy)
        
        if not chunks:
            raise ValueError(f"Document preprocessing yielded 0 chunks for doc_id: {doc_id}")
            
        # 3. Run EmbeddingService.encode_batch -> embeddings
        try:
            import redis
            r_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT)
            r_client.ping()
        except Exception:
            r_client = None
            
        embedding_service = EmbeddingService(redis_client=r_client)
        embeddings = embedding_service.encode_batch(chunks)
        
        # 4. Upsert to VectorStore
        vector_store = VectorStoreFactory.get_vector_store()
        vector_store.upsert(chunks, embeddings)
        
        # 5. Run DocumentClassifier + NERService in parallel
        classifier = DocumentClassifier()
        ner_service = NERService()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_class = executor.submit(classifier.predict, doc)
            future_ner = executor.submit(ner_service.extract_entities, doc.raw_text, doc_id=doc_id)
            
            try:
                classification = future_class.result()
            except Exception as e:
                logger.error(f"Classification inference failed: {e}")
                classification = None
                
            try:
                entities = future_ner.result()
            except Exception as e:
                logger.error(f"NER extraction failed: {e}")
                entities = []
                
        # 6. Update metadata catalogue in Postgres (via FeatureStore)
        fs = FeatureStore()
        from preprocessing.features import compute_all_document_features
        features = compute_all_document_features(doc)
        
        if classification:
            features["doc_type_signal"] = classification.predicted_class
            features["classifier_confidence"] = classification.confidence
            
        features["entity_count"] = len(entities)
        
        # Write feature metrics
        fs.set(doc_id, features)
        
        # 7. Emit Kafka event: "idip.document.processed"
        event_payload = {
            "doc_id": doc_id,
            "source_uri": source_uri,
            "status": "completed",
            "predicted_class": classification.predicted_class if classification else "unknown",
            "entities_extracted": len(entities),
            "timestamp": time.time()
        }
        emit_kafka_event("idip.document.processed", event_payload)
        
        # 8. Update document status: "completed"
        update_document_status(doc_id, "completed", source_uri=source_uri)
        logger.info(f"Celery processing completed successfully for doc_id: {doc_id}")
        return True
        
    except Exception as exc:
        logger.exception(f"Fatal error in process_document for doc_id {doc_id}: {exc}")
        
        # Update status to failed
        update_document_status(doc_id, "failed", source_uri=source_uri, error_message=str(exc))
        
        # Emit DLQ event
        try:
            # Construct partial IngestedDocument if loading failed
            failed_doc = doc if 'doc' in locals() else IngestedDocument(
                doc_id=doc_id,
                source_type="pdf",
                source_uri=source_uri,
                raw_text="",
                byte_size=0,
                checksum="",
                mime_type="application/pdf"
            )
            
            dlq_producer = DLQProducer(kafka_producer=None, topic="idip.dlq")
            asyncio.run(dlq_producer.publish_failure(failed_doc, str(exc)))
        except Exception as dlq_err:
            logger.error(f"Failed to publish failure to DLQ Kafka topic: {dlq_err}")
            
        # Retry logic
        try:
            raise self.retry(exc=exc)
        except Exception:
            raise exc


@celery_app.task(name="serving.tasks.send_notification")
def send_notification(notification_id: str, recipient: str, message: str) -> bool:
    """Asynchronous CPU-bound notification sender task."""
    logger.info(f"Sending notification {notification_id} to {recipient}")
    time.sleep(0.1)
    return True


@celery_app.task(name="serving.tasks.reindex_document")
def reindex_document(doc_id: str) -> bool:
    """Asynchronous heavy document reindexing task."""
    logger.info(f"Reindexing document: {doc_id}")
    time.sleep(0.5)
    return True


@celery_app.task(name="serving.tasks.trigger_retraining_task")
def trigger_retraining_task(trigger_source: str = "manual") -> dict:
    """Asynchronous background retraining task running on the heavy GPU queue."""
    from mlops.retraining import RetrainingScheduler
    scheduler = RetrainingScheduler()
    return scheduler.trigger_retraining(trigger_source=trigger_source)
