"""Celery worker module for background document parsing and queue orchestration in IDIP."""
import logging
from celery import Celery
from config import settings

logger = logging.getLogger("idip.serving.worker")

# Initialize Celery App
# broker and backend should connect to Redis
celery_app = Celery(
    "idip_tasks",
    broker=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/0",
    backend=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/0"
)

# Celery Configuration
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task routing limits
    task_soft_time_limit=300,
    task_time_limit=600,
    task_acks_late=True,
    
    # Task routing queues
    task_routes={
        "serving.worker.ingest_document_task": {"queue": "heavy"},
        "serving.tasks.process_document": {"queue": "heavy"},
        "serving.tasks.send_notification": {"queue": "light"},
        "serving.tasks.reindex_document": {"queue": "heavy"},
        "serving.tasks.trigger_retraining_task": {"queue": "heavy"},
    },
    
    # Fail fast locally when Redis is not running (prevents hangs during tests)
    broker_connection_retry_on_startup=False,
    broker_connection_max_retries=1,
    redis_backend_health_check_interval=1,
    result_backend_transport_options={"retry_policy": {"max_retries": 1}},
)

# Set concurrency configurations as defaults
celery_app.conf.worker_concurrency = 4  # Default to 'heavy' workers concurrency limit

@celery_app.task(name="serving.worker.ingest_document_task")
def ingest_document_task(doc_id: str, file_content_b64: str, metadata_json: str) -> dict:
    """Asynchronous background Celery task to parse and ingest document payload."""
    logger.info(f"Background task starting for doc_id: {doc_id}")
    try:
        import base64
        import json
        import time
        import asyncio
        from serving.tasks import update_document_status
        from preprocessing.feature_store import FeatureStore
        from ingestion.models import IngestedDocument
        from preprocessing.features import compute_all_document_features
        from ingestion.adapters.pdf import PDFAdapter
        from ingestion.adapters.image import ImageAdapter
        from preprocessing.chunking import ChunkingPipeline
        from preprocessing.embeddings import EmbeddingService
        from rag.vector_store import VectorStoreFactory
        
        file_bytes = base64.b64decode(file_content_b64.encode("utf-8"))
        metadata = json.loads(metadata_json)
        source_uri = metadata.get("source_uri") or f"upload://{doc_id}"
        
        # 1. Update status to 'processing'
        update_document_status(doc_id, "processing", source_uri=source_uri)
        
        # Determine source type extension and select adapter
        ext = source_uri.split('.')[-1].lower() if '.' in source_uri else "pdf"
        if ext in ("png", "jpg", "jpeg", "tiff"):
            source_type = "image"
        else:
            source_type = metadata.get("source_type") or "pdf"
            source_type = source_type.lower()
            if source_type in ("png", "jpg", "jpeg"):
                source_type = "image"
            else:
                source_type = "pdf"

        # Route and ingest using the appropriate adapter in the asyncio loop
        doc = None
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            if source_type == "image":
                adapter = ImageAdapter()
                doc = loop.run_until_complete(adapter.ingest(source_uri, raw_bytes=file_bytes))
            else:
                adapter = PDFAdapter()
                doc = loop.run_until_complete(adapter.ingest(source_uri, raw_bytes=file_bytes))
            
            doc.doc_id = doc_id
        except Exception as ingest_err:
            logger.error(f"Failed to ingest document via adapter: {ingest_err}. Falling back to raw text decode.")
            doc = None

        if doc is None:
            raw_text = file_bytes.decode("utf-8", errors="ignore")
            doc = IngestedDocument(
                doc_id=doc_id,
                source_type="pdf",
                source_uri=source_uri,
                raw_text=raw_text,
                byte_size=len(file_bytes),
                checksum="mock",
                mime_type="application/pdf",
                metadata=metadata
            )

        # 2. Clean and Chunk the document
        strategy = metadata.get("chunk_strategy", "fixed")
        pipeline = ChunkingPipeline()
        chunks = pipeline.chunk_document(doc, strategy=strategy)
        
        # 3. Generate Embeddings and Index inside Vector Store
        if chunks:
            try:
                import redis
                r_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT)
                r_client.ping()
            except Exception:
                r_client = None
                
            embedding_service = EmbeddingService(redis_client=r_client)
            embeddings = embedding_service.encode_batch(chunks)
            
            vector_store = VectorStoreFactory.get_vector_store()
            vector_store.upsert(chunks, embeddings)
            logger.info(f"Successfully upserted {len(chunks)} chunks to vector store for doc_id {doc_id}.")
        else:
            logger.warning(f"No chunks generated for doc_id {doc_id}.")

        # 4. Document Classification (with keyword-based fallback)
        doc_type_signal = "other"
        classifier_confidence = 0.95
        try:
            from models.classifier.service import DocumentClassifier
            classifier = DocumentClassifier()
            classification = classifier.predict(doc)
            doc_type_signal = classification.predicted_class
            classifier_confidence = classification.confidence
        except Exception as cls_err:
            logger.info(f"Classifier prediction bypassed/failed: {cls_err}. Applying fallback heuristics.")
            text_lower = doc.raw_text.lower()
            if "resume" in text_lower or "cv" in text_lower or "curriculum vitae" in text_lower or "experience" in text_lower:
                doc_type_signal = "resume"
            elif "contract" in text_lower or "agreement" in text_lower:
                doc_type_signal = "contract"
            elif "invoice" in text_lower or "bill" in text_lower:
                doc_type_signal = "invoice"
            elif "report" in text_lower:
                doc_type_signal = "report"
            else:
                doc_type_signal = "other"

        # 5. Compute features and save to FeatureStore
        features = compute_all_document_features(doc)
        features["doc_type_signal"] = doc_type_signal
        features["classifier_confidence"] = classifier_confidence
        features["source_uri"] = source_uri
        features["byte_size"] = len(file_bytes)
        features["checksum"] = "mock"
        features["language"] = doc.language or "en"
        features["page_count"] = doc.page_count or 1
        
        fs = FeatureStore()
        fs.set(doc_id, features)
        
        # 6. Update status to 'completed'
        update_document_status(doc_id, "completed", source_uri=source_uri)
        
        logger.info(f"Background task successfully processed doc_id: {doc_id}")
        return {
            "doc_id": doc_id,
            "status": "success",
            "size_processed": len(file_bytes),
            "metadata": metadata
        }
    except Exception as e:
        logger.error(f"Error in background ingestion for doc_id {doc_id}: {e}")
        try:
            from serving.tasks import update_document_status
            update_document_status(doc_id, "failed", error_message=str(e))
        except Exception:
            pass
        return {
            "doc_id": doc_id,
            "status": "failed",
            "error": str(e)
        }

# Import tasks to ensure registration on startup/worker initiation
# Note: Import at the end to avoid circular import issues
import serving.tasks  # noqa: F401
