import time
import json
import logging
import os
import uuid
import datetime
from typing import Dict, Any, Optional

from ingestion.models import IngestedDocument, SourceType
from ingestion.exceptions import AdapterError, FileSizeExceededError
from ingestion.base import BaseSourceAdapter
from ingestion.adapters.pdf import PDFAdapter
from ingestion.adapters.image import ImageAdapter
from ingestion.adapters.api import RESTAPIAdapter
from ingestion.adapters.stream import KafkaStreamAdapter
from ingestion.adapters.database import DatabaseAdapter

logger = logging.getLogger("idip.ingestion")

class IngestionOrchestrator:
    """Orchestrates document routing to adapters based on source type and tracks metrics."""
    
    def __init__(self, adapters: Optional[Dict[SourceType, BaseSourceAdapter]] = None, max_file_size: int = 52_428_800):
        # 50 MB threshold by default
        self.max_file_size = max_file_size
        
        # Initialize default adapters if none are supplied
        if adapters:
            self.adapters = adapters
        else:
            self.adapters = {
                "pdf": PDFAdapter(),
                "image": ImageAdapter(),
                "api": RESTAPIAdapter(),
                # Kafka and DB adapters require runtime configs, to be registered manually
            }

    def register_adapter(self, source_type: SourceType, adapter: BaseSourceAdapter) -> None:
        """Register or override an adapter for a source type."""
        self.adapters[source_type] = adapter

    async def route_and_ingest(self, source_type: SourceType, source_uri: str, **kwargs) -> IngestedDocument:
        """
        Routes the document to the corresponding adapter, checking file size constraints.
        Emits structured JSON logs containing performance metrics.
        """
        start_time = time.perf_counter()
        
        adapter = self.adapters.get(source_type)
        if not adapter:
            raise AdapterError(f"No adapter registered for source type: {source_type}")

        # Check local file sizes before processing
        if source_type in ["pdf", "image"] and os.path.exists(source_uri):
            file_size = os.path.getsize(source_uri)
            if file_size > self.max_file_size:
                # Large file (>50MB) streaming fallback or rejection
                # Let's process via a streaming method or raise size limit exception
                raise FileSizeExceededError(
                    f"File '{source_uri}' exceeds size limit of {self.max_file_size} bytes (size: {file_size} bytes). "
                    f"Large files must be micro-batched via stream or partition ingestion."
                )

        try:
            # Perform ingestion
            doc = await adapter.ingest(source_uri, **kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Emit structured JSON logs
            log_data = {
                "message": "Document ingested successfully",
                "doc_id": doc.doc_id,
                "source_type": source_type,
                "byte_size": doc.byte_size,
                "duration_ms": round(duration_ms, 2)
            }
            logger.info(json.dumps(log_data))
            
            return doc
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            log_data = {
                "message": "Document ingestion failed",
                "source_type": source_type,
                "source_uri": source_uri,
                "duration_ms": round(duration_ms, 2),
                "error": str(e)
            }
            logger.error(json.dumps(log_data))
            raise
