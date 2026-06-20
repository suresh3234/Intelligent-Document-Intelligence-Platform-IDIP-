import uuid
import datetime
import json
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine

from ingestion.base import BaseSourceAdapter
from ingestion.models import IngestedDocument
from ingestion.exceptions import AdapterError

class DatabaseAdapter(BaseSourceAdapter):
    """Adapter for processing database CDC events (via Debezium/Kafka WAL stream) using SQLAlchemy."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine: Optional[AsyncEngine] = None

    def connect(self) -> None:
        """Initializes the async SQLAlchemy engine."""
        try:
            self.engine = create_async_engine(self.database_url, echo=False)
        except Exception as e:
            raise AdapterError(f"Failed to create SQLAlchemy async engine: {str(e)}") from e

    async def ingest(self, source_uri: str, **kwargs) -> IngestedDocument:
        """
        Ingests a CDC database WAL event message from Kafka.
        Parses Debezium schemas for PostgreSQL events (c=Create, u=Update, d=Delete).
        """
        cdc_payload = kwargs.get("cdc_payload")
        if not cdc_payload:
            raise AdapterError("DatabaseAdapter requires 'cdc_payload' keyword argument containing the CDC event.")
            
        try:
            # Parse payload string if passed as bytes/str
            if isinstance(cdc_payload, (bytes, str)):
                event = json.loads(cdc_payload)
            else:
                event = cdc_payload

            op = event.get("op")  # 'c' = insert, 'u' = update, 'd' = delete
            source = event.get("source", {})
            schema = source.get("schema", "public")
            table = source.get("table", "unknown")
            
            # Identify changes
            before = event.get("before")
            after = event.get("after")
            
            if op == "d":
                # Delete event: document has been removed from db
                raw_text = f"DELETED RECORD: {json.dumps(before)}"
                target_record = before or {}
            else:
                # Insert or Update event
                raw_text = json.dumps(after) if after else ""
                target_record = after or {}
                
            raw_bytes = json.dumps(event).encode("utf-8")
            checksum = self.compute_checksum(raw_bytes)
            byte_size = len(raw_bytes)
            
            # Capture specific ID fields if present to preserve document context
            record_id = target_record.get("id") or target_record.get("uuid") or str(uuid.uuid4())
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{schema}.{table}.{record_id}"))

            metadata = {
                "db_operation": op,
                "db_schema": schema,
                "db_table": table,
                "record_id": record_id,
                "before": before,
                "after": after
            }

            return IngestedDocument(
                doc_id=doc_id,
                ingestion_ts=datetime.datetime.utcnow(),
                source_type="database",
                source_uri=f"postgresql://{schema}/{table}/{record_id}",
                raw_text=raw_text,
                raw_bytes=raw_bytes,
                byte_size=byte_size,
                checksum=checksum,
                language=kwargs.get("language", "en"),
                mime_type="application/json",
                page_count=None,
                metadata=metadata
            )

        except Exception as e:
            raise AdapterError(f"Failed to process CDC database event: {str(e)}") from e

    async def close(self) -> None:
        """Disposes the SQLAlchemy engine connection."""
        if self.engine:
            try:
                await self.engine.dispose()
            except Exception:
                pass
            self.engine = None
