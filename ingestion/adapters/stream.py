import uuid
import datetime
import json
from typing import Optional, Dict, Any, List
from ingestion.base import BaseSourceAdapter
from ingestion.models import IngestedDocument
from ingestion.exceptions import AdapterError

# Try imports, fallback to dummy objects for test isolation
try:
    from confluent_kafka import Consumer, KafkaError
except ImportError:
    Consumer = None  # type: ignore
    KafkaError = None  # type: ignore

class KafkaStreamAdapter(BaseSourceAdapter):
    """Adapter for ingesting streaming documents via Kafka and buffering micro-batches to AWS S3."""
    
    def __init__(self, kafka_config: Dict[str, Any], schema_registry_client: Optional[Any] = None, s3_client: Optional[Any] = None, bucket_name: str = "idip-data-bucket"):
        self.kafka_config = kafka_config
        self.schema_registry_client = schema_registry_client
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.consumer = None
        
        # Configure group.id if not present
        if "group.id" not in self.kafka_config:
            self.kafka_config["group.id"] = "idip-ingestion-group"
        if "auto.offset.reset" not in self.kafka_config:
            self.kafka_config["auto.offset.reset"] = "earliest"

    def connect(self) -> None:
        """Initializes the Kafka Consumer connection."""
        if Consumer is None:
            raise AdapterError("confluent-kafka package is missing from environment.")
        try:
            self.consumer = Consumer(self.kafka_config)
            self.consumer.subscribe(["^raw.*"])
        except Exception as e:
            raise AdapterError(f"Failed to connect and subscribe to Kafka topics: {str(e)}") from e

    async def ingest(self, source_uri: str, **kwargs) -> IngestedDocument:
        """
        Implements the single document fallback interface for streaming.
        Generally, stream adapters use micro-batching rather than single document fetches.
        """
        raw_bytes = kwargs.get("raw_bytes")
        if raw_bytes is None:
            raise AdapterError("KafkaStreamAdapter ingest requires raw_bytes in keyword arguments.")
        
        checksum = self.compute_checksum(raw_bytes)
        byte_size = len(raw_bytes)
        raw_text = self.safe_decode(raw_bytes)
        
        # Check if Schema Registry is available to decode Avro
        decoded_payload = raw_text
        if self.schema_registry_client:
            try:
                decoded_payload = self.schema_registry_client.decode(raw_bytes)
            except Exception as avro_err:
                raise AdapterError(f"Avro deserialization failed: {str(avro_err)}") from avro_err
        
        return IngestedDocument(
            doc_id=str(uuid.uuid4()),
            ingestion_ts=datetime.datetime.utcnow(),
            source_type="stream",
            source_uri=source_uri,
            raw_text=str(decoded_payload),
            raw_bytes=raw_bytes,
            byte_size=byte_size,
            checksum=checksum,
            language=kwargs.get("language", "en"),
            mime_type=kwargs.get("mime_type", "application/json"),
            page_count=None,
            metadata=kwargs.get("metadata", {})
        )

    async def consume_micro_batch(self, max_messages: int = 100, timeout_sec: float = 1.0) -> List[IngestedDocument]:
        """
        Polls Kafka for a batch of messages, processes them, 
        uploads the raw batch buffer to S3, and returns the documents list.
        """
        if not self.consumer:
            self.connect()

        documents: List[IngestedDocument] = []
        batch_payloads: List[Dict[str, Any]] = []
        
        # Poll loop
        for _ in range(max_messages):
            msg = self.consumer.poll(timeout=timeout_sec) # type: ignore
            if msg is None:
                break
            if msg.error():
                # Handle error
                if msg.error().code() == KafkaError._PARTITION_EOF: # type: ignore
                    continue
                else:
                    raise AdapterError(f"Kafka consumer error: {msg.error()}")
            
            payload_bytes = msg.value()
            if not payload_bytes:
                continue

            try:
                checksum = self.compute_checksum(payload_bytes)
                
                # Deserialization step
                if self.schema_registry_client:
                    decoded = self.schema_registry_client.decode(payload_bytes)
                else:
                    decoded = json.loads(payload_bytes.decode("utf-8"))
                
                doc_id = str(uuid.uuid4())
                source_uri = f"kafka://{msg.topic()}/{msg.partition()}/{msg.offset()}"
                
                doc = IngestedDocument(
                    doc_id=doc_id,
                    ingestion_ts=datetime.datetime.utcnow(),
                    source_type="stream",
                    source_uri=source_uri,
                    raw_text=json.dumps(decoded),
                    raw_bytes=payload_bytes,
                    byte_size=len(payload_bytes),
                    checksum=checksum,
                    language="en",
                    mime_type="application/json",
                    metadata={
                        "topic": msg.topic(),
                        "partition": msg.partition(),
                        "offset": msg.offset()
                    }
                )
                documents.append(doc)
                batch_payloads.append({
                    "doc_id": doc_id,
                    "uri": source_uri,
                    "payload": decoded,
                    "ingestion_ts": doc.ingestion_ts.isoformat()
                })
            except Exception as parse_err:
                # Log and continue parsing other messages in batch
                continue

        # Upload batch to S3 if payloads exist
        if batch_payloads and self.s3_client:
            try:
                now = datetime.datetime.utcnow()
                s3_key = f"raw/{now.year}/{now.month:02d}/{now.day:02d}/stream/batch-{uuid.uuid4()}.json"
                batch_data = json.dumps(batch_payloads).encode("utf-8")
                
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=batch_data,
                    ContentType="application/json"
                )
                
                # Update documents metadata with S3 location
                for doc in documents:
                    doc.metadata["s3_batch_uri"] = f"s3://{self.bucket_name}/{s3_key}"
            except Exception as s3_err:
                raise AdapterError(f"Failed to flush micro-batch to S3: {str(s3_err)}") from s3_err

        return documents

    def close(self) -> None:
        """Closes Kafka Consumer resource gracefully."""
        if self.consumer:
            try:
                self.consumer.close()
            except Exception:
                pass
            self.consumer = None
