import json
import logging
import asyncio
import datetime
from typing import Optional, Any, Callable, Dict

from ingestion.models import IngestedDocument

logger = logging.getLogger("idip.ingestion.dlq")

class DLQProducer:
    """Publishes validation-failed document envelopes to a Kafka DLQ topic."""
    
    def __init__(self, kafka_producer: Optional[Any] = None, topic: str = "idip.dlq"):
        self.producer = kafka_producer
        self.topic = topic

    async def publish_failure(self, doc: IngestedDocument, failure_reason: str) -> None:
        """Serializes failed IngestedDocument envelope and publishes to Kafka DLQ."""
        payload = {
            "failure_reason": failure_reason,
            "failed_at": datetime.datetime.utcnow().isoformat(),
            # Convert IngestedDocument model to dict, handling binary fields
            "document": doc.model_dump(mode="json")
        }
        
        message_bytes = json.dumps(payload).encode("utf-8")
        
        if self.producer:
            try:
                # Expecting an async producer interface (like aiokafka) or synchronous confluent-kafka producer
                if hasattr(self.producer, "send_and_wait"):
                    res = self.producer.send_and_wait(self.topic, message_bytes)
                    if asyncio.iscoroutine(res) or hasattr(res, "__await__"):
                        await res
                elif hasattr(self.producer, "produce"):
                    self.producer.produce(self.topic, value=message_bytes)
                    if hasattr(self.producer, "flush"):
                        self.producer.flush()
                else:
                    # Generic send hook
                    self.producer.send(self.topic, message_bytes)
            except Exception as e:
                logger.error(f"Failed to publish document {doc.doc_id} to DLQ Kafka topic: {str(e)}")
        else:
            # Fallback local log if Kafka client not injected
            logger.warning(f"[Mock DLQ Publish] Topic: {self.topic}, Document ID: {doc.doc_id}, Reason: {failure_reason}")


class DLQConsumer:
    """Consumes from the Kafka DLQ, increments metrics, and schedules retries."""
    
    def __init__(self, prometheus_counter: Optional[Any] = None, retry_callback: Optional[Callable[[Dict[str, Any]], Any]] = None):
        self.dlq_counter = prometheus_counter
        self.retry_callback = retry_callback

    async def handle_dlq_message(self, message_bytes: bytes) -> None:
        """Processes DLQ message, logs alert, increments metric, and schedules a retry."""
        try:
            payload = json.loads(message_bytes.decode("utf-8"))
            failure_reason = payload.get("failure_reason", "Unknown validation failure")
            failed_at = payload.get("failed_at")
            doc_data = payload.get("document", {})
            doc_id = doc_data.get("doc_id", "unknown-doc-id")
            source_uri = doc_data.get("source_uri", "unknown-uri")

            # Log alert
            logger.error(
                f"ALERT: Ingestion DLQ event captured. Document ID: {doc_id}, "
                f"Source: {source_uri}, Reason: {failure_reason}, Failed At: {failed_at}"
            )

            # Increment Prometheus counter: idip_dlq_total
            if self.dlq_counter:
                try:
                    self.dlq_counter.inc(labels={"source_type": doc_data.get("source_type", "unknown"), "reason": failure_reason})
                except Exception:
                    # Fallback to standard counter API
                    try:
                        self.dlq_counter.inc()
                    except Exception:
                        pass

            # Schedule delayed retry (after 5 minutes = 300 seconds)
            asyncio.create_task(self._schedule_retry(doc_data))

        except Exception as e:
            logger.error(f"Error parsing or handling DLQ Kafka message: {str(e)}")

    async def _schedule_retry(self, doc_data: Dict[str, Any]) -> None:
        """Waits 5 minutes (300 seconds) before executing the retry callback."""
        doc_id = doc_data.get("doc_id", "unknown-doc-id")
        logger.info(f"Scheduling retry task for Document ID {doc_id} in 300 seconds...")
        
        # In testing environments, we might want to override delay
        delay = 300
        # If running unit tests, accelerate the delay to keep tests fast
        if doc_data.get("metadata", {}).get("test_mode") is True:
            delay = 0.1

        await asyncio.sleep(delay)
        
        if self.retry_callback:
            try:
                logger.info(f"Executing retry task execution for Document ID {doc_id}...")
                await self.retry_callback(doc_data)
            except Exception as retry_err:
                logger.error(f"Retry execution failed for Document ID {doc_id}: {str(retry_err)}")
        else:
            logger.warning(f"No retry callback registered. Document ID {doc_id} skipped.")
