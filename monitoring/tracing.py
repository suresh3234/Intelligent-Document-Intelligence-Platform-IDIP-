import os
import logging
import contextlib
from typing import Dict, Any, Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource

from config.settings import Settings

logger = logging.getLogger("idip.monitoring.tracing")
tracer = trace.get_tracer("idip")

def setup_tracing(service_name: str = "idip-service", settings: Optional[Settings] = None) -> None:
    """Configures the global OpenTelemetry TracerProvider and registers exporter processors."""
    # Prevent re-registration warnings
    if isinstance(trace.get_tracer_provider(), TracerProvider):
        logger.info("OpenTelemetry TracerProvider already configured.")
        return

    settings = settings or Settings()
    env = settings.ENVIRONMENT

    # Create tracer provider resource
    resource = Resource.create(attributes={"service.name": service_name, "environment": env})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    # Determine span processor exporter based on environment configuration
    span_processor = None

    if env == "production":
        try:
            # GCP Cloud Trace Exporter integration
            from opentelemetry.exporter.google_cloud_trace import GoogleCloudTraceSpanExporter
            exporter = GoogleCloudTraceSpanExporter()
            span_processor = BatchSpanProcessor(exporter)
            logger.info("Configured Google Cloud Trace Span Exporter for production tracing.")
        except ImportError:
            logger.warning("google-cloud-trace exporter package is not installed. Falling back to ConsoleSpanExporter.")

    if not span_processor:
        try:
            # Jaeger OTLP/gRPC Span Exporter integration
            # We configure OTLPSpanExporter pointing to local or env target
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
            exporter = OTLPSpanExporter(endpoint=endpoint)
            span_processor = BatchSpanProcessor(exporter)
            logger.info(f"Configured OTLP/gRPC Span Exporter pointing to: {endpoint}")
        except ImportError:
            try:
                # Console fallback
                exporter = ConsoleSpanExporter()
                span_processor = BatchSpanProcessor(exporter)
                logger.info("Configured ConsoleSpanExporter fallback for local development tracing.")
            except Exception as e:
                logger.error(f"Failed to configure console exporter fallback: {e}")

    if span_processor:
        provider.add_span_processor(span_processor)

    logger.info("OpenTelemetry distributed tracing successfully configured.")


@contextlib.contextmanager
def trace_span(span_name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Context manager to easily construct nested tracer spans with custom attributes:
      doc_id, query_id, model_name, cache_hit, confidence
    """
    with tracer.start_as_current_span(span_name) as span:
        if attributes:
            for key, val in attributes.items():
                if val is not None:
                    # Cast list and dict attributes to strings for standard trace logs
                    if isinstance(val, (dict, list)):
                        val = str(val)
                    span.set_attribute(key, val)
        yield span
