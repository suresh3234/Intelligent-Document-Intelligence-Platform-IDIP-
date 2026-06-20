from prometheus_client import REGISTRY, Counter, Histogram, Gauge

def _get_or_create_counter(name: str, documentation: str, labelnames: tuple = ()) -> Counter:
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Counter(name, documentation, labelnames)

def _get_or_create_histogram(name: str, documentation: str, labelnames: tuple = ()) -> Histogram:
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Histogram(name, documentation, labelnames)

def _get_or_create_gauge(name: str, documentation: str, labelnames: tuple = ()) -> Gauge:
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Gauge(name, documentation, labelnames)

# --- COUNTERS ---

idip_documents_ingested_total = _get_or_create_counter(
    "idip_documents_ingested_total",
    "Total number of raw documents ingested into the system",
    ["source_type", "status"]
)

idip_queries_total = _get_or_create_counter(
    "idip_queries_total",
    "Total RAG queries processed by the serving layer",
    ["model", "cached"]
)

idip_dlq_total = _get_or_create_counter(
    "idip_dlq_total",
    "Total documents processed by the DLQ due to failure",
    ["source_type", "failure_reason"]
)

# Reuse cache metrics from serving.cache if available to prevent double registration
try:
    from serving.cache import idip_cache_hits_total, idip_cache_misses_total
except ImportError:
    idip_cache_hits_total = _get_or_create_counter(
        "idip_cache_hits_total",
        "Total semantic cache hit counts"
    )

    idip_cache_misses_total = _get_or_create_counter(
        "idip_cache_misses_total",
        "Total semantic cache miss counts"
    )

# --- HISTOGRAMS ---

idip_ingestion_duration_seconds = _get_or_create_histogram(
    "idip_ingestion_duration_seconds",
    "Document ingestion processing latency in seconds",
    ["source_type"]
)

idip_inference_duration_seconds = _get_or_create_histogram(
    "idip_inference_duration_seconds",
    "ML model inference execution duration in seconds",
    ["model_name"]
)

idip_rag_pipeline_duration_seconds = _get_or_create_histogram(
    "idip_rag_pipeline_duration_seconds",
    "RAG pipeline execution duration in seconds"
)

idip_reranker_duration_seconds = _get_or_create_histogram(
    "idip_reranker_duration_seconds",
    "Reranker processing duration in seconds"
)

# --- GAUGES ---

idip_model_confidence_score = _get_or_create_gauge(
    "idip_model_confidence_score",
    "Recent average model confidence scores",
    ["model_name"]
)

idip_drift_score = _get_or_create_gauge(
    "idip_drift_score",
    "Latest computed feature and concept drift scores",
    ["feature_name"]
)

idip_active_workers = _get_or_create_gauge(
    "idip_active_workers",
    "Current active background worker counts"
)

idip_queue_depth = _get_or_create_gauge(
    "idip_queue_depth",
    "Current count of enqueued Celery background tasks",
    ["queue_name"]
)

# --- A/B TESTING METRICS ---

idip_ab_queries_total = _get_or_create_counter(
    "idip_ab_queries_total",
    "Total A/B test queries by variant and status",
    ["experiment_id", "variant", "status"]
)

idip_ab_latency_seconds = _get_or_create_histogram(
    "idip_ab_latency_seconds",
    "A/B test latency in seconds by variant",
    ["experiment_id", "variant"]
)

idip_ab_cache_hits_total = _get_or_create_counter(
    "idip_ab_cache_hits_total",
    "Total A/B test cache hits by variant",
    ["experiment_id", "variant"]
)

idip_ab_cache_misses_total = _get_or_create_counter(
    "idip_ab_cache_misses_total",
    "Total A/B test cache misses by variant",
    ["experiment_id", "variant"]
)

idip_ab_response_quality = _get_or_create_gauge(
    "idip_ab_response_quality",
    "A/B test response quality scores by variant",
    ["experiment_id", "variant"]
)

