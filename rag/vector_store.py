from abc import ABC, abstractmethod
import os
import json
import logging
import threading
from typing import List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel, Field

# Lazy-import libraries to fail gracefully if selected but not installed
try:
    import faiss
except ImportError:
    faiss = None

try:
    from pinecone import Pinecone
except ImportError:
    Pinecone = None

try:
    import weaviate
except ImportError:
    weaviate = None

from preprocessing.models import TextChunk
from config import settings

logger = logging.getLogger("idip.rag.vector_store")

# --- Models ---

class SearchResult(BaseModel):
    """Encapsulates a vector search match result."""
    chunk_id: str
    doc_id: str
    score: float
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    rank: int

class UpsertResult(BaseModel):
    """Contains the status of an upsert transaction."""
    success: bool
    upserted_count: int

class VectorStoreStats(BaseModel):
    """Database statistics overview."""
    total_vectors: int
    index_name: str
    backend_type: str

# --- Interface ---

class VectorStoreInterface(ABC):
    """Abstract interface defining required vector database operations."""

    @abstractmethod
    def upsert(self, chunks: List[TextChunk], embeddings: List[np.ndarray]) -> UpsertResult:
        """Upsert a list of document chunks and their corresponding embedding vectors."""
        pass

    @abstractmethod
    def query(self, embedding: np.ndarray, top_k: int, filters: Dict[str, Any]) -> List[SearchResult]:
        """Perform similarity query on embedding vector, returning top_k matched chunks."""
        pass

    @abstractmethod
    def delete(self, doc_id: str) -> bool:
        """Deletes all vector properties matching the given document ID."""
        pass

    @abstractmethod
    def get_stats(self) -> VectorStoreStats:
        """Queries metrics about the current state of the vector store index."""
        pass

# --- 1. Pinecone Backend ---

class PineconeBackend(VectorStoreInterface):
    """Pinecone vector database driver utilizing namespace logical isolation."""

    def __init__(self, api_key: Optional[str] = None, environment: Optional[str] = None):
        if Pinecone is None:
            raise ImportError("pinecone-client is not installed in the system environment.")
        
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY", "mock-pinecone-api-key")
        self.environment = environment or os.environ.get("PINECONE_ENVIRONMENT", "us-west1-gcp")
        self.pc = Pinecone(api_key=self.api_key)
        self.index_name = f"idip-{settings.ENVIRONMENT.lower()}"
        self.index = self.pc.Index(self.index_name)

    def upsert(self, chunks: List[TextChunk], embeddings: List[np.ndarray]) -> UpsertResult:
        # Group items by source_type to assign namespaces
        namespaces: Dict[str, List[tuple]] = {}
        for i, chunk in enumerate(chunks):
            ns = chunk.metadata.get("source_type") or "default"
            namespaces.setdefault(ns, []).append((i, chunk))

        total_upserted = 0
        for ns, items in namespaces.items():
            # Batches of 100 as per Pinecone recommendation/limitations
            for start in range(0, len(items), 100):
                batch_items = items[start : start + 100]
                vectors = []
                for idx, chunk in batch_items:
                    meta = {
                        "doc_id": chunk.doc_id,
                        "text": chunk.text,
                        "chunk_index": int(chunk.chunk_index),
                        "token_count": int(chunk.token_count),
                        "char_start": int(chunk.char_start),
                        "char_end": int(chunk.char_end),
                        "chunk_strategy": chunk.chunk_strategy,
                    }
                    
                    # Store required metadata filters in the Pinecone payload
                    for key in ("language", "doc_type_signal", "source_type"):
                        if key in chunk.metadata:
                            meta[key] = chunk.metadata[key]
                        elif key == "source_type":
                            meta[key] = ns

                    if "ingestion_ts" in chunk.metadata:
                        meta["ingestion_ts"] = str(chunk.metadata["ingestion_ts"])

                    vectors.append({
                        "id": chunk.chunk_id,
                        "values": embeddings[idx].tolist(),
                        "metadata": meta
                    })

                self.index.upsert(vectors=vectors, namespace=ns)
                total_upserted += len(vectors)

        return UpsertResult(success=True, upserted_count=total_upserted)

    def query(self, embedding: np.ndarray, top_k: int, filters: Dict[str, Any]) -> List[SearchResult]:
        filters_copy = dict(filters)
        ns = filters_copy.pop("source_type", "default")

        resp = self.index.query(
            vector=embedding.tolist(),
            top_k=top_k,
            filter=filters_copy,
            namespace=ns,
            include_metadata=True
        )

        results = []
        for rank, match in enumerate(resp.get("matches", [])):
            meta = match.get("metadata", {})
            results.append(SearchResult(
                chunk_id=match["id"],
                doc_id=meta.get("doc_id", ""),
                score=match.get("score", 0.0),
                text=meta.get("text", ""),
                metadata=meta,
                rank=rank + 1
            ))
        return results

    def delete(self, doc_id: str) -> bool:
        # Check standard default source types and default namespaces to wipe
        namespaces = ["pdf", "image", "api", "stream", "database", "default"]
        for ns in namespaces:
            try:
                self.index.delete(filter={"doc_id": doc_id}, namespace=ns)
            except Exception as e:
                logger.debug(f"Failed deleting doc_id {doc_id} in namespace {ns}: {e}")
        return True

    def get_stats(self) -> VectorStoreStats:
        stats = self.index.describe_index_stats()
        return VectorStoreStats(
            total_vectors=stats.get("total_vector_count", 0),
            index_name=self.index_name,
            backend_type="pinecone"
        )

# --- 2. Weaviate Backend ---

class WeaviateBackend(VectorStoreInterface):
    """Weaviate vector database driver supporting dynamic tenant isolation."""

    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        if weaviate is None:
            raise ImportError("weaviate-client is not installed in the system environment.")
        
        self.url = url or os.environ.get("WEAVIATE_URL", "http://localhost:8080")
        self.api_key = api_key or os.environ.get("WEAVIATE_API_KEY")
        
        auth = weaviate.AuthApiKey(api_key=self.api_key) if self.api_key else None
        self.client = weaviate.Client(url=self.url, auth_client_secret=auth)
        self.class_name = "IDIPChunk"
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Sets up the IDIPChunk schema in Weaviate with appropriate indices and multi-tenancy."""
        if not self.client.schema.exists(self.class_name):
            class_obj = {
                "class": self.class_name,
                "vectorizer": "none",
                "multiTenancyConfig": {"enabled": True},
                "vectorIndexConfig": {
                    "ef": 200,
                    "efConstruction": 128,
                    "maxConnections": 64
                },
                "properties": [
                    {"name": "chunk_id", "dataType": ["text"]},
                    {"name": "doc_id", "dataType": ["text"]},
                    {"name": "text", "dataType": ["text"]},
                    {"name": "chunk_index", "dataType": ["int"]},
                    {"name": "token_count", "dataType": ["int"]},
                    {"name": "char_start", "dataType": ["int"]},
                    {"name": "char_end", "dataType": ["int"]},
                    {"name": "page_number", "dataType": ["int"]},
                    {"name": "section_heading", "dataType": ["text"]},
                    {"name": "chunk_strategy", "dataType": ["text"]},
                    {"name": "source_type", "dataType": ["text"]},
                    {"name": "language", "dataType": ["text"]},
                    {"name": "doc_type_signal", "dataType": ["text"]}
                ]
            }
            self.client.schema.create_class(class_obj)

    def upsert(self, chunks: List[TextChunk], embeddings: List[np.ndarray]) -> UpsertResult:
        if not chunks:
            return UpsertResult(success=True, upserted_count=0)

        # Deduce tenant ID
        tenant = chunks[0].metadata.get("tenant_id") or chunks[0].metadata.get("client_id") or "default_tenant"
        
        # Dynamically register the tenant inside Weaviate schema
        try:
            self.client.schema.add_class_tenants(
                class_name=self.class_name,
                tenants=[weaviate.Tenant(name=tenant)]
            )
        except Exception as e:
            # Tenant already exists or error ignored in mock scenarios
            logger.debug(f"Tenant registration skip/error: {e}")

        # Batch upsert objects
        with self.client.batch(batch_size=100) as batch:
            for chunk, emb in zip(chunks, embeddings, strict=True):
                properties = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "chunk_index": int(chunk.chunk_index),
                    "token_count": int(chunk.token_count),
                    "char_start": int(chunk.char_start),
                    "char_end": int(chunk.char_end),
                    "page_number": int(chunk.page_number) if chunk.page_number else 0,
                    "section_heading": chunk.section_heading or "",
                    "chunk_strategy": chunk.chunk_strategy,
                    "source_type": chunk.metadata.get("source_type") or "default",
                    "language": chunk.metadata.get("language") or "en",
                    "doc_type_signal": chunk.metadata.get("doc_type_signal") or "other"
                }
                batch.add_data_object(
                    data_object=properties,
                    class_name=self.class_name,
                    vector=emb.tolist(),
                    tenant=tenant
                )

        return UpsertResult(success=True, upserted_count=len(chunks))

    def query(self, embedding: np.ndarray, top_k: int, filters: Dict[str, Any]) -> List[SearchResult]:
        filters_copy = dict(filters)
        tenant = filters_copy.pop("tenant_id", None) or filters_copy.pop("client_id", None) or "default_tenant"
        text_query = filters_copy.pop("text_query", "")

        query_builder = self.client.query.get(
            self.class_name,
            [
                "chunk_id", "doc_id", "text", "chunk_index", "token_count",
                "char_start", "char_end", "page_number", "section_heading",
                "chunk_strategy", "source_type", "language", "doc_type_signal"
            ]
        )
        query_builder = query_builder.with_tenant(tenant)

        # Build equal filter mapping
        if filters_copy:
            operands = []
            for k, v in filters_copy.items():
                operands.append({
                    "path": [k],
                    "operator": "Equal",
                    "valueText": str(v)
                })
            if len(operands) == 1:
                query_builder = query_builder.with_where(operands[0])
            elif operands:
                query_builder = query_builder.with_where({
                    "operator": "And",
                    "operands": operands
                })

        # Hybrid Search: alpha=0.7 specifies Dense search dominance (Dense=0.7, Keyword BM25=0.3)
        query_builder = query_builder.with_hybrid(
            query=text_query,
            alpha=0.7,
            vector=embedding.tolist()
        )
        query_builder = query_builder.with_limit(top_k)
        query_builder = query_builder.with_additional(["id", "score"])

        resp = query_builder.do()
        data = resp.get("data", {}).get("Get", {}).get(self.class_name, [])

        results = []
        for rank, item in enumerate(data):
            add_info = item.get("_additional", {})
            score = add_info.get("score") or 0.0
            if isinstance(score, str):
                score = float(score)

            meta = {k: v for k, v in item.items() if k not in ("chunk_id", "doc_id", "text", "_additional")}

            results.append(SearchResult(
                chunk_id=item.get("chunk_id", ""),
                doc_id=item.get("doc_id", ""),
                score=score,
                text=item.get("text", ""),
                metadata=meta,
                rank=rank + 1
            ))

        return results

    def delete(self, doc_id: str) -> bool:
        # Multi-tenancy delete call on default tenant
        try:
            self.client.batch.delete_objects(
                class_name=self.class_name,
                where={
                    "path": ["doc_id"],
                    "operator": "Equal",
                    "valueText": doc_id
                },
                tenant="default_tenant"
            )
        except Exception as e:
            logger.debug(f"Failed deleting doc_id {doc_id} on default_tenant: {e}")
        return True

    def get_stats(self) -> VectorStoreStats:
        return VectorStoreStats(
            total_vectors=-1,
            index_name=self.class_name,
            backend_type="weaviate"
        )

# --- 3. FAISS Backend ---

class FAISSBackend(VectorStoreInterface):
    """Local, thread-safe FAISS backend database utilizing disk serialization and post-filtering."""

    def __init__(
        self,
        dimension: int = 1024,
        index_type: str = "FlatIP",
        index_dir: str = "data/faiss_index"
    ):
        if faiss is None:
            raise ImportError("faiss library is not installed in the system environment.")
        
        self.dimension = dimension
        self.index_type = index_type
        self.index_dir = index_dir
        self.index_path = os.path.join(self.index_dir, "index.faiss")
        self.map_path = os.path.join(self.index_dir, "id_map.json")
        self.lock = threading.RLock()
        
        self.index = None
        self.id_map: Dict[str, Dict[str, Any]] = {}
        self._last_mtime = 0.0
        
        os.makedirs(self.index_dir, exist_ok=True)
        self._load_or_create()

    def _load_or_create(self) -> None:
        with self.lock:
            if os.path.exists(self.index_path) and os.path.exists(self.map_path):
                try:
                    self.index = faiss.read_index(self.index_path)
                    with open(self.map_path, "r", encoding="utf-8") as f:
                        self.id_map = json.load(f)
                    self._last_mtime = os.path.getmtime(self.index_path)
                    logger.info("Successfully loaded local FAISS index & mappings.")
                    return
                except Exception as e:
                    logger.warning(f"Failed reading FAISS files: {e}. Re-initializing index.")

            # Create standard Index Flat or approximate HNSW Flat
            if self.index_type == "HNSWFlat":
                # M=32 connections, efConstruction=128, efSearch=64
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)
                self.index.hnsw.efSearch = 64
                self.index.hnsw.efConstruction = 128
            else:
                self.index = faiss.IndexFlatIP(self.dimension)
            self.id_map = {}
            self._last_mtime = 0.0

    def _save(self) -> None:
        with self.lock:
            faiss.write_index(self.index, self.index_path)
            with open(self.map_path, "w", encoding="utf-8") as f:
                json.dump(self.id_map, f, ensure_ascii=False, indent=2)
            if os.path.exists(self.index_path):
                self._last_mtime = os.path.getmtime(self.index_path)

    def upsert(self, chunks: List[TextChunk], embeddings: List[np.ndarray]) -> UpsertResult:
        with self.lock:
            vectors = []
            for emb in embeddings:
                # Normalise for cosine inner product matches
                norm = np.linalg.norm(emb)
                norm_emb = emb / norm if norm > 0 else emb
                vectors.append(norm_emb.astype(np.float32))

            if not vectors:
                return UpsertResult(success=True, upserted_count=0)

            vectors_np = np.vstack(vectors)
            start_idx = self.index.ntotal
            self.index.add(vectors_np)

            # Map the FAISS vector index IDs to Pydantic metadata
            for i, chunk in enumerate(chunks):
                faiss_id = str(start_idx + i)
                meta = dict(chunk.metadata)
                meta["source_type"] = chunk.metadata.get("source_type") or "default"
                meta["language"] = chunk.metadata.get("language") or "en"
                meta["doc_type_signal"] = chunk.metadata.get("doc_type_signal") or "other"
                if "ingestion_ts" in meta:
                    meta["ingestion_ts"] = str(meta["ingestion_ts"])

                self.id_map[faiss_id] = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "chunk_index": int(chunk.chunk_index),
                    "token_count": int(chunk.token_count),
                    "char_start": int(chunk.char_start),
                    "char_end": int(chunk.char_end),
                    "chunk_strategy": chunk.chunk_strategy,
                    "metadata": meta
                }

            self._save()
            return UpsertResult(success=True, upserted_count=len(chunks))

    def query(self, embedding: np.ndarray, top_k: int, filters: Dict[str, Any]) -> List[SearchResult]:
        # Normalise vector
        norm = np.linalg.norm(embedding)
        norm_emb = embedding / norm if norm > 0 else embedding
        query_np = np.array([norm_emb], dtype=np.float32)

        with self.lock:
            # Check if index files on disk have changed since last load
            if os.path.exists(self.index_path):
                mtime = os.path.getmtime(self.index_path)
                if self._last_mtime < mtime:
                    logger.info("FAISS index file changed on disk. Reloading in query call...")
                    self._load_or_create()

            total = self.index.ntotal
            if total == 0:
                return []

            # Retrieve larger subset for post-filtering
            k_search = min(1000, total)
            scores, indices = self.index.search(query_np, k_search)
            
            raw_scores = scores[0]
            raw_indices = indices[0]

            results = []
            rank = 1
            for score, idx_val in zip(raw_scores, raw_indices, strict=True):
                if idx_val == -1:
                    continue
                faiss_id = str(idx_val)
                chunk_data = self.id_map.get(faiss_id)
                if not chunk_data:
                    continue

                # Filter assertions
                matched = True
                meta = chunk_data.get("metadata", {})
                for fk, fv in filters.items():
                    if fk in chunk_data:
                        if chunk_data[fk] != fv:
                            matched = False
                            break
                    elif fk in meta:
                        if meta[fk] != fv:
                            matched = False
                            break
                    else:
                        matched = False
                        break

                if not matched:
                    continue

                results.append(SearchResult(
                    chunk_id=chunk_data["chunk_id"],
                    doc_id=chunk_data["doc_id"],
                    score=float(score),
                    text=chunk_data["text"],
                    metadata=meta,
                    rank=rank
                ))
                rank += 1
                if len(results) >= top_k:
                    break

            return results

    def delete(self, doc_id: str) -> bool:
        with self.lock:
            vectors_to_keep = []
            metadata_to_keep = []
            
            # Extract elements we keep, rebuilding the FAISS index
            for faiss_id_str, chunk_data in self.id_map.items():
                if chunk_data["doc_id"] == doc_id:
                    continue
                
                faiss_id = int(faiss_id_str)
                vec = self.index.reconstruct(faiss_id)
                vectors_to_keep.append(vec)
                metadata_to_keep.append(chunk_data)

            if self.index_type == "HNSWFlat":
                new_index = faiss.IndexHNSWFlat(self.dimension, 32)
                new_index.hnsw.efSearch = 64
                new_index.hnsw.efConstruction = 128
            else:
                new_index = faiss.IndexFlatIP(self.dimension)

            new_id_map = {}
            if vectors_to_keep:
                vectors_np = np.vstack(vectors_to_keep).astype(np.float32)
                new_index.add(vectors_np)
                for i, c_data in enumerate(metadata_to_keep):
                    new_id_map[str(i)] = c_data

            self.index = new_index
            self.id_map = new_id_map
            self._save()
            return True

    def get_stats(self) -> VectorStoreStats:
        with self.lock:
            return VectorStoreStats(
                total_vectors=self.index.ntotal,
                index_name=self.index_path,
                backend_type="faiss"
            )

# --- Factory ---

class VectorStoreFactory:
    """Factory creating the appropriate Vector Store backend matching application settings."""

    @staticmethod
    def get_vector_store() -> VectorStoreInterface:
        provider = settings.VECTOR_DB_PROVIDER.lower()
        if provider not in ("pinecone", "weaviate", "faiss"):
            raise ValueError(f"Unsupported VECTOR_DB_PROVIDER provider config: {settings.VECTOR_DB_PROVIDER}")
        
        # Check if environment is development or mock key configurations are in use
        is_development = settings.ENVIRONMENT.lower() in ("dev", "development")
        is_mock_pinecone = os.environ.get("PINECONE_API_KEY", "mock-").startswith("mock-")
        is_mock_weaviate = os.environ.get("WEAVIATE_API_KEY", "mock-").startswith("mock-") or "localhost" in os.environ.get("WEAVIATE_URL", "")
        
        if is_development or is_mock_pinecone or is_mock_weaviate or provider == "faiss":
            # Default to local FAISS backend for seamless development setup
            index_type = "FlatIP" if is_development else "HNSWFlat"
            return FAISSBackend(index_type=index_type)

        if provider == "pinecone":
            return PineconeBackend()
        elif provider == "weaviate":
            return WeaviateBackend()
        elif provider == "faiss":
            index_type = "FlatIP" if is_development else "HNSWFlat"
            return FAISSBackend(index_type=index_type)
        else:
            raise ValueError(f"Unsupported VECTOR_DB_PROVIDER provider config: {settings.VECTOR_DB_PROVIDER}")
