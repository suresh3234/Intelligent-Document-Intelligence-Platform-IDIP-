import os
import pytest
import numpy as np
import threading
from unittest.mock import MagicMock, patch
from preprocessing.models import TextChunk
from rag.vector_store import (
    SearchResult,
    FAISSBackend,
    PineconeBackend,
    WeaviateBackend,
    VectorStoreFactory
)
from config import settings

@pytest.fixture
def dummy_chunks_and_embeddings():
    """Generates 10 chunks and corresponding mock embeddings of dimension 1024."""
    chunks = [
        TextChunk(
            chunk_id=f"c-{i}",
            doc_id="doc-1" if i < 5 else "doc-2",
            chunk_index=i,
            text=f"Sample text block {i}.",
            token_count=10,
            char_start=i * 50,
            char_end=i * 50 + 50,
            chunk_strategy="fixed",
            metadata={
                "source_type": "pdf" if i < 5 else "api",
                "language": "en" if i % 2 == 0 else "fr",
                "doc_type_signal": "contract" if i < 5 else "invoice"
            }
        )
        for i in range(10)
    ]
    embeddings = [np.random.randn(1024).astype(np.float32) for _ in range(10)]
    return chunks, embeddings

# --- FAISS Backend Tests (Using Real FAISS) ---

def test_faiss_backend_crud(tmp_path, dummy_chunks_and_embeddings):
    chunks, embeddings = dummy_chunks_and_embeddings
    index_dir = str(tmp_path / "faiss_index")
    
    # 1. Initialize
    backend = FAISSBackend(dimension=1024, index_type="FlatIP", index_dir=index_dir)
    
    # 2. Upsert
    result = backend.upsert(chunks, embeddings)
    assert result.success is True
    assert result.upserted_count == 10
    
    # 3. Query stats
    stats = backend.get_stats()
    assert stats.total_vectors == 10
    
    # 4. Search Query (Exact search FlatIP)
    # Search closest vector to the first chunk's vector
    query_vector = embeddings[0]
    hits = backend.query(query_vector, top_k=3, filters={})
    assert len(hits) == 3
    assert hits[0].chunk_id == "c-0"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)  # Exact match for normalized vector
    
    # 5. Search with metadata filtering
    # Search with filter language='fr' (should match c-1, c-3, c-5, c-7, c-9)
    hits_filter = backend.query(query_vector, top_k=5, filters={"language": "fr"})
    for hit in hits_filter:
        assert hit.metadata["language"] == "fr"
        
    # Search with filter source_type='api' (should match doc-2 chunks)
    hits_api = backend.query(query_vector, top_k=5, filters={"source_type": "api"})
    for hit in hits_api:
        assert hit.metadata["source_type"] == "api"
        assert hit.doc_id == "doc-2"

    # 6. Delete
    # Delete doc-1 (first 5 chunks)
    del_ok = backend.delete("doc-1")
    assert del_ok is True
    
    # Stats check after deletion
    stats_post = backend.get_stats()
    assert stats_post.total_vectors == 5
    
    # Ensure c-0 is deleted and c-5 remains
    hits_post = backend.query(query_vector, top_k=5, filters={})
    for hit in hits_post:
        assert hit.doc_id == "doc-2"
        assert int(hit.chunk_id.split("-")[-1]) >= 5

def test_faiss_persistence(tmp_path, dummy_chunks_and_embeddings):
    chunks, embeddings = dummy_chunks_and_embeddings
    index_dir = str(tmp_path / "faiss_persist")
    
    # Create and write index to disk
    backend1 = FAISSBackend(dimension=1024, index_type="HNSWFlat", index_dir=index_dir)
    backend1.upsert(chunks[:5], embeddings[:5])
    
    # Verify files created on disk
    assert os.path.exists(os.path.join(index_dir, "index.faiss"))
    assert os.path.exists(os.path.join(index_dir, "id_map.json"))
    
    # Initialize a second backend pointing to same directory (loads from disk)
    backend2 = FAISSBackend(dimension=1024, index_type="HNSWFlat", index_dir=index_dir)
    stats = backend2.get_stats()
    assert stats.total_vectors == 5
    
    # Query loaded index
    hits = backend2.query(embeddings[0], top_k=1, filters={})
    assert len(hits) == 1
    assert hits[0].chunk_id == "c-0"

def test_faiss_thread_safety(tmp_path, dummy_chunks_and_embeddings):
    chunks, embeddings = dummy_chunks_and_embeddings
    index_dir = str(tmp_path / "faiss_threads")
    backend = FAISSBackend(dimension=1024, index_type="FlatIP", index_dir=index_dir)
    
    # Run concurrent reads and writes
    def writer_task():
        for i in range(5):
            backend.upsert([chunks[i]], [embeddings[i]])
            
    def reader_task():
        for _ in range(5):
            _ = backend.query(embeddings[0], top_k=2, filters={})

    threads = []
    for _ in range(3):
        threads.append(threading.Thread(target=writer_task))
        threads.append(threading.Thread(target=reader_task))
        
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
    # Stats check
    stats = backend.get_stats()
    assert stats.total_vectors > 0

# --- Pinecone Backend Tests (Mocked) ---

@patch("rag.vector_store.Pinecone")
def test_pinecone_backend(mock_pc_cls, dummy_chunks_and_embeddings):
    chunks, embeddings = dummy_chunks_and_embeddings
    
    # Mock Pinecone instance & Index
    mock_pc = MagicMock()
    mock_index = MagicMock()
    mock_index.describe_index_stats.return_value = {"total_vector_count": 42}
    mock_index.query.return_value = {
        "matches": [
            {"id": "c-0", "score": 0.95, "metadata": {"doc_id": "doc-1", "text": "Sample text."}}
        ]
    }
    
    mock_pc.Index.return_value = mock_index
    mock_pc_cls.return_value = mock_pc
    
    # Instantiate
    backend = PineconeBackend(api_key="fake-key", environment="us-west1")
    
    # Upsert
    result = backend.upsert(chunks, embeddings)
    assert result.success is True
    assert result.upserted_count == 10
    
    # Assert namespaces logic. We have 5 chunks with source_type='pdf' and 5 chunks with source_type='api'
    # Therefore, Index.upsert should be called twice (one for namespace 'pdf', one for namespace 'api')
    assert mock_index.upsert.call_count == 2
    
    # Search
    hits = backend.query(embeddings[0], top_k=2, filters={"source_type": "pdf"})
    assert len(hits) == 1
    assert hits[0].chunk_id == "c-0"
    
    # Delete
    del_ok = backend.delete("doc-1")
    assert del_ok is True
    assert mock_index.delete.call_count > 0
    
    # Stats
    stats = backend.get_stats()
    assert stats.total_vectors == 42
    assert stats.backend_type == "pinecone"

# --- Weaviate Backend Tests (Mocked) ---

@patch("rag.vector_store.weaviate")
def test_weaviate_backend(mock_weaviate_mod, dummy_chunks_and_embeddings):
    chunks, embeddings = dummy_chunks_and_embeddings
    
    # Mock Weaviate Client
    mock_client = MagicMock()
    mock_client.schema.exists.return_value = False
    
    # Context manager mock for client.batch
    mock_batch = MagicMock()
    mock_batch.__enter__.return_value = mock_batch
    mock_client.batch.return_value = mock_batch
    
    # Mock client.query.get
    mock_query_builder = MagicMock()
    mock_query_builder.with_tenant.return_value = mock_query_builder
    mock_query_builder.with_where.return_value = mock_query_builder
    mock_query_builder.with_hybrid.return_value = mock_query_builder
    mock_query_builder.with_limit.return_value = mock_query_builder
    mock_query_builder.with_additional.return_value = mock_query_builder
    mock_query_builder.do.return_value = {
        "data": {
            "Get": {
                "IDIPChunk": [
                    {
                        "chunk_id": "c-0",
                        "doc_id": "doc-1",
                        "text": "Sample text.",
                        "_additional": {"score": 0.88}
                    }
                ]
            }
        }
    }
    mock_client.query.get.return_value = mock_query_builder
    mock_weaviate_mod.Client.return_value = mock_client
    
    # Instantiate
    backend = WeaviateBackend(url="http://mock-weaviate:8080")
    
    # Verify schema creation
    mock_client.schema.create_class.assert_called_once()
    
    # Upsert with tenant
    result = backend.upsert(chunks, embeddings)
    assert result.success is True
    assert result.upserted_count == 10
    
    # Verify tenant added in schema (default_tenant if none provided)
    mock_client.schema.add_class_tenants.assert_called()
    
    # Query hybrid
    hits = backend.query(embeddings[0], top_k=2, filters={"tenant_id": "client-abc", "text_query": "contract"})
    assert len(hits) == 1
    assert hits[0].chunk_id == "c-0"
    assert hits[0].score == 0.88
    
    # Delete
    del_ok = backend.delete("doc-1")
    assert del_ok is True

# --- Factory Tests ---

def test_vector_store_factory():
    # 1. FAISS config test
    with patch.object(settings, "VECTOR_DB_PROVIDER", "faiss"):
        with patch.object(settings, "ENVIRONMENT", "development"):
            backend = VectorStoreFactory.get_vector_store()
            assert isinstance(backend, FAISSBackend)
            assert backend.index_type == "FlatIP"
            
        with patch.object(settings, "ENVIRONMENT", "production"):
            backend = VectorStoreFactory.get_vector_store()
            assert isinstance(backend, FAISSBackend)
            assert backend.index_type == "HNSWFlat"

    # 2. Unsupported provider
    with patch.object(settings, "VECTOR_DB_PROVIDER", "unsupported-provider"):
        with pytest.raises(ValueError):
            VectorStoreFactory.get_vector_store()
