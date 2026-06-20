import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from rag.models import RAGResponse, Citation
from rag.vector_store import SearchResult
from rag.reranker import CrossEncoderReranker
from rag.pipeline import RAGPipeline
from rag.prompt_templates import get_template_by_doc_type
from config import settings

@pytest.fixture
def mock_embedding_service():
    service = MagicMock()
    service.encode_query.return_value = np.ones((1024,), dtype=np.float32)
    service.encode_single.return_value = np.ones((1024,), dtype=np.float32)
    return service

@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    # Return 3 search results (representing doc-1, doc-2, doc-3)
    store.query.return_value = [
        SearchResult(
            chunk_id="c-0",
            doc_id="abcdef12-1234-5678-abcd-ef1234567890",
            score=0.9,
            text="Extractable information about sales taxes in the invoice.",
            metadata={"source_type": "pdf", "doc_type_signal": "invoice", "source_uri": "s3://billing/inv1.pdf"},
            rank=1
        ),
        SearchResult(
            chunk_id="c-1",
            doc_id="7890abcd-1234-5678-abcd-ef1234567890",
            score=0.8,
            text="This agreement specifies legal terms and conditions.",
            metadata={"source_type": "api", "doc_type_signal": "contract", "source_uri": "https://api.docs/contract1"},
            rank=2
        ),
        SearchResult(
            chunk_id="c-2",
            doc_id="11223344-1234-5678-abcd-ef1234567890",
            score=0.7,
            text="Standard company report summary for Q1 2026.",
            metadata={"source_type": "stream", "doc_type_signal": "report", "source_uri": "kafka://report-stream"},
            rank=3
        )
    ]
    return store

@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    
    # Custom generate behavior
    def mock_generate(prompt):
        if "Write a brief hypothetical paragraph" in prompt:
            # HyDE expansion generation
            return "Hypothetical answer document text."
        else:
            # Final RAG generation
            # Cite first chunk (abcdef12) and second chunk (7890abcd)
            return (
                "Answer: According to the invoice, we paid sales taxes [DOC-abcdef12]. "
                "The legal terms are listed in the agreement [DOC-7890abcd].\n"
                "Confidence Score: 0.85"
            )
            
    client.generate.side_effect = mock_generate
    return client

# --- Reranker Tests ---

@patch("rag.reranker.CrossEncoder")
def test_reranker_lazy_and_scoring(mock_cross_encoder_cls):
    mock_instance = MagicMock()
    mock_instance.predict.return_value = [0.8, 0.2]
    mock_cross_encoder_cls.return_value = mock_instance
    
    reranker = CrossEncoderReranker()
    scores = reranker.score_pairs("query", ["doc 1", "doc 2"])
    
    assert scores == [0.8, 0.2]
    mock_cross_encoder_cls.assert_called_once()

# --- Prompt Selection Tests ---

def test_prompt_templates_selection():
    # Invoice template
    t_inv = get_template_by_doc_type("invoice")
    assert "invoice and billing document audits" in t_inv.render(chunks=[], query="test")
    
    # Contract template
    t_con = get_template_by_doc_type("contract")
    assert "contract and agreement reviews" in t_con.render(chunks=[], query="test")
    
    # Default template
    t_def = get_template_by_doc_type("other")
    assert "Intelligent Document Intelligence Platform" in t_def.render(chunks=[], query="test")

# --- End-to-End Pipeline Tests (Mocked LLM & Embeddings) ---

@patch("rag.pipeline.CrossEncoderReranker")
def test_rag_pipeline_execution(
    mock_reranker_cls,
    mock_vector_store,
    mock_embedding_service,
    mock_llm_client
):
    # Setup mock reranker to return descending scores above threshold
    mock_reranker = MagicMock()
    mock_reranker.score_pairs.return_value = [0.95, 0.82, 0.35]  # Last is below threshold (0.4)
    mock_reranker_cls.return_value = mock_reranker
    
    pipeline = RAGPipeline(
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service,
        reranker=mock_reranker,
        llm_client=mock_llm_client
    )
    
    response = pipeline.execute(query="What is sales tax and contract terms?", top_k=2)
    
    # --- Assertions ---
    assert isinstance(response, RAGResponse)
    
    # 1. Answer text check
    assert "sales taxes" in response.answer
    assert "legal terms" in response.answer
    
    # 2. HyDE queries generated check
    assert len(response.hyde_queries) == 3
    assert response.hyde_queries[0] == "Hypothetical answer document text."
    
    # 3. Vector query over-retrieval check (top_k * 3 = 6)
    mock_vector_store.query.assert_called_once()
    args, kwargs = mock_vector_store.query.call_args
    assert kwargs["top_k"] == 6
    
    # 4. Reranker execution check
    # Should keep only 2 elements (0.95 and 0.82) after threshold check and top_k limiting
    assert len(response.reranked_chunks) == 2
    assert response.reranked_chunks[0].chunk_id == "c-0"
    assert response.reranked_chunks[0].score == 0.95
    assert response.reranked_chunks[1].chunk_id == "c-1"
    assert response.reranked_chunks[1].score == 0.82
    
    # 5. Citations extraction check
    # Generated text cited [DOC-abcdef12] and [DOC-7890abcd]
    assert len(response.citations) == 2
    
    cit_doc1 = next(c for c in response.citations if c.doc_id_short == "abcdef12")
    assert cit_doc1.doc_id == "abcdef12-1234-5678-abcd-ef1234567890"
    assert cit_doc1.source_uri == "s3://billing/inv1.pdf"
    assert "sales taxes" in cit_doc1.text_snippet
    
    cit_doc2 = next(c for c in response.citations if c.doc_id_short == "7890abcd")
    assert cit_doc2.doc_id == "7890abcd-1234-5678-abcd-ef1234567890"
    assert cit_doc2.source_uri == "https://api.docs/contract1"
    assert "legal terms" in cit_doc2.text_snippet
    
    # 6. Confidence and latency checks
    assert response.confidence == 0.85
    assert response.low_confidence is False  # 0.85 >= settings.CONFIDENCE_THRESHOLD (0.75)
    assert response.total_latency_ms > 0.0

@patch("rag.pipeline.CrossEncoderReranker")
def test_rag_pipeline_low_confidence(
    mock_reranker_cls,
    mock_vector_store,
    mock_embedding_service,
    mock_llm_client
):
    # Setup mock reranker
    mock_reranker = MagicMock()
    mock_reranker.score_pairs.return_value = [0.9, 0.5, 0.3]
    mock_reranker_cls.return_value = mock_reranker
    
    # Return low confidence generated response
    mock_llm_client.generate.side_effect = lambda prompt: (
        "Answer: Standard response. [DOC-abcdef12]\n"
        "Confidence Score: 0.55"
    )
    
    pipeline = RAGPipeline(
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service,
        reranker=mock_reranker,
        llm_client=mock_llm_client
    )
    
    response = pipeline.execute(query="Test low confidence query", top_k=1)
    
    assert response.confidence == 0.55
    assert response.low_confidence is True  # 0.55 < settings.CONFIDENCE_THRESHOLD (0.75)
