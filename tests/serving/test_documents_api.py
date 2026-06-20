"""Integration tests for the new IDIP documents management endpoints."""
import json
import base64
import hmac
import hashlib
import time
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import MagicMock, patch

from serving.api import app
from serving.dependencies import get_db_session, get_vector_store

def create_mock_jwt(api_key: str, secret: str = "idip_secret_key_1234567890", expired: bool = False) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    exp_time = time.time() - 3600 if expired else time.time() + 3600
    payload = {"api_key": api_key, "exp": exp_time}
    
    def b64_url_encode(d: dict) -> str:
        s = json.dumps(d)
        return base64.urlsafe_b64encode(s.encode("utf-8")).decode("utf-8").rstrip("=")
        
    h_b64 = b64_url_encode(header)
    p_b64 = b64_url_encode(payload)
    
    signing_input = f"{h_b64}.{p_b64}".encode("utf-8")
    sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).decode("utf-8").rstrip("=")
    
    return f"{h_b64}.{p_b64}.{sig_b64}"

@pytest.fixture
def valid_auth_headers():
    token = create_mock_jwt("test_client_key")
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture(autouse=True)
def setup_app_state():
    app.state.ner_service = MagicMock()
    app.state.classifier_service = MagicMock()
    app.state.vision_analyzer = MagicMock()
    app.state.llm_service = MagicMock()
    app.state.embedding_service = MagicMock()
    app.state.vector_store = MagicMock()
    app.state.ensemble_router = MagicMock()
    app.state.guardrail_checker = MagicMock()
    
    # Clear overrides after each test
    yield
    app.dependency_overrides.clear()

@pytest.mark.asyncio
async def test_list_documents(valid_auth_headers):
    """Verify that GET /v1/documents lists, filters, and paginates correctly."""
    mock_db = MagicMock()
    
    # Set up mock execute results for count and select queries
    mock_scalar = MagicMock()
    mock_scalar.scalar.return_value = 1
    
    # We will mock mock_db.execute to return different results depending on the query string
    def dynamic_execute(sql, params=None):
        sql_str = str(sql)
        mock_result = MagicMock()
        if "COUNT" in sql_str:
            mock_result.scalar.return_value = 1
        elif "feature_store" in sql_str:
            mock_result.fetchall.return_value = [("doc-123", "invoice")]
        else:
            mock_result.fetchall.return_value = [("doc-123", "s3://demo/invoice.pdf", "completed", "2026-06-19 12:00:00")]
        return mock_result

    mock_db.execute.side_effect = dynamic_execute
    app.dependency_overrides[get_db_session] = lambda: mock_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get(
            "/v1/documents?page=1&limit=20&status=completed&source_type=pdf",
            headers=valid_auth_headers
        )
        
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert data["total"] == 1
    assert len(data["documents"]) == 1
    doc = data["documents"][0]
    assert doc["doc_id"] == "doc-123"
    assert doc["filename"] == "invoice.pdf"
    assert doc["source_type"] == "pdf"
    assert doc["status"] == "completed"
    assert doc["doc_type"] == "invoice"


@pytest.mark.asyncio
async def test_delete_document(valid_auth_headers):
    """Verify that DELETE /v1/documents/{doc_id} deletes data from DB and vector store."""
    mock_db = MagicMock()
    mock_db.execute.return_value.fetchone.return_value = ("doc-123",)

    mock_vector_store = MagicMock()
    mock_vector_store.delete.return_value = True

    app.dependency_overrides[get_db_session] = lambda: mock_db
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.delete(
            "/v1/documents/doc-123",
            headers=valid_auth_headers
        )
        
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "deleted"
    assert data["doc_id"] == "doc-123"
    mock_vector_store.delete.assert_called_once_with("doc-123")


@pytest.mark.asyncio
async def test_bulk_delete_documents(valid_auth_headers):
    """Verify that POST /v1/documents/bulk-delete deletes multiple items."""
    mock_db = MagicMock()
    mock_vector_store = MagicMock()

    app.dependency_overrides[get_db_session] = lambda: mock_db
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/documents/bulk-delete",
            headers=valid_auth_headers,
            json={"doc_ids": ["doc-1", "doc-2"]}
        )
        
    assert response.status_code == 200
    data = response.json()
    assert data["deleted"] == 2


@pytest.mark.asyncio
@patch("serving.tasks.process_document.delay")
async def test_reprocess_document(mock_celery_delay, valid_auth_headers):
    """Verify that POST /v1/documents/{doc_id}/reprocess triggers background reprocessing task."""
    mock_db = MagicMock()
    mock_db.execute.return_value.fetchone.return_value = ("s3://demo/invoice.pdf",)

    app.dependency_overrides[get_db_session] = lambda: mock_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/documents/doc-123/reprocess",
            headers=valid_auth_headers
        )
        
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "reprocessing_queued"
    assert data["doc_id"] == "doc-123"
    mock_celery_delay.assert_called_once_with("doc-123", "s3://demo/invoice.pdf")


@pytest.mark.asyncio
async def test_document_pipeline_status_stream(valid_auth_headers):
    """Verify that GET /v1/documents/{doc_id}/status yields event stream progress."""
    mock_db = MagicMock()
    # Returns queued then complete
    mock_db.execute.return_value.fetchone.side_effect = [
        ("processing", None),
        ("processing", None),
        ("completed", None),
        ("completed", None),
        ("completed", None),
        ("completed", None)
    ]

    app.dependency_overrides[get_db_session] = lambda: mock_db

    with patch("asyncio.sleep", return_value=None):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get(
                "/v1/documents/doc-123/status",
                headers=valid_auth_headers
            )
            
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        
        content = response.text
        lines = content.split("\n\n")
        events = [line.replace("data: ", "") for line in lines if line.startswith("data: ")]
        
        assert len(events) > 0
        first_event = json.loads(events[0])
        assert first_event["step"] == "Received"
