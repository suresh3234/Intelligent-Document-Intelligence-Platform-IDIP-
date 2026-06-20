import pytest
import json
from unittest.mock import patch
from ingestion.adapters.database import DatabaseAdapter
from ingestion.exceptions import AdapterError

@pytest.mark.asyncio
async def test_database_adapter_insert_cdc():
    adapter = DatabaseAdapter("postgresql+asyncpg://user:pass@localhost/db")
    
    cdc_payload = {
        "op": "c",
        "before": None,
        "after": {
            "id": "contract-abc-123",
            "title": "Ingested SLA Agreement",
            "body": "This contract details SLA requirements..."
        },
        "source": {
            "connector": "postgresql",
            "schema": "public",
            "table": "contracts"
        }
    }
    
    doc = await adapter.ingest(
        source_uri="postgresql://public/contracts/contract-abc-123",
        cdc_payload=cdc_payload
    )
    
    assert doc.source_type == "database"
    assert doc.metadata["db_operation"] == "c"
    assert doc.metadata["db_table"] == "contracts"
    assert doc.metadata["record_id"] == "contract-abc-123"
    assert "Ingested SLA Agreement" in doc.raw_text
    assert doc.doc_id is not None  # Derived deterministically using UUID5

@pytest.mark.asyncio
async def test_database_adapter_delete_cdc():
    adapter = DatabaseAdapter("postgresql+asyncpg://user:pass@localhost/db")
    
    cdc_payload = {
        "op": "d",
        "before": {
            "id": "invoice-789",
            "total": 1200.50
        },
        "after": None,
        "source": {
            "connector": "postgresql",
            "schema": "billing",
            "table": "invoices"
        }
    }
    
    doc = await adapter.ingest(
        source_uri="postgresql://billing/invoices/invoice-789",
        cdc_payload=cdc_payload
    )
    
    assert doc.source_type == "database"
    assert doc.metadata["db_operation"] == "d"
    assert doc.metadata["record_id"] == "invoice-789"
    assert "DELETED RECORD" in doc.raw_text
    assert "invoice-789" in doc.raw_text

@pytest.mark.asyncio
async def test_database_adapter_missing_payload():
    adapter = DatabaseAdapter("postgresql+asyncpg://user:pass@localhost/db")
    with pytest.raises(AdapterError):
        await adapter.ingest("postgresql://test")
