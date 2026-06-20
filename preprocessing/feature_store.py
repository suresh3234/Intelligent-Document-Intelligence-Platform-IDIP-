import datetime
import logging
from typing import Dict, Any, Optional
from sqlalchemy import create_engine, text
from config import settings

logger = logging.getLogger("idip.preprocessing.feature_store")

# Type cast mapping for feature store retrieval
FEATURE_CASTS = {
    "page_count": int,
    "avg_words_per_page": float,
    "has_tables": lambda v: v == "True",
    "has_images": lambda v: v == "True",
    "language": str,
    "doc_type_signal": str,
    "reading_level": float,
    "entity_density": float,
    "text_quality_score": float
}

class FeatureStore:
    """
    Feature Store with PostgreSQL backend.
    Computes, registers, and retrieves analytical document properties.
    """
    
    def __init__(self, db_url: Optional[str] = None):
        if not db_url:
            db_url = (
                f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
                f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
            )
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self._initialize_table()

    def _initialize_table(self) -> None:
        """Initializes the database schema if the table does not exist."""
        # Using dialect-agnostic column definitions
        query = text("""
            CREATE TABLE IF NOT EXISTS feature_store (
                doc_id VARCHAR(255) NOT NULL,
                feature_name VARCHAR(100) NOT NULL,
                feature_value VARCHAR(255) NOT NULL,
                computed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (doc_id, feature_name)
            )
        """)
        try:
            with self.engine.begin() as conn:
                conn.execute(query)
            logger.info("FeatureStore table check/initialization completed successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize FeatureStore table: {e}")
            raise

    def set(self, doc_id: str, features: Dict[str, Any]) -> None:
        """
        Stores computed features for a document.
        Uses UPSERT syntax (ON CONFLICT DO UPDATE) to prevent key collisions.
        """
        if not features:
            return

        now = datetime.datetime.utcnow()
        dialect_name = self.engine.dialect.name
        
        # Dialect specific UPSERT statements
        if dialect_name in ("postgresql", "sqlite"):
            upsert_query = text("""
                INSERT INTO feature_store (doc_id, feature_name, feature_value, computed_at)
                VALUES (:doc_id, :feature_name, :feature_value, :computed_at)
                ON CONFLICT (doc_id, feature_name) DO UPDATE SET
                    feature_value = EXCLUDED.feature_value,
                    computed_at = EXCLUDED.computed_at
            """)
        else:
            # Fallback standard INSERT or REPLACE if using other test drivers
            upsert_query = text("""
                REPLACE INTO feature_store (doc_id, feature_name, feature_value, computed_at)
                VALUES (:doc_id, :feature_name, :feature_value, :computed_at)
            """)

        try:
            with self.engine.begin() as conn:
                for k, v in features.items():
                    # Serialize values to string
                    val_str = str(v)
                    conn.execute(
                        upsert_query,
                        {
                            "doc_id": doc_id,
                            "feature_name": k,
                            "feature_value": val_str,
                            "computed_at": now
                        }
                    )
            logger.info(f"Successfully wrote {len(features)} features for doc_id {doc_id}.")
        except Exception as e:
            logger.error(f"Failed to set features for doc_id {doc_id} in Postgres: {e}")
            raise

    def get(self, doc_id: str) -> Dict[str, Any]:
        """
        Retrieves all features stored for a given doc_id.
        Automatically casts values back to their expected Python types.
        """
        query = text("""
            SELECT feature_name, feature_value 
            FROM feature_store 
            WHERE doc_id = :doc_id
        """)
        
        raw_features = {}
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(query, {"doc_id": doc_id}).fetchall()
                for row in rows:
                    name, val_str = row[0], row[1]
                    raw_features[name] = val_str
        except Exception as e:
            logger.error(f"Failed to query features for doc_id {doc_id}: {e}")
            raise

        if not raw_features:
            return {}

        # Apply correct types
        typed_features = {}
        for k, v in raw_features.items():
            if k in FEATURE_CASTS:
                try:
                    typed_features[k] = FEATURE_CASTS[k](v)
                except Exception as e:
                    logger.warning(f"Failed casting feature '{k}' value '{v}': {e}. Returning as string.")
                    typed_features[k] = v
            else:
                typed_features[k] = v

        return typed_features
