import sqlite3
import numpy as np
from preprocessing.feature_store import FeatureStore
from preprocessing.embeddings import EmbeddingService
from rag.vector_store import VectorStoreFactory

doc_id = "f90c9b39-ef70-477d-ba04-65ba1d6d1f2c"
query = "what are the skills does he have?"

# 1. Encode query
import redis
try:
    r_client = redis.Redis(host="localhost", port=6379)
    r_client.ping()
except Exception:
    r_client = None

embedding_service = EmbeddingService(redis_client=r_client)
query_vector = embedding_service.encode_query(query)
print("Query Vector Norm:", np.linalg.norm(query_vector))
print("Query Vector Shape:", query_vector.shape)

# 2. Query FAISS directly
vector_store = VectorStoreFactory.get_vector_store()
print("FAISS ntotal:", vector_store.index.ntotal)

norm = np.linalg.norm(query_vector)
norm_emb = query_vector / norm if norm > 0 else query_vector
query_np = np.array([norm_emb], dtype=np.float32)

scores, indices = vector_store.index.search(query_np, min(1000, vector_store.index.ntotal))
print("FAISS raw search indices:", indices[0])
print("FAISS raw search scores:", scores[0])

# Match to id_map
for idx_val in indices[0]:
    if idx_val == -1:
        continue
    faiss_id = str(idx_val)
    chunk_data = vector_store.id_map.get(faiss_id)
    if chunk_data:
        print(f"Index {idx_val}: doc_id={chunk_data.get('doc_id')}, text_len={len(chunk_data.get('text'))}")
        # Check filters
        filters = {"doc_id": doc_id}
        matched = True
        meta = chunk_data.get("metadata", {})
        for fk, fv in filters.items():
            if fk in chunk_data:
                print(f"  Check '{fk}': chunk_data={chunk_data[fk]} vs filter={fv} (equal: {chunk_data[fk] == fv})")
            elif fk in meta:
                print(f"  Check '{fk}': meta={meta[fk]} vs filter={fv} (equal: {meta[fk] == fv})")
            else:
                print(f"  Check '{fk}': Not found in chunk_data or meta!")

# Run via interface
matches = vector_store.query(query_vector, top_k=5, filters={"doc_id": doc_id})
print("Result matches count:", len(matches))
