import numpy as np
from preprocessing.embeddings import EmbeddingService
from rag.vector_store import VectorStoreFactory
from rag.reranker import CrossEncoderReranker
from config import settings
import redis

doc_id = "6bdd5720-e66f-4bc8-91e6-f669e7ba76fd"
query = "what are the skills does he have?"

try:
    r_client = redis.Redis(host="localhost", port=6379)
    r_client.ping()
except Exception:
    r_client = None

embedding_service = EmbeddingService(redis_client=r_client)
query_vector = embedding_service.encode_query(query)

vector_store = VectorStoreFactory.get_vector_store()
candidates = vector_store.query(query_vector, top_k=15, filters={"doc_id": doc_id})

print("Found", len(candidates), "candidates")
if candidates:
    texts = [cand.text for cand in candidates]
    reranker = CrossEncoderReranker()
    scores = reranker.score_pairs(query, texts)
    for cand, score in zip(candidates, scores):
        print(f"Chunk ID: {cand.chunk_id}")
        print(f"Reranker Score: {score}")
        print(f"Snippet: {cand.text[:200]}")
        print(f"Is above threshold ({settings.RERANKER_THRESHOLD}): {score > settings.RERANKER_THRESHOLD}")
        print("-" * 50)
