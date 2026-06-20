import re
import time
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import langdetect
import tiktoken

from preprocessing.models import TextChunk
from rag.models import Citation, RAGResponse
from rag.vector_store import VectorStoreInterface, SearchResult
from rag.reranker import CrossEncoderReranker
from rag.prompt_templates import get_template_by_doc_type
from config import settings

logger = logging.getLogger("idip.rag.pipeline")

class RAGPipeline:
    """
    Coordinating pipeline for Retrieval-Augmented Generation (RAG).
    Handles preprocessing, HyDE query expansion, reranking, citation assembly, and generation.
    """

    def __init__(
        self,
        vector_store: VectorStoreInterface,
        embedding_service: Any,
        reranker: Optional[CrossEncoderReranker] = None,
        llm_client: Optional[Any] = None
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.reranker = reranker or CrossEncoderReranker()
        self.llm_client = llm_client
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def execute(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> RAGResponse:
        """
        Executes the end-to-end RAG pipeline.
        Returns a structured RAGResponse envelope.
        """
        start_time = time.time()
        filters = filters or {}
        
        # --- Step 1: Query Preprocessing ---
        # 1.1 Language Detection
        try:
            query_lang = langdetect.detect(query)
        except Exception:
            query_lang = "en"
            
        # 1.2 Query Expansion (HyDE)
        hyde_queries = []
        if self.llm_client is not None:
            logger.info("Generating hypothetical documents using HyDE...")
            hyde_prompt = f"Write a brief hypothetical paragraph answering this search query: {query}"
            for _ in range(3):
                try:
                    hypothetical_answer = self.llm_client.generate(hyde_prompt)
                    if hypothetical_answer:
                        hyde_queries.append(hypothetical_answer.strip())
                except Exception as e:
                    logger.warning(f"Failed generating HyDE query: {e}")

        # 1.3 Embed original query and HyDE documents
        logger.info("Encoding queries and hypothetical documents...")
        emb_vectors = [self.embedding_service.encode_query(query)]
        for hq in hyde_queries:
            emb_vectors.append(self.embedding_service.encode_single(hq))

        # 1.4 Compute average query vector & L2 normalize
        mean_vector = np.mean(emb_vectors, axis=0)
        norm = np.linalg.norm(mean_vector)
        query_vector = mean_vector / norm if norm > 0 else mean_vector

        # --- Step 2: Retrieval ---
        # 2.1 Over-retrieval (top_k * 3) for reranking
        logger.info(f"Retrieving top_{top_k * 3} candidate chunks from Vector Store...")
        candidates = self.vector_store.query(query_vector, top_k=top_k * 3, filters=filters)
        
        # 2.2 Deduplicate by doc_id (keep highest-scoring chunk per doc)
        seen_docs = set()
        deduped_candidates: List[SearchResult] = []
        for cand in candidates:
            if cand.doc_id not in seen_docs:
                seen_docs.add(cand.doc_id)
                deduped_candidates.append(cand)

        # --- Step 3: Reranking ---
        reranked_chunks: List[SearchResult] = []
        if deduped_candidates:
            logger.info(f"Reranking {len(deduped_candidates)} candidates using Cross-Encoder...")
            candidate_texts = [cand.text for cand in deduped_candidates]
            reranker_scores = self.reranker.score_pairs(query, candidate_texts)
            
            scored_candidates = []
            for cand, score in zip(deduped_candidates, reranker_scores, strict=True):
                cand.score = score  # Override store score with reranker score
                scored_candidates.append(cand)
            
            # Sort by reranker score descending
            scored_candidates.sort(key=lambda x: x.score, reverse=True)
            
            has_doc_filter = filters and any(k in filters for k in ["doc_id", "source_uri"])
            if has_doc_filter:
                reranked_chunks = scored_candidates[:top_k]
            else:
                filtered = [c for c in scored_candidates if c.score > settings.RERANKER_THRESHOLD]
                if not filtered and scored_candidates:
                    reranked_chunks = scored_candidates[:1]
                else:
                    reranked_chunks = filtered[:top_k]

        # --- Step 4: Context Building ---
        context_str = ""
        final_chunks_for_prompt = []
        tokens_used = 0
        
        # Tiktoken truncation limit: 3072 tokens
        for chunk in reranked_chunks:
            citation_marker = f"[DOC-{chunk.doc_id[:8]}]: "
            block = f"{citation_marker}{chunk.text}\n---\n"
            block_tokens = len(self.tokenizer.encode(block))
            
            if tokens_used + block_tokens > 3072:
                logger.info("Context length exceeded 3072 tokens. Truncating remainder.")
                break
                
            context_str += block
            tokens_used += block_tokens
            final_chunks_for_prompt.append(chunk)

        # Render prompt template matching document type signal
        primary_doc_type = "other"
        if final_chunks_for_prompt:
            primary_doc_type = final_chunks_for_prompt[0].metadata.get("doc_type_signal") or "other"
            
        template = get_template_by_doc_type(primary_doc_type)
        prompt = template.render(chunks=final_chunks_for_prompt, query=query)

        # --- Step 5: Generation ---
        answer = "I cannot find the answer in the provided documents."
        confidence = 0.0
        
        if self.llm_client is not None and final_chunks_for_prompt:
            logger.info("Invoking LLM for answer generation...")
            try:
                raw_generation = self.llm_client.generate(prompt)
                answer, confidence = self._parse_llm_output(raw_generation)
            except Exception as e:
                logger.error(f"Failed during LLM query generation: {e}")
                answer = "Error generating answer from LLM endpoint."
                confidence = 0.0

        # --- Step 6: Response Packaging & Citation Extraction ---
        citations = self._extract_citations(answer, final_chunks_for_prompt)
        
        latency_ms = (time.time() - start_time) * 1000
        low_confidence_flag = confidence < settings.CONFIDENCE_THRESHOLD

        return RAGResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
            reranked_chunks=reranked_chunks,
            hyde_queries=hyde_queries,
            total_latency_ms=float(round(latency_ms, 2)),
            low_confidence=low_confidence_flag
        )

    def _parse_llm_output(self, raw_text: str) -> tuple[str, float]:
        """Parses the generated LLM text extracting the Answer and Confidence Score."""
        answer = ""
        confidence = 1.0
        
        # Split on standard headers
        if "Answer:" in raw_text:
            parts = raw_text.split("Answer:", 1)[1]
            if "Confidence Score:" in parts:
                answer_part, conf_part = parts.split("Confidence Score:", 1)
                answer = answer_part.strip()
                try:
                    confidence = float(conf_part.strip())
                except ValueError:
                    confidence = 1.0
            else:
                answer = parts.strip()
        else:
            answer = raw_text.strip()
            
        return answer, confidence

    def _extract_citations(self, answer: str, source_chunks: List[SearchResult]) -> List[Citation]:
        """Parses the generated answer to construct structured Citation references."""
        # Find all cited document tag markers e.g. [DOC-abcdef12] or [DOC-acme-inv]
        matches = re.findall(r"\[DOC-([a-f0-9a-zA-Z_-]{2,16})\]", answer, re.IGNORECASE)
        unique_matches = set(matches)
        
        citations = []
        for short_id in unique_matches:
            # Match to source chunk
            matched_chunk = None
            for chunk in source_chunks:
                if chunk.doc_id[:8].lower() == short_id.lower():
                    matched_chunk = chunk
                    break
                    
            if matched_chunk:
                source_uri = (
                    matched_chunk.metadata.get("source_uri") or 
                    matched_chunk.metadata.get("uri") or 
                    ""
                )
                citations.append(Citation(
                    doc_id=matched_chunk.doc_id,
                    doc_id_short=short_id.lower(),
                    source_uri=source_uri,
                    text_snippet=matched_chunk.text
                ))
                
        return citations
