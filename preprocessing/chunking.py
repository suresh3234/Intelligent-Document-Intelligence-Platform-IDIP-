import uuid
from typing import List, Optional, Dict, Any
import numpy as np
import tiktoken
import nltk
from nltk.tokenize import sent_tokenize

# Dynamic setup of NLTK resources
for res in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{res}")
    except LookupError:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass

from preprocessing.models import TextChunk
from preprocessing.cleaning import TextCleaner
from ingestion.models import IngestedDocument

class ChunkingPipeline:
    """Orchestrates document cleaning and chunking according to selected strategy."""
    
    def __init__(self, embedding_service: Optional[Any] = None, tokenizer_name: str = "cl100k_base"):
        self.cleaner = TextCleaner()
        self.encoding = tiktoken.get_encoding(tokenizer_name)
        self.embedding_service = embedding_service

    def chunk_document(
        self, 
        doc: IngestedDocument, 
        strategy: str = "fixed", 
        chunk_size: int = 512, 
        overlap: int = 64, 
        similarity_threshold: float = 0.75
    ) -> List[TextChunk]:
        """
        Cleans IngestedDocument and splits raw_text into chunks using specified strategy.
        Supported strategies: 'fixed', 'semantic', 'sentence_window'.
        """
        cleaned_text = self.cleaner.clean(doc.raw_text)
        if not cleaned_text:
            return []

        if strategy == "fixed":
            return self._chunk_fixed(doc.doc_id, cleaned_text, chunk_size, overlap)
        elif strategy == "semantic":
            return self._chunk_semantic(doc.doc_id, cleaned_text, similarity_threshold)
        elif strategy == "sentence_window":
            return self._chunk_sentence_window(doc.doc_id, cleaned_text)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    def _chunk_fixed(self, doc_id: str, text: str, chunk_size: int, overlap: int) -> List[TextChunk]:
        """Strategy A: Fixed-size chunking (respecting sentence boundaries)."""
        sentences = sent_tokenize(text)
        if not sentences:
            return []

        chunks: List[TextChunk] = []
        chunk_index = 0
        
        # Calculate tokens per sentence
        sentence_tokens = [len(self.encoding.encode(s)) for s in sentences]
        
        i = 0
        n_sentences = len(sentences)
        
        while i < n_sentences:
            current_chunk_sentences: List[str] = []
            current_tokens = 0
            
            start_i = i
            # Keep adding sentences until chunk_size is exceeded
            while i < n_sentences and current_tokens + sentence_tokens[i] <= chunk_size:
                current_chunk_sentences.append(sentences[i])
                current_tokens += sentence_tokens[i]
                i += 1
            
            # Corner case: Single sentence exceeds chunk_size limit
            if not current_chunk_sentences and i < n_sentences:
                current_chunk_sentences.append(sentences[i])
                current_tokens += sentence_tokens[i]
                i += 1

            chunk_text = " ".join(current_chunk_sentences)
            char_start = text.find(current_chunk_sentences[0])
            char_end = char_start + len(chunk_text)
            
            is_table = self.cleaner.is_table_chunk(chunk_text)
            
            chunks.append(TextChunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                chunk_index=chunk_index,
                text=chunk_text,
                token_count=current_tokens,
                char_start=char_start,
                char_end=char_end,
                chunk_strategy="fixed",
                metadata={"is_table": is_table}
            ))
            chunk_index += 1

            # Slicing next loop starting point back to support overlap
            if i < n_sentences:
                overlap_tokens = 0
                backtrack_count = 0
                # Scan backwards to get target overlap, ensuring we do not backtrack past start_i
                for idx in range(i - 1, start_i, -1):
                    if overlap_tokens + sentence_tokens[idx] > overlap:
                        break
                    overlap_tokens += sentence_tokens[idx]
                    backtrack_count += 1
                
                i = max(i - backtrack_count, start_i + 1)

        return chunks

    def _chunk_semantic(self, doc_id: str, text: str, similarity_threshold: float) -> List[TextChunk]:
        """Strategy B: Semantic chunking."""
        sentences = sent_tokenize(text)
        if not sentences:
            return []

        # If only 1 sentence, return as single chunk
        if len(sentences) == 1:
            tokens = len(self.encoding.encode(sentences[0]))
            return [TextChunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                chunk_index=0,
                text=sentences[0],
                token_count=tokens,
                char_start=0,
                char_end=len(sentences[0]),
                chunk_strategy="semantic",
                metadata={"is_table": self.cleaner.is_table_chunk(sentences[0])}
            )]

        # Generate sentence embeddings
        if self.embedding_service:
            embeddings = self.embedding_service.encode(sentences)
        else:
            # Fallback to random embeddings for isolation if no service provided
            embeddings = [np.random.rand(384) for _ in sentences]

        # Calculate cosine similarities between consecutive sentences
        similarities = []
        for j in range(len(sentences) - 1):
            emb1 = embeddings[j]
            emb2 = embeddings[j + 1]
            dot = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            sim = float(dot / (norm1 * norm2)) if (norm1 > 0 and norm2 > 0) else 0.0
            similarities.append(sim)

        # Slice sentences into semantic groups based on similarity threshold
        groups: List[List[str]] = [[sentences[0]]]
        for j, sim in enumerate(similarities):
            if sim < similarity_threshold:
                # Start a new semantic group
                groups.append([sentences[j + 1]])
            else:
                groups[-1].append(sentences[j + 1])

        # Merge groups that are smaller than 128 tokens
        merged_groups: List[List[str]] = []
        current_group: List[str] = []
        current_tokens = 0

        for group in groups:
            group_text = " ".join(group)
            group_tokens = len(self.encoding.encode(group_text))
            
            if not current_group:
                current_group = group
                current_tokens = group_tokens
            elif current_tokens < 128:
                # Merge with current group
                current_group.extend(group)
                current_tokens += group_tokens
            else:
                merged_groups.append(current_group)
                current_group = group
                current_tokens = group_tokens
        
        if current_group:
            merged_groups.append(current_group)

        # Build chunks, capping groups larger than 768 tokens (split using fixed-size logic)
        chunks: List[TextChunk] = []
        chunk_index = 0

        for group in merged_groups:
            group_text = " ".join(group)
            group_tokens = len(self.encoding.encode(group_text))

            if group_tokens > 768:
                # Cap / split large group using fixed-size logic (chunk_size=512, overlap=64)
                sub_chunks = self._chunk_fixed(doc_id, group_text, chunk_size=512, overlap=64)
                for sc in sub_chunks:
                    sc.chunk_index = chunk_index
                    sc.chunk_strategy = "semantic"
                    chunks.append(sc)
                    chunk_index += 1
            else:
                char_start = text.find(group[0])
                char_end = char_start + len(group_text)
                
                chunks.append(TextChunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    text=group_text,
                    token_count=group_tokens,
                    char_start=char_start,
                    char_end=char_end,
                    chunk_strategy="semantic",
                    metadata={"is_table": self.cleaner.is_table_chunk(group_text)}
                ))
                chunk_index += 1

        return chunks

    def _chunk_sentence_window(self, doc_id: str, text: str) -> List[TextChunk]:
        """Strategy C: Sentence window chunking (context size N=3)."""
        sentences = sent_tokenize(text)
        if not sentences:
            return []

        chunks: List[TextChunk] = []
        n_sentences = len(sentences)

        for idx in range(n_sentences):
            base_sentence = sentences[idx]
            
            # Select surrounding sentences context window (prev 3, current, next 3)
            start_win = max(0, idx - 3)
            end_win = min(n_sentences, idx + 4)
            window_sentences = sentences[start_win:end_win]
            
            window_text = " ".join(window_sentences)
            token_count = len(self.encoding.encode(window_text))
            
            # Calculate char offsets based on window text location in the original text
            char_start = text.find(window_sentences[0])
            char_end = char_start + len(window_text)
            
            metadata = {
                "base_sentence": base_sentence,
                "context_window": window_sentences,
                "is_table": self.cleaner.is_table_chunk(window_text)
            }

            chunks.append(TextChunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                chunk_index=idx,
                text=window_text,
                token_count=token_count,
                char_start=char_start,
                char_end=char_end,
                chunk_strategy="sentence_window",
                metadata=metadata
            ))

        return chunks
