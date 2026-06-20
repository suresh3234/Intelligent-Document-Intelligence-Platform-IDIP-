import logging
from typing import List
import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger("idip.rag.reranker")

class CrossEncoderReranker:
    """
    Cross-Encoder reranking model wrapper.
    Scores relevance of (query, document_chunk) pairs to refine retrieval quality.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> CrossEncoder:
        """Lazily instantiates the CrossEncoder model to optimize startup time and memory."""
        if self._model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading CrossEncoder reranker model '{self.model_name}' on device '{device}'...")
            self._model = CrossEncoder(self.model_name, device=device)
        return self._model

    def score_pairs(self, query: str, texts: List[str]) -> List[float]:
        """
        Computes relevance scores for list of text chunks against a given query.
        Returns a list of float scores.
        """
        if not texts:
            return []

        pairs = [[query, text] for text in texts]
        try:
            scores = self.model.predict(pairs, show_progress_bar=False)
            # Standardize output list shape
            if isinstance(scores, float):
                return [scores]
            return [float(s) for s in scores]
        except Exception as e:
            logger.error(f"Failed scoring query-chunk pairs during reranking: {e}")
            raise
