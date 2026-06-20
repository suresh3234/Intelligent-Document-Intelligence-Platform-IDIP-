import hashlib
import pandas as pd
from great_expectations.dataset.pandas_dataset import PandasDataset
import langdetect
from typing import Any, List

from ingestion.models import IngestedDocument
from ingestion.exceptions import ValidationError, QualityGateError
from ingestion.deduplication import DeduplicationService

class ValidationPipeline:
    """Quality gate validation pipeline implementing schema, dedup, lang, and content quality checks."""
    
    def __init__(self, redis_client: Any, allowed_languages: List[str] = None):
        self.dedup_service = DeduplicationService(redis_client)
        self.allowed_languages = allowed_languages or ["en", "fr", "de", "es", "hi", "zh"]

    async def validate(self, doc: IngestedDocument) -> None:
        """
        Executes sequential checks on IngestedDocument.
        Raises ValidationError or DuplicateDocumentError on failure.
        """
        # --- Check 1: Schema Validation (Great Expectations) ---
        await self._check_schema(doc)

        # --- Check 2: Deduplication ---
        await self.dedup_service.check_and_register(
            source_uri=doc.source_uri,
            checksum=doc.checksum,
            doc_id=doc.doc_id
        )

        # --- Check 3: Language Detection ---
        await self._check_language(doc)

        # --- Check 4: Content Quality ---
        await self._check_content_quality(doc)

    async def self_check_schema(self, doc: IngestedDocument) -> None:
        """Helper to run schema check externally."""
        await self._check_schema(doc)

    async def _check_schema(self, doc: IngestedDocument) -> None:
        # Verify checksum matches SHA-256 of raw_bytes
        if doc.raw_bytes:
            computed_checksum = hashlib.sha256(doc.raw_bytes).hexdigest()
            if doc.checksum != computed_checksum:
                raise ValidationError(f"Checksum mismatch: expected {doc.checksum}, computed {computed_checksum}")
        
        # Load into Pandas DataFrame for Great Expectations validation
        df = pd.DataFrame([{
            "raw_text": doc.raw_text,
            "byte_size": doc.byte_size,
            "language": doc.language
        }])
        
        ge_df = PandasDataset(df)
        
        # Great Expectations validation rules
        null_check = ge_df.expect_column_values_to_not_be_null("raw_text")
        if not null_check.success:
            raise ValidationError("Schema check failed: raw_text column must not be null.")
            
        len_check = ge_df.expect_column_value_lengths_to_be_between("raw_text", min_value=51)
        if not len_check.success:
            raise ValidationError("Schema check failed: raw_text length must be greater than 50 characters.")
            
        size_check = ge_df.expect_column_values_to_be_between("byte_size", min_value=100, max_value=52_428_800)
        if not size_check.success:
            raise ValidationError("Schema check failed: byte_size must be between 100B and 50MB.")
            
        lang_check = ge_df.expect_column_values_to_be_in_set("language", self.allowed_languages)
        if not lang_check.success:
            raise ValidationError(f"Schema check failed: language '{doc.language}' not in allowed list {self.allowed_languages}.")

    async def _check_language(self, doc: IngestedDocument) -> None:
        try:
            # Run langdetect on text
            langs = langdetect.detect_langs(doc.raw_text)
            if not langs:
                raise ValidationError("Language detection yielded no results.")
                
            top_lang = langs[0]
            
            # If confidence < 0.85, flag in metadata
            if top_lang.prob < 0.85:
                doc.metadata["uncertain_language"] = True
                
            # If detected language not in allowed list, raise validation error
            if top_lang.lang not in self.allowed_languages:
                raise ValidationError(
                    f"Language detection failed: detected language '{top_lang.lang}' "
                    f"(prob: {top_lang.prob:.2f}) is not in allowed list {self.allowed_languages}"
                )
                
            # Update the document's language code with detected language
            doc.language = top_lang.lang
            
        except langdetect.lang_detect_exception.LangDetectException as e:
            raise ValidationError(f"Language detection library error: {str(e)}") from e

    async def _check_content_quality(self, doc: IngestedDocument) -> None:
        text = doc.raw_text
        if not text.strip():
            raise QualityGateError("Quality gate failed: document contains only whitespace characters.")

        # Detect blank/whitespace-only pages
        pages = text.split("--- PAGE ")
        # If split occurred, first index might be empty prefix
        if len(pages) > 1:
            for p in pages:
                if not p.strip():
                    continue
                # Strip out headers like "1 ---\n"
                content = p.split("---", 1)[-1].strip() if "---" in p else p.strip()
                # Exclude page template lines
                if not content:
                    raise QualityGateError("Quality gate failed: blank/whitespace-only page detected.")

        # Flag scanned-but-unreadable documents (avg word length < 2)
        words = text.split()
        if not words:
            raise QualityGateError("Quality gate failed: document has no words.")
            
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 2:
            raise QualityGateError(
                f"Quality gate failed: scanned-but-unreadable document. "
                f"Average word length ({avg_word_len:.2f}) is less than 2."
            )

        # Score text quality: ratio of alpha characters to total characters > 0.6
        # Strip whitespaces and punctuation to compare strictly content if needed?
        # Standard: "ratio of alpha chars to total chars > 0.6"
        total_chars = len(text)
        alpha_chars = sum(1 for c in text if c.isalpha())
        
        alpha_ratio = alpha_chars / total_chars if total_chars > 0 else 0
        if alpha_ratio <= 0.6:
            raise QualityGateError(
                f"Quality gate failed: low text quality ratio. "
                f"Ratio of alpha characters to total characters ({alpha_ratio:.2f}) is not > 0.6."
            )
        
        # Inject computed quality score in metadata
        doc.metadata["quality_score"] = float(alpha_ratio)
