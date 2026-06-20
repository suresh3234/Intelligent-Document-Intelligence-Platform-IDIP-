import re
import nltk
from typing import Dict, Any, Optional
from ingestion.models import IngestedDocument

def compute_page_count(text: str, doc_page_count: Optional[int] = None) -> int:
    """Calculates the document page count based on page delimiters or metadata."""
    page_markers = re.findall(r'--- PAGE \d+ ---', text)
    if page_markers:
        return len(page_markers)
    if doc_page_count is not None and doc_page_count > 0:
        return doc_page_count
    return 1

def compute_avg_words_per_page(text: str, page_count: int) -> float:
    """Calculates the average number of words per page."""
    words = text.split()
    total_words = len(words)
    effective_pages = page_count if page_count > 0 else 1
    return float(round(total_words / effective_pages, 2))

def check_has_tables(text: str) -> bool:
    """Checks if the document contains tables heuristically."""
    lines = text.split("\n")
    for line in lines:
        if line.count("\t") >= 2 or line.count("|") >= 3:
            return True
    return False

def check_has_images(metadata: Dict[str, Any], source_type: str) -> bool:
    """Checks if the document contains images based on metadata or source type."""
    if source_type == "image":
        return True
    if metadata:
        # Check standard metadata keys for image extraction flags
        for key in ("has_images", "has_image", "image_extracted", "images"):
            if key in metadata and metadata[key]:
                return True
    return False

def detect_doc_type_signal(text: str) -> str:
    """Heuristically detects document type (invoice, contract, report, email, other) using keyword signals."""
    text_lower = text.lower()
    
    # 1. Email check
    if "subject:" in text_lower and ("to:" in text_lower or "from:" in text_lower or "cc:" in text_lower):
        return "email"
        
    # 2. Invoice check
    invoice_keywords = ["invoice", "receipt", "billing", "amount due", "payable", "purchase order", "po number", "total due"]
    invoice_score = sum(1 for kw in invoice_keywords if kw in text_lower)
    
    # 3. Contract check
    contract_keywords = ["contract", "agreement", "lease", "shall agree", "hereby", "signatories", "confidentiality", "indemnification", "parties"]
    contract_score = sum(1 for kw in contract_keywords if kw in text_lower)
    
    # 4. Report check
    report_keywords = ["report", "annual", "quarterly", "executive summary", "introduction", "conclusion", "methodology", "analysis", "findings"]
    report_score = sum(1 for kw in report_keywords if kw in text_lower)
    
    scores = {
        "invoice": invoice_score,
        "contract": contract_score,
        "report": report_score
    }
    
    max_type = max(scores, key=scores.get)
    if scores[max_type] > 0:
        return max_type
        
    return "other"

def count_syllables(word: str) -> int:
    """Calculates number of syllables in a word using basic English vowel heuristics."""
    word = word.lower()
    # Remove non-alphabet characters
    word = re.sub(r'[^a-z]', '', word)
    if not word:
        return 0
        
    vowels = "aeiouy"
    count = 0
    if word[0] in vowels:
        count += 1
        
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
            
    if word.endswith("e"):
        count -= 1
        
    if count <= 0:
        count = 1
        
    return count

def calculate_reading_level(text: str) -> float:
    """
    Computes the Flesch-Kincaid Grade Level score:
    0.39 * (total words / total sentences) + 11.8 * (total syllables / total words) - 15.59
    """
    if not text.strip():
        return 0.0
        
    try:
        sentences = nltk.tokenize.sent_tokenize(text)
    except Exception:
        # Fallback split if punkt not downloaded
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        
    words = text.split()
    
    num_sentences = len(sentences)
    num_words = len(words)
    
    if num_sentences == 0:
        num_sentences = 1
    if num_words == 0:
        return 0.0
        
    total_syllables = sum(count_syllables(w) for w in words)
    
    grade_level = 0.39 * (num_words / num_sentences) + 11.8 * (total_syllables / num_words) - 15.59
    return max(0.0, float(round(grade_level, 2)))

def calculate_entity_density(text: str) -> float:
    """Estimates entity density as: (proper noun capitalized sequences count) / (total word count)."""
    words = text.split()
    if not words:
        return 0.0
        
    proper_nouns = 0
    for i, word in enumerate(words):
        if not word:
            continue
        # Proper noun heuristic: starts with capitalized letter and is alphabetic
        if word[0].isupper() and word[0].isalpha():
            # Exclude if first word of paragraph/sentence
            if i == 0:
                continue
            prev_word = words[i - 1]
            if prev_word and prev_word[-1] in (".", "?", "!"):
                continue
            proper_nouns += 1
            
    return float(round(proper_nouns / len(words), 4))

def compute_text_quality_score(text: str) -> float:
    """Calculates alphabetic content ratio as a proxy for text OCR quality."""
    total_chars = len(text)
    if total_chars == 0:
        return 0.0
    alpha_chars = sum(1 for c in text if c.isalpha())
    return float(round(alpha_chars / total_chars, 4))

def compute_all_document_features(doc: IngestedDocument) -> Dict[str, Any]:
    """Computes all 9 target document metrics and returns a consolidated feature dictionary."""
    text = doc.raw_text
    page_cnt = compute_page_count(text, doc.page_count)
    avg_words = compute_avg_words_per_page(text, page_cnt)
    has_t = check_has_tables(text)
    has_img = check_has_images(doc.metadata, doc.source_type)
    doc_type = detect_doc_type_signal(text)
    read_lvl = calculate_reading_level(text)
    ent_dens = calculate_entity_density(text)
    quality_sc = compute_text_quality_score(text)
    
    return {
        "page_count": page_cnt,
        "avg_words_per_page": avg_words,
        "has_tables": has_t,
        "has_images": has_img,
        "language": doc.language,
        "doc_type_signal": doc_type,
        "reading_level": read_lvl,
        "entity_density": ent_dens,
        "text_quality_score": quality_sc
    }
