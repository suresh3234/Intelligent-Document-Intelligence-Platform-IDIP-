import re
import unicodedata
from collections import Counter
from typing import List, Set

class TextCleaner:
    """Cleans raw text data using unicode normalization, control character removal, and boilerplate checks."""
    
    def clean(self, text: str) -> str:
        """Runs the sequential preprocessing/cleaning chain on text."""
        if not text:
            return ""
            
        # 1. Unicode normalisation (NFKC)
        normalized = unicodedata.normalize("NFKC", text)
        
        # 2. Remove control characters (keeping newlines \n and tabs \t)
        # Control characters are in categories Cc (Other, control) and Cf (Other, format)
        cleaned_chars = []
        for char in normalized:
            category = unicodedata.category(char)
            if category in ("Cc", "Cf"):
                if char in ("\n", "\t"):
                    cleaned_chars.append(char)
            else:
                cleaned_chars.append(char)
        no_controls = "".join(cleaned_chars)
        
        # 3. Detect and remove boilerplate (headers/footers appearing on >80% of pages)
        boilerplate_removed = self.remove_boilerplate(no_controls)
        
        # 4. Collapse repeated whitespace
        # Collapse spaces and tabs to a single space
        collapsed = re.sub(r'[^\S\r\n\t]+', ' ', boilerplate_removed)
        # Collapse multiple newlines to a double newline (preserving paragraphs)
        collapsed = re.sub(r'\n{3,}', '\n\n', collapsed)
        # Strip outer spaces
        return collapsed.strip()

    def remove_boilerplate(self, text: str) -> str:
        """Identifies and filters header/footer lines appearing on more than 80% of pages."""
        # Split text into pages by PAGE delimiter
        page_pattern = r'--- PAGE \d+ ---\n?'
        pages = re.split(page_pattern, text)
        
        # Keep track of page boundaries to reconstruct if needed
        page_delimiters = re.findall(page_pattern, text)
        
        valid_pages = [p for p in pages if p.strip()]
        num_pages = len(valid_pages)
        
        # Boilerplate check requires at least 3 pages to establish a pattern
        if num_pages < 3:
            return text
            
        line_counts: Counter[str] = Counter()
        page_lines: List[List[str]] = []
        
        for page in valid_pages:
            lines = [line.strip() for line in page.split("\n") if line.strip()]
            page_lines.append(lines)
            # Count distinct lines per page to avoid counting duplicate items inside a page
            for distinct_line in set(lines):
                line_counts[distinct_line] += 1
                
        # Boilerplate lines appear on >80% of pages and are not trivial lengths
        boilerplate: Set[str] = set()
        for line, count in line_counts.items():
            if len(line) > 3 and (count / num_pages) > 0.80:
                boilerplate.add(line)
                
        if not boilerplate:
            return text
            
        # Reconstruct pages without boilerplate lines
        reconstructed_pages = []
        for lines in page_lines:
            cleaned_lines = [l for l in lines if l not in boilerplate]
            reconstructed_pages.append("\n".join(cleaned_lines))
            
        # Re-assemble with delimiters if pages were cleanly split
        if len(page_delimiters) == len(reconstructed_pages):
            parts = []
            for delim, content in zip(page_delimiters, reconstructed_pages, strict=False):
                parts.append(f"{delim}{content}")
            return "\n\n".join(parts)
            
        return "\n\n".join(reconstructed_pages)

    def is_table_chunk(self, chunk_text: str) -> bool:
        """Determines heuristically if a chunk contains tabular structured columns."""
        lines = chunk_text.split("\n")
        for line in lines:
            # Check for 3 tab-separated columns (requires at least 2 tabs)
            if line.count("\t") >= 2:
                return True
            # Check for standard markdown table row (requires at least 3 '|' separators)
            if line.count("|") >= 3:
                return True
        return False
