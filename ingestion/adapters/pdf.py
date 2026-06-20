import uuid
import datetime
import os
from typing import Optional, Dict, Any, List
import fitz  # PyMuPDF

from ingestion.base import BaseSourceAdapter
from ingestion.models import IngestedDocument
from ingestion.exceptions import AdapterError

class PDFAdapter(BaseSourceAdapter):
    """Adapter for ingesting and processing PDF documents using PyMuPDF."""
    
    async def ingest(self, source_uri: str, raw_bytes: Optional[bytes] = None, **kwargs) -> IngestedDocument:
        """
        Ingests a PDF file. Extracts text blocks, detects headings by size, 
        and extracts tables.
        """
        try:
            if raw_bytes is None:
                if not os.path.exists(source_uri):
                    raise AdapterError(f"Local PDF file not found at: {source_uri}")
                with open(source_uri, "rb") as f:
                    raw_bytes = f.read()

            byte_size = len(raw_bytes)
            checksum = self.compute_checksum(raw_bytes)
            
            # Open PDF with PyMuPDF
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
            page_count = doc.page_count
            
            extracted_text_parts: List[str] = []
            headings: List[Dict[str, Any]] = []
            tables_metadata: List[Dict[str, Any]] = []
            
            # First pass: find typical font size to help detect headers
            sizes_frequency: Dict[float, int] = {}
            for page_num in range(page_count):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if block.get("type") == 0:  # Text block
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                size = round(span.get("size", 10.0), 1)
                                sizes_frequency[size] = sizes_frequency.get(size, 0) + len(span.get("text", ""))

            # Identify the most common font size as base font size
            base_font_size = max(sizes_frequency, key=sizes_frequency.get) if sizes_frequency else 10.0
            heading_threshold = base_font_size * 1.25

            # Second pass: extract structured content
            for page_num in range(page_count):
                page = doc[page_num]
                page_text_blocks: List[str] = []
                
                # Heuristic Table Extraction using PyMuPDF find_tables
                try:
                    tables = page.find_tables()
                    for idx, table in enumerate(tables):
                        table_data = table.extract()
                        if table_data:
                            tables_metadata.append({
                                "page": page_num + 1,
                                "table_index": idx,
                                "bbox": list(table.bbox),
                                "headers": table_data[0] if len(table_data) > 0 else [],
                                "row_count": len(table_data),
                                "col_count": len(table_data[0]) if len(table_data) > 0 else 0
                            })
                            # Append table as markdown representation
                            md_rows = []
                            for row in table_data:
                                md_rows.append("| " + " | ".join([str(cell or "").replace("\n", " ") for cell in row]) + " |")
                            if len(md_rows) > 1:
                                md_rows.insert(1, "| " + " | ".join(["---"] * len(table_data[0])) + " |")
                            page_text_blocks.append("\n" + "\n".join(md_rows) + "\n")
                except Exception as table_err:
                    # Fallback if find_tables fails or isn't available
                    pass

                # Text Extraction with block bboxes
                blocks = page.get_text("dict")["blocks"]
                # Sort blocks top-to-bottom, left-to-right
                blocks.sort(key=lambda b: (round(b["bbox"][1], 1), round(b["bbox"][0], 1)))
                
                for block in blocks:
                    if block.get("type") == 0:  # Text
                        block_text_parts = []
                        for line in block.get("lines", []):
                            line_text_parts = []
                            for span in line.get("spans", []):
                                text = span.get("text", "").strip()
                                if not text:
                                    continue
                                size = span.get("size", 10.0)
                                flags = span.get("flags", 0)
                                is_bold = bool(flags & 2)
                                
                                # Heading detection threshold check
                                if size >= heading_threshold or (size > base_font_size and is_bold):
                                    headings.append({
                                        "text": text,
                                        "page": page_num + 1,
                                        "font_size": size,
                                        "bbox": list(span.get("bbox", []))
                                    })
                                    line_text_parts.append(f"\n## {text}\n")
                                else:
                                    line_text_parts.append(text)
                            
                            line_text = " ".join(line_text_parts)
                            if line_text:
                                block_text_parts.append(line_text)
                                
                        block_text = "\n".join(block_text_parts)
                        if block_text:
                            page_text_blocks.append(block_text)

                extracted_text_parts.append(f"--- PAGE {page_num + 1} ---\n" + "\n\n".join(page_text_blocks))

            raw_text = "\n\n".join(extracted_text_parts)
            doc.close()

            metadata = {
                "headings": headings,
                "tables": tables_metadata,
                "pdf_metadata": doc.metadata,
                "base_font_size": base_font_size
            }
            
            return IngestedDocument(
                doc_id=str(uuid.uuid4()),
                ingestion_ts=datetime.datetime.utcnow(),
                source_type="pdf",
                source_uri=source_uri,
                raw_text=raw_text,
                raw_bytes=raw_bytes,
                byte_size=byte_size,
                checksum=checksum,
                language=kwargs.get("language", "en"),
                mime_type="application/pdf",
                page_count=page_count,
                metadata=metadata
            )
            
        except Exception as e:
            raise AdapterError(f"PDF extraction failed: {str(e)}") from e
