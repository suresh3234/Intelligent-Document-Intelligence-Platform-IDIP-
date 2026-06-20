"""Async Model Router implementation for IDIP."""
import asyncio
import time
import logging
from typing import List, Dict, Any, Optional
from PIL import Image
import io
from pydantic import BaseModel, Field

from ingestion.models import IngestedDocument
from models.ner.service import NERService, EntityResult
from models.classifier.service import DocumentClassifier, ClassificationResult
from models.vision.service import VisionDocumentAnalyzer, VisionResult
from models.llm.inference import LLMInferenceService

logger = logging.getLogger("idip.models.router")

class ModelExecutionTrace(BaseModel):
    """Pydantic model representing the detailed execution trace of models ran."""
    doc_id: str = Field(..., description="Target document ID")
    source_type: str = Field(..., description="Document source format")
    models_run: List[str] = Field(..., description="List of model labels executed")
    start_time: float = Field(..., description="Unix timestamp of routing start")
    end_time: float = Field(..., description="Unix timestamp of routing completion")
    latency_ms: float = Field(..., description="Total elapsed routing time in ms")
    execution_details: Dict[str, Any] = Field(..., description="Raw output predictions per model")

class ModelRouter:
    """
    Model Router orchestrating inference routing dynamically.
    Routes documents by source format (e.g. image vs pdf), executes models in parallel
    via asyncio task pools, profiles execution timings, and records details in traces.
    """

    def __init__(
        self,
        ner_service: Optional[NERService] = None,
        classifier_service: Optional[DocumentClassifier] = None,
        vision_analyzer: Optional[VisionDocumentAnalyzer] = None,
        llm_service: Optional[LLMInferenceService] = None
    ):
        # Initialize sub-services lazily if not provided
        self.ner_service = ner_service or NERService()
        self.classifier_service = classifier_service or DocumentClassifier()
        self.vision_analyzer = vision_analyzer or VisionDocumentAnalyzer()
        self.llm_service = llm_service or LLMInferenceService()

    async def route_document(
        self,
        doc: IngestedDocument,
        run_llm: bool = False,
        llm_prompt: Optional[str] = None
    ) -> ModelExecutionTrace:
        """
        Routes document to correct inference components based on source type.
        Runs independent model calls concurrently using asyncio.to_thread and asyncio.gather.
        """
        start_time = time.time()
        source_type = doc.source_type
        doc_id = doc.doc_id
        
        models_run: List[str] = []
        execution_details: Dict[str, Any] = {}

        # 1. Define model tasks wrappers using asyncio.to_thread to prevent blocking the async loop
        async def run_ner_task(text: str) -> List[EntityResult]:
            # Run in worker thread
            res = await asyncio.to_thread(self.ner_service.extract_entities, text, doc_id)
            return res

        async def run_classifier_task(document: IngestedDocument) -> ClassificationResult:
            # Run in worker thread
            res = await asyncio.to_thread(self.classifier_service.predict, document)
            return res

        async def run_vision_task(image_bytes: Optional[bytes], text: str) -> VisionResult:
            if not image_bytes:
                # Stub empty PIL Image if bytes missing
                img = Image.new("RGB", (224, 224), color="white")
            else:
                img = Image.open(io.BytesIO(image_bytes))
            
            res = await asyncio.to_thread(self.vision_analyzer.analyze_document, img, text)
            return res

        async def run_llm_task(prompt: str) -> str:
            res = await asyncio.to_thread(self.llm_service.generate, prompt)
            return res

        # 2. Dynamic Routing Logic by source type
        try:
            if source_type == "image":
                logger.info(f"Routing document {doc_id} (image) sequentially: Vision -> NER...")
                # Image documents execute sequentially: Vision first, then NER (so NER can potentially use Vision OCR outputs)
                vision_res = await run_vision_task(doc.raw_bytes, doc.raw_text)
                models_run.append("vision")
                execution_details["vision"] = vision_res.model_dump()

                # NER uses text extracted from the vision analyzer
                ocr_text = vision_res.regions[0].text if vision_res.regions else doc.raw_text
                ner_res = await run_ner_task(ocr_text or "")
                models_run.append("ner")
                execution_details["ner"] = [ent.model_dump() for ent in ner_res]

            elif source_type in ("pdf", "api", "database"):
                logger.info(f"Routing document {doc_id} ({source_type}) in parallel: Classifier + NER...")
                # Run Classifier and NER concurrently
                class_task = run_classifier_task(doc)
                ner_task = run_ner_task(doc.raw_text)
                
                # Execute in parallel
                class_res, ner_res = await asyncio.gather(class_task, ner_task)
                
                models_run.extend(["classifier", "ner"])
                execution_details["classifier"] = class_res.model_dump()
                execution_details["ner"] = [ent.model_dump() for ent in ner_res]

            else:
                logger.info(f"Routing document {doc_id} (fallback/stream) in parallel: Classifier + NER...")
                # Fallback path for other source types (e.g. stream)
                class_task = run_classifier_task(doc)
                ner_task = run_ner_task(doc.raw_text)
                
                class_res, ner_res = await asyncio.gather(class_task, ner_task)
                
                models_run.extend(["classifier", "ner"])
                execution_details["classifier"] = class_res.model_dump()
                execution_details["ner"] = [ent.model_dump() for ent in ner_res]

            # 3. Always run LLM for generation tasks if requested
            if run_llm and llm_prompt:
                logger.info(f"Routing document {doc_id} to LLM generator...")
                llm_res = await run_llm_task(llm_prompt)
                models_run.append("llm")
                execution_details["llm"] = llm_res

        except Exception as e:
            logger.error(f"Failed to route document {doc_id}: {e}")
            raise RuntimeError(f"Model Routing execution failed: {e}") from e

        end_time = time.time()
        latency = (end_time - start_time) * 1000

        return ModelExecutionTrace(
            doc_id=doc_id,
            source_type=source_type,
            models_run=models_run,
            start_time=start_time,
            end_time=end_time,
            latency_ms=float(round(latency, 2)),
            execution_details=execution_details
        )
