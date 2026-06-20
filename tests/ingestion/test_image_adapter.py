import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from ingestion.adapters.image import ImageAdapter
from ingestion.exceptions import AdapterError

@pytest.mark.asyncio
async def test_image_adapter_ingest_success():
    adapter = ImageAdapter()
    
    # Mock image representation (100x100 3-channel image)
    mock_img = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_gray = np.zeros((100, 100), dtype=np.uint8)
    
    # Mock cv2 methods — patch pytesseract at the module alias level to avoid
    # ModuleNotFoundError when pytesseract package is not installed in this env
    with patch("cv2.imdecode", return_value=mock_img) as mock_imdecode, \
         patch("cv2.cvtColor", return_value=mock_gray) as mock_cvtColor, \
         patch("cv2.Canny", return_value=mock_gray) as mock_canny, \
         patch("cv2.HoughLinesP", return_value=None) as mock_hough, \
         patch("cv2.threshold", return_value=(0, mock_gray)) as mock_thresh, \
         patch("cv2.medianBlur", return_value=mock_gray) as mock_blur, \
         patch("ingestion.adapters.image.pytesseract") as mock_ocr_module:

        mock_ocr_module.image_to_string.return_value = "Extracted text from OCR test"

        doc = await adapter.ingest("test.png", raw_bytes=b"fake-image-bytes")

        mock_imdecode.assert_called_once()
        mock_cvtColor.assert_called_once()
        mock_canny.assert_called_once()
        mock_hough.assert_called_once()
        mock_thresh.assert_called_once()
        mock_blur.assert_called_once()
        mock_ocr_module.image_to_string.assert_called_once()

        assert doc.source_type == "image"
        assert doc.mime_type == "image/png"
        assert doc.raw_text == "Extracted text from OCR test"
        assert doc.metadata["width"] == 100
        assert doc.metadata["height"] == 100
        assert doc.metadata["skew_angle"] == 0.0


@pytest.mark.asyncio
async def test_image_adapter_decode_fail():
    adapter = ImageAdapter()
    with patch("cv2.imdecode", return_value=None):
        with pytest.raises(AdapterError):
            await adapter.ingest("test.png", raw_bytes=b"invalid-bytes")
