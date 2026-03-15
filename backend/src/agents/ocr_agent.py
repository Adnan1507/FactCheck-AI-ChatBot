"""
ocr_agent.py
────────────
Agent 3 — OCR (Optical Character Recognition)
Extracts text from:
  • Images (JPG, PNG, etc.) using EasyOCR
  • PDF documents using PyMuPDF

This text is then passed to the fact-checker for verification.
"""

import io
import fitz          # PyMuPDF — for reading PDFs
import easyocr
from PIL import Image
import numpy as np


# ── Lazy-load the OCR reader (loads once, reused for speed) ──────────────────
# EasyOCR downloads its model on first run — this can take ~1 minute initially
_reader = None

def _get_reader():
    """Returns a shared EasyOCR reader (English language)."""
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)   # gpu=False works on all PCs
    return _reader


# ── Main Function: Extract text from an uploaded image ───────────────────────
def extract_text_from_image(image_bytes: bytes) -> dict:
    """
    Extracts all text visible in an image using EasyOCR.

    Args:
        image_bytes : Raw bytes of the uploaded image.

    Returns:
        A dict with:
          - 'text'  : Extracted text as a single string.
          - 'error' : Error message if something went wrong.
    """
    try:
        reader = _get_reader()

        # Convert bytes → PIL Image → NumPy array (what EasyOCR needs)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        # Run OCR
        results = reader.readtext(image_np, detail=0)   # detail=0 → text only

        extracted_text = "\n".join(results).strip()

        if not extracted_text:
            return {
                "text":  "",
                "error": "⚠️ No text found in the image. Please upload a clearer image.",
            }

        return {"text": extracted_text, "error": None}

    except Exception as e:
        return {
            "text":  "",
            "error": f"❌ Image OCR failed: {str(e)}",
        }


# ── Main Function: Extract text from an uploaded PDF ─────────────────────────
def extract_text_from_pdf(pdf_bytes: bytes) -> dict:
    """
    Extracts all text from a PDF document using PyMuPDF.

    Args:
        pdf_bytes : Raw bytes of the uploaded PDF file.

    Returns:
        A dict with:
          - 'text'  : All extracted text joined together.
          - 'pages' : Number of pages in the PDF.
          - 'error' : Error message if something went wrong.
    """
    try:
        # Open the PDF from bytes
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        all_text = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                all_text.append(f"[Page {page_num + 1}]\n{text.strip()}")

        doc.close()

        full_text = "\n\n".join(all_text).strip()

        if not full_text:
            return {
                "text":  "",
                "pages": len(doc),
                "error": (
                    "⚠️ No text found in the PDF. "
                    "It might be a scanned image PDF — try uploading as an image instead."
                ),
            }

        return {
            "text":  full_text,
            "pages": len(all_text),
            "error": None,
        }

    except Exception as e:
        return {
            "text":  "",
            "pages": 0,
            "error": f"❌ PDF extraction failed: {str(e)}",
        }
