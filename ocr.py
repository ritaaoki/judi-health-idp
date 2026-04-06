"""
Layer 1: OCR — Evolution of our approach
==========================================

What we tried and why we moved on:

1. EasyOCR (via Docling)
   - First attempt: open-source, runs locally, no API cost
   - Problem: trained on printed text, not handwriting
   - Result: missed most fields on a phone photo of a handwritten form
   - Lesson: traditional OCR engines fail on mixed print/handwriting at low DPI

2. Google Document AI Form Parser (current)
   - Purpose-built for structured forms
   - Understands form layout natively — returns key-value pairs directly
   - "First Name:" → label, "Rita" → value, no LLM needed for mapping
   - Handles handwriting significantly better than EasyOCR
   - Fine-tuneable with as few as 5-10 labeled forms
   - Cost: ~$30 per 1,000 pages

3. Future: fine-tuned Document AI on pre-auth forms specifically
   - Upload labeled pre-auth forms
   - Train custom processor via AutoML (no ML expertise required)
   - Target: 99% accuracy on this specific form type

Fallback chain:
  Google Document AI → EasyOCR (via Docling) → pytesseract
"""

import os
import base64
import json
from pathlib import Path
from validate import validate_extraction

OCR_CONFIDENCE_THRESHOLD = 75.0

# Your Google Cloud project details
# Set these in .env or environment variables
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us")  # or "eu"
GCP_PROCESSOR_ID = os.getenv("GCP_PROCESSOR_ID", "")  # Form Parser processor ID


def run_ocr(image_path: str) -> dict:
    """
    Run OCR with fallback chain:
      1. Google Document AI Form Parser
      2. Docling + EasyOCR
      3. pytesseract
    """
    # Try Google Document AI first
    if GCP_PROJECT_ID and GCP_PROCESSOR_ID:
        try:
            return _run_google_document_ai(image_path)
        except ImportError:
            print("  [OCR] google-cloud-documentai not installed")
            print("  [OCR] Run: pip install google-cloud-documentai")
        except Exception as e:
            print(f"  [OCR] Google Document AI failed: {e}")
            print("  [OCR] Falling back to EasyOCR...")
    else:
        print("  [OCR] GCP_PROJECT_ID or GCP_PROCESSOR_ID not set in .env")
        print("  [OCR] Falling back to EasyOCR...")

    # Try Docling + EasyOCR
    try:
        return _run_docling_easyocr(image_path)
    except ImportError:
        print("  [OCR] Docling/EasyOCR not installed, falling back to pytesseract...")
    except Exception as e:
        print(f"  [OCR] EasyOCR failed: {e}, falling back to pytesseract...")

    # Final fallback
    return _run_pytesseract(image_path)


def _run_google_document_ai(image_path: str) -> dict:
    """
    Google Document AI Form Parser.

    Returns key-value pairs directly from the form — no LLM needed
    to map raw text to fields. The processor understands form structure.

    Setup:
      1. Enable Document AI API in GCP Console
      2. Create a Form Parser processor
      3. Set GCP_PROJECT_ID, GCP_LOCATION, GCP_PROCESSOR_ID in .env
      4. Authenticate: gcloud auth application-default login
         OR set GOOGLE_APPLICATION_CREDENTIALS to your service account key

    Fine-tuning for better accuracy:
      - Upload labeled pre-auth forms in Document AI console
      - Train custom processor with AutoML (no ML expertise needed)
      - As few as 5-10 documents to start seeing improvement
    """
    from google.cloud import documentai
    from google.api_core.client_options import ClientOptions

    opts = ClientOptions(api_endpoint=f"{GCP_LOCATION}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    processor_name = client.processor_path(GCP_PROJECT_ID, GCP_LOCATION, GCP_PROCESSOR_ID)

    # Read image file
    with open(image_path, "rb") as f:
        image_content = f.read()

    # Detect MIME type
    suffix = Path(image_path).suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".pdf": "application/pdf"}
    mime_type = mime_map.get(suffix, "image/jpeg")

    raw_document = documentai.RawDocument(content=image_content, mime_type=mime_type)
    request = documentai.ProcessRequest(name=processor_name, raw_document=raw_document)

    result = client.process_document(request=request)
    document = result.document

    # Extract key-value pairs from Form Parser output
    # Document AI returns these directly — no parsing needed
    key_value_pairs = {}
    confidences = []

    for page in document.pages:
        for field in page.form_fields:
            key_text = field.field_name.text_anchor.content if field.field_name.text_anchor else ""
            val_text = field.field_value.text_anchor.content if field.field_value.text_anchor else ""
            conf = field.field_value.confidence if hasattr(field.field_value, 'confidence') else 0.85

            key_clean = key_text.strip().rstrip(":")
            val_clean = val_text.strip()

            if key_clean:
                key_value_pairs[key_clean] = {
                    "value": val_clean if val_clean else None,
                    "confidence": conf,
                }
                confidences.append(conf)

    # Also get raw text for LLM fallback on complex fields
    raw_text = document.text

    avg_confidence = (sum(confidences) / len(confidences) * 100) if confidences else 80.0

    # Format key-value pairs as structured text for LLM
    kv_text = "\n".join(
        f"{k}: {v['value'] or 'not found'}"
        for k, v in key_value_pairs.items()
    )
    combined_text = f"=== FORM FIELDS (Google Document AI) ===\n{kv_text}\n\n=== RAW TEXT ===\n{raw_text}"

    routing = _route_fields(combined_text, avg_confidence)
    routing["key_value_pairs"] = key_value_pairs  # Pass structured data downstream

    return {
        "text": combined_text,
        "confidence": avg_confidence,
        "engine": "Google Document AI Form Parser",
        "key_value_pairs": key_value_pairs,
        "routing": routing,
    }


def _run_docling_easyocr(image_path: str) -> dict:
    """
    Docling + EasyOCR — our first attempt.

    What we learned: EasyOCR is trained on printed text and struggles
    significantly with handwriting, especially at low DPI (phone photos).
    Missed most fields on the pre-auth form. Moved to Document AI.
    """
    from docling.document_converter import DocumentConverter
    from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import ImageFormatOption

    print("  [OCR] WARNING: EasyOCR struggles with handwriting at low DPI")
    print("  [OCR] Consider using Google Document AI for better accuracy")

    easyocr_options = EasyOcrOptions(lang=["en"], use_gpu=False)
    pipeline_options = PdfPipelineOptions(do_ocr=True, ocr_options=easyocr_options)
    converter = DocumentConverter(
        format_options={
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
        }
    )
    result = converter.convert(image_path)
    text = result.document.export_to_markdown()

    routing = _route_fields(text, 65.0)  # Intentionally low — EasyOCR is unreliable here
    return {
        "text": text,
        "confidence": 65.0,
        "engine": "Docling + EasyOCR (limited handwriting support)",
        "routing": routing,
    }


def _run_pytesseract(image_path: str) -> dict:
    """Last resort fallback."""
    import pytesseract
    from PIL import Image

    print("  [OCR] WARNING: pytesseract has very limited handwriting support")
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, config="--psm 6")
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    confidences = [int(c) for c in data["conf"] if str(c).lstrip("-").isdigit() and int(c) >= 0]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    routing = _route_fields(text, avg_confidence)
    return {
        "text": text,
        "confidence": avg_confidence,
        "engine": "pytesseract (fallback — very limited handwriting support)",
        "routing": routing,
    }


def _route_fields(text: str, confidence: float) -> dict:
    """Layer 1 routing: validate medication at Layer 1, escalate rest to LLM."""
    validated_fields = {}
    escalated_fields = []
    reasons = []

    if confidence < OCR_CONFIDENCE_THRESHOLD:
        return {
            "send_to_llm": True,
            "reasons": [f"OCR confidence {confidence:.1f}% below threshold — escalating all to LLM"],
            "validated_fields": {},
            "escalated_fields": ["all_fields"],
        }

    medication = _extract_medication_from_text(text)
    if medication:
        mock = {"medication": {"brand_name": medication, "generic_name": None, "strength": None}}
        validation = validate_extraction(mock)
        if validation["medication_valid"]:
            validated_fields["medication_brand_name"] = {
                "value": medication,
                "validated_by": "layer1_db_lookup",
                "llm_needed": False,
            }
            reasons.append(f"'{medication}' validated at Layer 1 — skipping LLM for this field")
        else:
            escalated_fields.append("medication")
            reasons.append(f"'{medication}' not in DB — escalating to LLM")
    else:
        escalated_fields.append("medication")
        reasons.append("Medication not identified at Layer 1 — escalating to LLM")

    escalated_fields.extend([
        "patient_info", "prescriber_info",
        "pharmacy_info", "diagnosis", "request_metadata"
    ])

    return {
        "send_to_llm": True,
        "reasons": reasons,
        "validated_fields": validated_fields,
        "escalated_fields": escalated_fields,
    }


def _extract_medication_from_text(text: str) -> str | None:
    known = [
        "Ozempic", "Wegovy", "Mounjaro", "Trulicity", "Victoza",
        "Rybelsus", "Metformin", "Humira", "Keytruda", "Enbrel"
    ]
    text_lower = text.lower()
    for med in known:
        if med.lower() in text_lower:
            return med
    return None