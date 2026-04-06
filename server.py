"""
Judi Health IDP — Local Web Server
====================================
Loads ANTHROPIC_API_KEY from .env.
Runs real Docling + EasyOCR on uploaded image, then passes OCR text to Claude Haiku.

Usage:
    pip install -r requirements.txt
    python server.py
    → Open http://localhost:5000
"""

import json
import os
import base64
import io
import tempfile
import requests
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    print("WARNING: ANTHROPIC_API_KEY not found in .env")

app = Flask(__name__, static_folder=".")


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/process", methods=["POST"])
def process():
    if not API_KEY:
        return jsonify({"error": "ANTHROPIC_API_KEY not set in .env"}), 500

    data = request.json
    image_b64 = data.get("image_b64")
    media_type = data.get("media_type", "image/jpeg")
    threshold = data.get("threshold", 75)

    if not image_b64:
        return jsonify({"error": "No image provided"}), 400

    # ── Decode and save image to temp file for Docling ─────────────
    image_bytes = base64.b64decode(image_b64)

    # Compress if still too large
    MAX_BYTES = 4_800_000
    if len(image_bytes) > MAX_BYTES:
        try:
            from PIL import Image as PILImage
            img = PILImage.open(io.BytesIO(image_bytes))
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            ratio = (MAX_BYTES / len(image_bytes)) ** 0.5
            img = img.resize((int(img.width * ratio), int(img.height * ratio)), PILImage.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            image_bytes = buf.getvalue()
            image_b64 = base64.b64encode(image_bytes).decode()
            media_type = "image/jpeg"
        except Exception as e:
            return jsonify({"error": f"Image compression failed: {e}"}), 500

    # Write to temp file so Docling can read it
    suffix = ".jpg" if "jpeg" in media_type or "jpg" in media_type else ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    # ── Layer 1: Docling + EasyOCR ─────────────────────────────────
    try:
        from ocr import run_ocr
        ocr_result = run_ocr(tmp_path)
    except Exception as e:
        return jsonify({"error": f"OCR failed: {e}"}), 500
    finally:
        os.unlink(tmp_path)

    ocr_text = ocr_result["text"]
    ocr_confidence = ocr_result["confidence"]
    ocr_engine = ocr_result["engine"]
    routing = ocr_result.get("routing", {})
    above_threshold = ocr_confidence >= threshold

    # ── Layer 2: Claude Haiku — structured extraction ──────────────
    system_prompt = """You are a medical document extraction AI for Judi Health's Intelligent Document Processing pipeline.

You will receive OCR text extracted from a prior authorization form. Extract all fields with confidence scores.

Return ONLY valid JSON — no markdown, no explanation:

{
  "patient": {
    "first_name":    {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "last_name":     {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "date_of_birth": {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "gender":        {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "phone":         {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "address":       {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "city":          {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "state":         {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "zip_code":      {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "member_id":     {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false}
  },
  "prescriber": {
    "first_name":  {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "last_name":   {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "specialty":   {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "npi_number":  {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "phone":       {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "fax":         {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "address":     {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "city":        {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "state":       {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "zip_code":    {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false}
  },
  "pharmacy": {
    "name": {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "fax":  {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false}
  },
  "medication": {
    "brand_name":          {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "generic_name":        {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "strength":            {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "directions":          {"value": null, "confidence": 0.0, "layer": "llm", "needs_review": false},
    "quantity":            {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "day_supply":          {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "duration_of_therapy": {"value": null, "confidence": 0.0, "layer": "llm", "needs_review": false},
    "therapy_type":        {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false}
  },
  "diagnosis": {
    "icd10_codes":            {"value": [], "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "patient_height":         {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "patient_weight":         {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "previous_medications":   {"value": null, "confidence": 0.0, "layer": "llm", "needs_review": false},
    "documentation_provided": {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false}
  },
  "request": {
    "request_date":                 {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "expedite_review":              {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "prescriber_signature_present": {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false},
    "prescriber_signature_date":    {"value": null, "confidence": 0.0, "layer": "ocr", "needs_review": false}
  }
}

Confidence rules:
- Printed text, clearly readable: 0.92-0.99
- Printed text, partially obscured: 0.70-0.91
- Handwritten text, clearly readable: 0.80-0.95
- Handwritten text, ambiguous: 0.50-0.79
- Inferred from context: 0.40-0.65
- Not present or illegible: 0.0, value: null

Layer rules:
- "ocr": machine-readable printed text
- "llm": required interpretation, handwriting parsing, or inference

needs_review: true if confidence < 0.75, OR value is null, OR field is critical and confidence < 0.98
Critical fields: first_name, last_name, date_of_birth, address, brand_name, directions, icd10_codes, npi_number

Normalize: dates → YYYY-MM-DD, phones → XXX-XXX-XXXX
Checkboxes: look for X or check marks — extract as true/false, not null"""

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "Content-Type": "application/json",
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
        },
        json={
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 2000,
            "system": system_prompt,
            "messages": [{
                "role": "user",
                "content": f"Extract all fields from this prior authorization form OCR text:\n\n<ocr_text>\n{ocr_text}\n</ocr_text>"
            }]
        },
        timeout=30
    )

    if not response.ok:
        return jsonify({"error": f"Anthropic API error: {response.text}"}), 500

    api_data = response.json()
    raw_text = api_data["content"][0]["text"].strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    extracted = json.loads(raw_text)
    usage = api_data["usage"]
    cost = (usage["input_tokens"] / 1e6 * 0.80) + (usage["output_tokens"] / 1e6 * 4.00)

    # Post-process: enforce review flags
    CRITICAL = {"first_name","last_name","date_of_birth","address","brand_name","directions","icd10_codes","npi_number"}
    review_count = 0
    for section in extracted.values():
        for key, field in section.items():
            if not isinstance(field, dict):
                continue
            val = field.get("value")
            conf = field.get("confidence", 0)
            if val is None or val == [] or val == "":
                field["needs_review"] = True
            elif key in CRITICAL and conf < 0.98:
                field["needs_review"] = True
            if field.get("needs_review"):
                review_count += 1

    # Mark medication_brand_name as L1 validated if applicable
    if "medication_brand_name" in routing.get("validated_fields", {}):
        if "medication" in extracted and "brand_name" in extracted["medication"]:
            extracted["medication"]["brand_name"]["layer"] = "l1"

    return jsonify({
        "ocr": {
            "confidence": round(ocr_confidence, 1),
            "engine": ocr_engine,
            "above_threshold": above_threshold,
            "text_preview": ocr_text[:300],
        },
        "routing": {
            "validated_fields": list(routing.get("validated_fields", {}).keys()),
            "escalated_fields": routing.get("escalated_fields", []),
            "reasons": routing.get("reasons", []),
        },
        "extracted_data": extracted,
        "review_count": review_count,
        "usage": {
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "model": "claude-haiku-4-5-20251001",
            "cost_usd": round(cost, 6),
        }
    })


if __name__ == "__main__":
    print("\n  Judi Health IDP — Local Server")
    print("  ================================")
    print(f"  API key: {'loaded ✓' if API_KEY else 'NOT FOUND in .env'}")
    print("  Open: http://localhost:5000\n")
    app.run(debug=True, port=5000)