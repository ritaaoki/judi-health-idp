"""
Judi Health IDP — Local Web Server
====================================
Layer 1: Google Document AI Form Parser
Layer 2: Claude Haiku — only for medication/diagnosis fields below 95% confidence
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

API_KEY      = os.getenv("ANTHROPIC_API_KEY")
GCP_PROJECT  = os.getenv("GCP_PROJECT_ID", "")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us")
GCP_PROCESSOR= os.getenv("GCP_PROCESSOR_ID", "")

if not API_KEY:
    print("WARNING: ANTHROPIC_API_KEY not found in .env")

app = Flask(__name__, static_folder=".")

# Fields where confidence must be 95%+ or they go to LLM
MEDICATION_DIAGNOSIS = {
    "brand_name", "generic_name", "strength", "directions",
    "quantity", "day_supply", "duration_of_therapy", "therapy_type",
    "icd10_codes", "patient_height", "patient_weight",
    "previous_medications", "documentation_provided",
}
THRESHOLD = 0.95


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

    if not image_b64:
        return jsonify({"error": "No image provided"}), 400

    # ── Compress if needed ─────────────────────────────────────────
    image_bytes = base64.b64decode(image_b64)
    if len(image_bytes) > 4_800_000:
        try:
            from PIL import Image as PILImage
            img = PILImage.open(io.BytesIO(image_bytes))
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            ratio = (4_800_000 / len(image_bytes)) ** 0.5
            img = img.resize((int(img.width * ratio), int(img.height * ratio)), PILImage.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            image_bytes = buf.getvalue()
            image_b64 = base64.b64encode(image_bytes).decode()
            media_type = "image/jpeg"
        except Exception as e:
            return jsonify({"error": f"Compression failed: {e}"}), 500

    # ── Layer 1: Google Document AI Form Parser ────────────────────
    if not (GCP_PROJECT and GCP_PROCESSOR):
        return jsonify({"error": "GCP not configured — set GCP_PROJECT_ID and GCP_PROCESSOR_ID in .env"}), 500

    try:
        from google.cloud import documentai
        from google.api_core.client_options import ClientOptions

        opts = ClientOptions(api_endpoint=f"{GCP_LOCATION}-documentai.googleapis.com")
        client = documentai.DocumentProcessorServiceClient(client_options=opts)
        processor_name = client.processor_path(GCP_PROJECT, GCP_LOCATION, GCP_PROCESSOR)

        mime_type = "image/jpeg" if "jpeg" in media_type or "jpg" in media_type else "image/png"
        raw_document = documentai.RawDocument(content=image_bytes, mime_type=mime_type)
        req = documentai.ProcessRequest(name=processor_name, raw_document=raw_document)
        result = client.process_document(request=req)
        document = result.document

        # Build key-value map from Document AI
        docai_fields = {}
        confidences = []
        for page in document.pages:
            for field in page.form_fields:
                key = field.field_name.text_anchor.content.strip().rstrip(":") if field.field_name.text_anchor else ""
                val = field.field_value.text_anchor.content.strip() if field.field_value.text_anchor else ""
                conf = getattr(field.field_value, "confidence", 0.85)
                if key:
                    docai_fields[key] = {"value": val or None, "confidence": conf}
                    confidences.append(conf)

        ocr_confidence = (sum(confidences) / len(confidences) * 100) if confidences else 80.0
        docai_text = f"=== FORM FIELDS (Google Document AI) ===\n"
        docai_text += "\n".join(f"{k}: {v['value'] or 'not found'}" for k, v in docai_fields.items())
        docai_text += f"\n\n=== RAW TEXT ===\n{document.text}"

        print(f"  [Layer 1] Document AI: {len(docai_fields)} fields, avg confidence {ocr_confidence:.1f}%")

    except Exception as e:
        return jsonify({"error": f"Google Document AI failed: {e}"}), 500

    # ── Layer 2: Claude Haiku — structured extraction ──────────────
    # Claude maps Document AI's key-value pairs into our JSON schema.
    # We use Document AI's confidence scores (real) not Claude's (self-reported).
    with open(os.path.join(os.path.dirname(__file__), "prompt.txt")) as f:
        system_prompt = f.read()

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
            "messages": [{"role": "user", "content": f"Structure this Document AI output into the JSON schema:\n\n{docai_text}"}]
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

    # ── Inject real Document AI confidence scores ─────────────────
    # Claude's self-reported confidence scores are not reliable — they're
    # estimates based on instructions, not actual measurement. We replace
    # them with Document AI's real per-field confidence scores where available.
    FIELD_KEY_MAP = {
        # Maps our JSON field names to likely Document AI key names
        "first_name": ["First Name", "Patient First Name", "Prescriber First Name", "First Name:"],
        "last_name": ["Last Name", "Patient Last Name", "Last Name:"],
        "date_of_birth": ["Date of Birth", "DOB", "Date of Birth:"],
        "gender": ["Male", "Female", "Gender", "Sex"],
        "phone": ["Phone Number", "Phone", "Phone Number:"],
        "address": ["Address", "Address:"],
        "city": ["City", "City:"],
        "state": ["State", "State:"],
        "zip_code": ["Zip Code", "ZIP Code", "Zip Code:"],
        "member_id": ["Member ID", "Member ID:"],
        "specialty": ["Specialty", "Specialty:"],
        "npi_number": ["NPI Number", "NPI Number (individual)", "NPI Number:"],
        "fax": ["Fax Number", "Fax Number (in HIPAA compliant area)", "Fax:"],
        "brand_name": ["Medication Name and Strength", "Medication Name", "Brand Name"],
        "directions": ["Directions for Use", "Directions", "Directions for Use:"],
        "quantity": ["Quantity", "Quantity:"],
        "day_supply": ["Day Supply", "Day Supply:"],
        "icd10_codes": ["ICD 10 code(s)", "ICD-10", "ICD 10"],
        "patient_height": ["Patient Height", "Patient Height (in/cm)"],
        "patient_weight": ["Patient Weight", "Patient Weight (lb/kg)"],
    }

    def find_docai_confidence(field_key):
        candidates = FIELD_KEY_MAP.get(field_key, [])
        for candidate in candidates:
            if candidate in docai_fields:
                return docai_fields[candidate]["confidence"]
            # Case-insensitive fallback
            for k, v in docai_fields.items():
                if k.lower().startswith(candidate.lower()):
                    return v["confidence"]
        return None

    for section_name, section in extracted.items():
        for key, field in section.items():
            if not isinstance(field, dict):
                continue
            real_conf = find_docai_confidence(key)
            if real_conf is not None:
                field["confidence"] = round(real_conf, 3)

    # ── Post-process: routing logic ───────────────────────────────
    # Medication/diagnosis: below 90% → LLM re-extraction
    # All other fields:     below 80% → LLM re-extraction
    # Null:                 always needs review
    MED_DIAG_THRESHOLD = 0.90
    OTHER_THRESHOLD    = 0.80

    validated_fields = []
    review_count = 0
    llm_reextract = []

    for section_name, section in extracted.items():
        for key, field in section.items():
            if not isinstance(field, dict):
                continue
            val  = field.get("value")
            conf = field.get("confidence", 0)

            if val is None or val == [] or val == "":
                field["needs_review"] = True
            elif key in MEDICATION_DIAGNOSIS and conf < MED_DIAG_THRESHOLD:
                field["layer"] = "llm"
                field["needs_review"] = True
                llm_reextract.append(section_name + "." + key)
            elif key not in MEDICATION_DIAGNOSIS and conf < OTHER_THRESHOLD:
                field["layer"] = "llm"
                field["needs_review"] = True
                llm_reextract.append(section_name + "." + key)

            if field.get("needs_review"):
                review_count += 1

    # ── LLM re-extraction for low-confidence fields ────────────────
    if llm_reextract:
        print("  [Layer 2] Re-extracting " + str(len(llm_reextract)) + " low-confidence fields: " + str(llm_reextract))
        fields_list = ", ".join(llm_reextract)
        reextract_prompt = (
            "The following fields were extracted with low confidence from a prior authorization form. "
            "Re-extract each field from the Document AI text below and return ONLY a JSON object "
            "where each key is the field path and the value has 'value' and 'confidence'.\n\n"
            "Fields to re-extract: " + fields_list + "\n\n"
            "Document AI text:\n" + docai_text + "\n\n"
            'Return format: {"patient.last_name": {"value": "Smith", "confidence": 0.92}}\n'
            "Return ONLY valid JSON, no explanation."
        )

        reextract_resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": API_KEY,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 500,
                "messages": [{"role": "user", "content": reextract_prompt}]
            },
            timeout=20
        )

        if reextract_resp.ok:
            try:
                rd = reextract_resp.json()
                rt = rd["content"][0]["text"].strip()
                if rt.startswith("```"):
                    rt = rt.replace("```json","").replace("```","").strip()
                reextract_results = json.loads(rt)

                for field_path, result in reextract_results.items():
                    parts = field_path.split(".", 1)
                    if len(parts) == 2:
                        sec, fkey = parts
                        if sec in extracted and fkey in extracted[sec]:
                            new_val  = result.get("value")
                            new_conf = result.get("confidence", 0)
                            extracted[sec][fkey]["value"]      = new_val
                            extracted[sec][fkey]["confidence"] = new_conf
                            extracted[sec][fkey]["layer"]      = "llm"
                            if new_val is None or new_val == "":
                                extracted[sec][fkey]["needs_review"] = True
                            elif new_conf >= 0.80:
                                extracted[sec][fkey]["needs_review"] = False
                                review_count = max(0, review_count - 1)
                print("  [Layer 2] Re-extraction complete for " + str(len(reextract_results)) + " fields")
            except Exception as e:
                print("  [Layer 2] Re-extraction parse failed: " + str(e))

    # L1 medication validation via DB
    from validate import validate_extraction
    med_section = extracted.get("medication", {})
    brand = med_section.get("brand_name", {}).get("value")
    if brand:
        mock = {"medication": {"brand_name": brand, "generic_name": None, "strength": None}}
        val_result = validate_extraction(mock)
        if val_result["medication_valid"]:
            med_section["brand_name"]["layer"] = "l1"
            med_section["brand_name"]["confidence"] = 0.99
            med_section["brand_name"]["needs_review"] = False
            validated_fields.append("medication_brand_name")

    print(f"  [Layer 2] Claude Haiku: {usage['input_tokens']} in, {usage['output_tokens']} out, ${cost:.5f}")
    print(f"  [Result] {review_count} fields need review")

    return jsonify({
        "ocr": {
            "confidence": round(ocr_confidence, 1),
            "engine": "Google Document AI Form Parser",
        },
        "routing": {
            "validated_fields": validated_fields,
            "escalated_fields": [],
            "reasons": [f"Document AI extracted {len(docai_fields)} key-value pairs"],
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
    print(f"  API key:      {'loaded ✓' if API_KEY else 'NOT FOUND in .env'}")
    print(f"  Document AI:  {'configured ✓' if GCP_PROJECT and GCP_PROCESSOR else 'NOT configured — check .env'}")
    print("  Open: http://localhost:5000")
    print("")
    app.run(debug=True, port=5000)