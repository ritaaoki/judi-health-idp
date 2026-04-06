"""
Layer 2: LLM Extraction
========================
Only runs when Layer 1 could NOT validate fields with high confidence.

Cost optimization decisions:
  - Model    : claude-haiku-4-5 (cheapest, ideal for structured extraction)
  - One call : extracts ALL fields in a single prompt (not one call per field)
  - Context  : we tell the LLM what Layer 1 already confirmed so it doesn't
               re-extract those fields unnecessarily
  - JSON out : forces structured output, no wasted prose tokens

Why Haiku?
  Extraction is pattern-matching, not reasoning. Haiku handles it at
  ~20x lower cost than Opus with equivalent accuracy for this task type.
  Switch to Sonnet only if Haiku accuracy drops below acceptable threshold.

Cost per form (approx):
  ~500 input tokens  × $0.80/M = $0.0004
  ~350 output tokens × $4.00/M = $0.0014
  Total: ~$0.0018 per form (less than 0.2 cents)
"""

import json
import os
import anthropic

MODEL            = "claude-haiku-4-5-20251001"
COST_PER_M_INPUT  = 0.80
COST_PER_M_OUTPUT = 4.00

SYSTEM_PROMPT = """You are a medical document extraction specialist for a prior authorization processing system.

Extract structured data from OCR text of medication prior authorization forms.

RULES:
1. Extract ONLY what is explicitly present. Never invent or assume values.
2. If a field is missing or illegible, use null.
3. Return ONLY valid JSON. No preamble, no markdown fences.
4. Normalize dates to YYYY-MM-DD. Normalize phone numbers to XXX-XXX-XXXX.
5. If a field is marked as ALREADY CONFIRMED in the context, include it as-is.

Return this exact structure:
{
  "patient": {
    "first_name": null, "last_name": null, "date_of_birth": null,
    "gender": null, "phone": null, "address": null,
    "city": null, "state": null, "zip_code": null, "member_id": null
  },
  "prescriber": {
    "first_name": null, "last_name": null, "specialty": null,
    "address": null, "city": null, "state": null, "zip_code": null,
    "npi_number": null, "phone": null, "fax": null
  },
  "pharmacy": { "name": null, "fax": null },
  "medication": {
    "brand_name": null, "generic_name": null, "strength": null,
    "directions": null, "quantity": null, "day_supply": null,
    "duration_of_therapy": null, "dispense_as_written": null, "therapy_type": null
  },
  "diagnosis": {
    "icd10_codes": [], "patient_height": null, "patient_weight": null,
    "previous_medications": null, "documentation_provided": null
  },
  "request": {
    "request_date": null, "expedite_review": null,
    "benefit_type": null, "prescriber_signature_present": null,
    "prescriber_signature_date": null
  }
}"""


def run_llm_extraction(ocr_text: str, layer1_validated: dict = None) -> dict:
    """
    Send OCR text to Claude Haiku for structured field extraction.

    layer1_validated: fields already confirmed in Layer 1 — passed as
    context so the LLM knows what's been verified and can skip re-extracting.

    Returns:
        {
            "extracted_data": dict,
            "fields_extracted": int,
            "usage": dict,
            "estimated_cost_usd": float,
        }
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Build context about what Layer 1 already confirmed
    layer1_context = ""
    if layer1_validated:
        layer1_context = "\n\nALREADY CONFIRMED BY LAYER 1 (do not re-extract, include as-is):\n"
        for field, data in layer1_validated.items():
            layer1_context += f"  - {field}: {data['value']} (validated against drug database)\n"

    user_message = f"""Extract all fields from this prior authorization form OCR text.
{layer1_context}
<ocr_text>
{ocr_text}
</ocr_text>

Return only the JSON object."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences defensively
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        extracted_data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  ⚠  LLM returned invalid JSON: {e}")
        extracted_data = {"error": "JSON parse failed", "raw": raw}

    fields_extracted = _count_non_null(extracted_data)
    input_tokens  = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = (input_tokens / 1_000_000 * COST_PER_M_INPUT) + \
           (output_tokens / 1_000_000 * COST_PER_M_OUTPUT)

    return {
        "extracted_data": extracted_data,
        "fields_extracted": fields_extracted,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": MODEL,
        },
        "estimated_cost_usd": cost,
    }


def _count_non_null(d, count=0):
    if isinstance(d, dict):
        for v in d.values():
            count = _count_non_null(v, count)
    elif isinstance(d, list):
        for item in d:
            count = _count_non_null(item, count)
    elif d is not None:
        count += 1
    return count
