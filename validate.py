"""
Validation: Medication Database Cross-Reference
================================================
Used by BOTH layers:

  Layer 1 (OCR): calls extract_medication_from_raw_text() + validate_extraction()
    → If high confidence + validates: field confirmed, LLM skipped entirely

  Layer 2 (LLM): calls validate_extraction() on structured output
    → Double-checks the LLM's extracted medication name

In production, replace the local MEDICATION_DB with:
  - FDA NDC (National Drug Code) API
  - RxNorm API (NLM)
  - Your internal formulary database

Why validation is cheap:
  A dictionary lookup is microseconds and costs nothing.
  Catching "0zempic" here means no wasted LLM call to fix it.
"""

import re

# ── Medication reference database ─────────────────────────────────────────────
MEDICATION_DB = {
    "ozempic": {
        "generic": "semaglutide",
        "drug_class": "GLP-1 receptor agonist",
        "common_strengths": ["0.5mg/dose", "1mg/dose", "2mg/dose"],
        "route": "subcutaneous injection",
    },
    "wegovy": {
        "generic": "semaglutide",
        "drug_class": "GLP-1 receptor agonist",
        "common_strengths": ["0.25mg", "0.5mg", "1mg", "1.7mg", "2.4mg"],
        "route": "subcutaneous injection",
    },
    "mounjaro": {
        "generic": "tirzepatide",
        "drug_class": "GIP/GLP-1 receptor agonist",
        "common_strengths": ["2.5mg", "5mg", "7.5mg", "10mg", "12.5mg", "15mg"],
        "route": "subcutaneous injection",
    },
    "trulicity": {
        "generic": "dulaglutide",
        "drug_class": "GLP-1 receptor agonist",
        "common_strengths": ["0.75mg", "1.5mg", "3mg", "4.5mg"],
        "route": "subcutaneous injection",
    },
    "victoza": {
        "generic": "liraglutide",
        "drug_class": "GLP-1 receptor agonist",
        "common_strengths": ["1.2mg", "1.8mg"],
        "route": "subcutaneous injection",
    },
    "rybelsus": {
        "generic": "semaglutide",
        "drug_class": "GLP-1 receptor agonist",
        "common_strengths": ["3mg", "7mg", "14mg"],
        "route": "oral tablet",
    },
    "metformin": {
        "generic": "metformin hydrochloride",
        "drug_class": "Biguanide",
        "common_strengths": ["500mg", "850mg", "1000mg"],
        "route": "oral tablet",
    },
    "humira": {
        "generic": "adalimumab",
        "drug_class": "TNF inhibitor",
        "common_strengths": ["40mg/0.8mL"],
        "route": "subcutaneous injection",
    },
    "keytruda": {
        "generic": "pembrolizumab",
        "drug_class": "PD-1 inhibitor",
        "common_strengths": ["100mg/4mL"],
        "route": "intravenous infusion",
    },
    "enbrel": {
        "generic": "etanercept",
        "drug_class": "TNF inhibitor",
        "common_strengths": ["25mg", "50mg"],
        "route": "subcutaneous injection",
    },
}

# Known OCR misreads → correct spelling
OCR_CORRECTIONS = {
    "0zempic":       "ozempic",
    "ozempik":       "ozempic",
    "ozemprc":       "ozempic",
    "semag1utide":   "semaglutide",
    "semaglut1de":   "semaglutide",
    "metforrmin":    "metformin",
    "metfomrin":     "metformin",
    "m0unjaro":      "mounjaro",
    "trulicty":      "trulicity",
}

# All known medication names for fast regex matching
ALL_MED_NAMES = list(MEDICATION_DB.keys()) + list(OCR_CORRECTIONS.keys())


def extract_medication_from_raw_text(raw_text: str) -> str | None:
    """
    Called by Layer 1 (OCR) to pull medication name directly from raw OCR text.

    Uses regex pattern matching against known drug names — no LLM needed.
    This is the key function that enables Layer 1 validation:
      if we can identify the medication here with high OCR confidence,
      we skip the LLM entirely for this field.

    Returns the matched medication name (title case), or None if not found.
    """
    text_lower = raw_text.lower()

    # Build pattern from all known names (longest first to avoid partial matches)
    sorted_names = sorted(ALL_MED_NAMES, key=len, reverse=True)
    pattern = r'\b(' + '|'.join(re.escape(n) for n in sorted_names) + r')\b'

    match = re.search(pattern, text_lower)
    if match:
        found = match.group(1)
        # Apply OCR correction if needed
        corrected = OCR_CORRECTIONS.get(found, found)
        return corrected.title()

    return None


def validate_extraction(extracted_data: dict) -> dict:
    """
    Validate extracted fields against reference databases.

    Called by:
      - Layer 1: to decide whether to skip LLM for a field
      - Layer 2: to double-check LLM's structured output

    Returns:
        {
            "medication_valid": bool,
            "medication_info": dict | None,
            "ocr_correction_applied": bool,
            "messages": [str],
            "confidence_boost": float,
        }
    """
    messages = []
    medication_valid = False
    medication_info  = None
    ocr_correction_applied = False

    med = extracted_data.get("medication", {})
    brand_name   = (med.get("brand_name")   or "").strip()
    generic_name = (med.get("generic_name") or "").strip()

    if not brand_name and not generic_name:
        messages.append("⚠  No medication name found to validate")
        return {
            "medication_valid": False,
            "medication_info": None,
            "ocr_correction_applied": False,
            "messages": messages,
            "confidence_boost": 0.0,
        }

    lookup_key = brand_name.lower()

    # Step 1: direct lookup
    if lookup_key in MEDICATION_DB:
        medication_valid = True
        medication_info  = MEDICATION_DB[lookup_key]
        messages.append(f"✓  '{brand_name}' found in medication database")
        messages.append(f"   Generic : {medication_info['generic']}")
        messages.append(f"   Class   : {medication_info['drug_class']}")

    # Step 2: OCR correction lookup
    elif lookup_key in OCR_CORRECTIONS:
        corrected = OCR_CORRECTIONS[lookup_key]
        if corrected in MEDICATION_DB:
            medication_valid = True
            medication_info  = MEDICATION_DB[corrected]
            ocr_correction_applied = True
            messages.append(f"✓  OCR correction: '{brand_name}' → '{corrected.title()}'")
            messages.append(f"   Generic : {medication_info['generic']}")

    # Step 3: try matching on generic name
    else:
        generic_key = generic_name.lower()
        for brand, info in MEDICATION_DB.items():
            if info["generic"].lower() == generic_key:
                medication_valid = True
                medication_info  = info
                messages.append(f"✓  Generic '{generic_name}' matched to '{brand.title()}'")
                break

        if not medication_valid:
            messages.append(f"⚠  '{brand_name}' not found in medication database")
            messages.append("   → Flag for pharmacist review before sending to pre-auth tool")

    # Strength validation
    if medication_valid and medication_info:
        strength = med.get("strength", "")
        if strength:
            strength_num = "".join(c for c in strength if c.isdigit() or c == ".")
            known = " ".join(medication_info["common_strengths"])
            if strength_num and strength_num in known:
                messages.append(f"✓  Strength '{strength}' is within known range")
            else:
                messages.append(f"⚠  Strength '{strength}' not in typical range — verify")

    return {
        "medication_valid": medication_valid,
        "medication_info":  medication_info,
        "ocr_correction_applied": ocr_correction_applied,
        "messages": messages,
        "confidence_boost": 5.0 if medication_valid else 0.0,
    }
