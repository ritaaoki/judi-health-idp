"""
Judi Health - Intelligent Document Processing Pipeline
=======================================================
Layer 1: docling OCR + Validation Routing
Layer 2: Claude LLM (only for fields that didn't pass Layer 1)

Usage:
    python main.py --image path/to/form.jpg
    python main.py --image path/to/form.jpg --verbose
"""

import argparse
import json
import sys
from pathlib import Path

from ocr import run_ocr
from llm import run_llm_extraction
from validate import validate_extraction


def run_pipeline(image_path: str, verbose: bool = False) -> dict:
    print("\n" + "="*60)
    print("  JUDI HEALTH — Intelligent Document Processing")
    print("="*60)

    # ── Layer 1: OCR + Validation Routing ─────────────────────────
    print("\n[Layer 1] OCR + Validation Routing...")
    ocr_result = run_ocr(image_path)
    routing = ocr_result["routing"]

    print(f"  ✓ OCR engine   : {ocr_result['engine']}")
    print(f"  ✓ Confidence   : {ocr_result['confidence']:.1f}%")

    print(f"\n  Routing decisions:")
    for reason in routing["reasons"]:
        print(f"    → {reason}")

    if routing["validated_fields"]:
        print(f"\n  ✓ Validated at Layer 1 (no LLM needed):")
        for field, info in routing["validated_fields"].items():
            print(f"    {field}: {info['value']}")

    if routing["escalated_fields"]:
        print(f"\n  ⬆  Escalating to Layer 2:")
        for field in routing["escalated_fields"]:
            print(f"    {field}")

    if verbose:
        print("\n--- RAW OCR TEXT ---")
        print(ocr_result["text"][:800])
        print("...")

    # ── Layer 2: LLM (only if needed) ─────────────────────────────
    llm_result = None
    if routing["send_to_llm"]:
        print("\n[Layer 2] LLM extraction (escalated fields only)...")
        llm_result = run_llm_extraction(ocr_result["text"])
        print(f"  ✓ Fields extracted : {llm_result['fields_extracted']}")
        print(f"  ✓ Input tokens     : {llm_result['usage']['input_tokens']}")
        print(f"  ✓ Output tokens    : {llm_result['usage']['output_tokens']}")
        print(f"  ✓ Est. cost        : ${llm_result['estimated_cost_usd']:.5f}")

        # Merge Layer 1 validated fields into LLM output
        if routing["validated_fields"]:
            med = llm_result["extracted_data"].setdefault("medication", {})
            for field, info in routing["validated_fields"].items():
                if field == "medication_brand_name":
                    med["brand_name"] = info["value"]
                    med["_validated_at_layer1"] = True
    else:
        print("\n[Layer 2] Skipped — all fields validated at Layer 1 ✓")

    # ── Final output ───────────────────────────────────────────────
    extracted = llm_result["extracted_data"] if llm_result else {}
    cost = llm_result["estimated_cost_usd"] if llm_result else 0.0

    print("\n" + "="*60)
    print("  EXTRACTED DATA")
    print("="*60)
    print(json.dumps(extracted, indent=2))

    status = "✓  READY FOR PRE-AUTH TOOL"
    if ocr_result["confidence"] < 70:
        status = "⚠  NEEDS HUMAN REVIEW (low OCR confidence)"

    print("\n" + "="*60)
    print(f"  Status : {status}")
    print(f"  Cost   : ${cost:.5f} per form")
    print("="*60)

    return {
        "ocr": {
            "engine": ocr_result["engine"],
            "confidence": ocr_result["confidence"],
            "routing": routing,
        },
        "extracted_data": extracted,
        "cost_usd": cost,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Judi Health IDP Pipeline")
    parser.add_argument("--image", required=True, help="Path to pre-auth form image")
    parser.add_argument("--verbose", action="store_true", help="Print raw OCR text")
    parser.add_argument("--output", help="Save JSON output to file")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: File not found: {args.image}")
        sys.exit(1)

    result = run_pipeline(args.image, verbose=args.verbose)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Output saved to: {args.output}")
