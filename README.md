# Judi Health — Intelligent Document Processing (IDP)
### Interview Preparation Prototype · April 2026

---

## What this is

A working prototype of an Intelligent Document Processing pipeline for medication prior authorization forms, built in preparation for an interview with Judi Health.

**Goal:** Develop an understanding of the IDP product — its technical challenges, failure modes, and the tradeoffs between cost and accuracy.

**Note:** This code was not optimized for cleanliness or production readiness. It was written to learn quickly, hit real failure modes, and build intuition for the problem. Comments throughout document what worked, what didn't, and why.

---

## What it does

Extracts structured data from a handwritten medication prior authorization form using a two-layer pipeline:

```
Phone photo / fax scan
        ↓
Layer 1 — Google Document AI Form Parser
  Returns key-value pairs directly from the form
  Per-field confidence scores
        ↓
Layer 2 — Claude Haiku (vision)
  Structures ambiguous fields
  Returns JSON with confidence + human review flags
        ↓
Web UI
  Shows extracted fields, confidence scores, layer attribution, review flags
```

A third proprietary layer exists in Judi Health's actual product. This prototype does not attempt to replicate it.

---

## What I learned

**EasyOCR failed badly.** Started with Docling + EasyOCR — open-source, no API cost. On a phone photo of a handwritten form it returned "1030 N" instead of "2030 N Adams Street" and got the date of birth wrong. EasyOCR is trained on printed text and struggles with handwriting at low DPI.
 
**Google Document AI worked.** The same photo that broke EasyOCR was correctly parsed — address, date of birth, prescriber details, and medication all extracted correctly.
 
**Confidence score ≠ accuracy.** You cannot set a universal confidence threshold across models or field types. I believe that the right approach is an empirical calibration study: run labeled forms through the pipeline, plot confidence vs actual accuracy per field type, and derive thresholds from data.
 
**Some fields consistently underperform.** Checkboxes and radio buttons (like gender) returned lower confidence across all runs. Handwriting degradation also mattered: member ID fields started confusing "13" and "B" as handwriting got sloppier across multiple form fills.
 
**Field labels vary across payers.** Tested across five forms: RxBenefits, CareSource, Aetna, Cigna, and UnitedHealthcare. Gender exists as a checkbox on some forms and is absent on others. The prototype incorrectly flagged it as missing on forms that simply didn't have the field. 

---

## Stack

| Layer | Tool |
|---|---|
| OCR | Google Document AI Form Parser |
| LLM extraction | Claude Haiku (`claude-haiku-4-5`) |
| Fallback OCR | Docling + EasyOCR (limited handwriting support) |
| Web server | Flask |
| Frontend | Vanilla HTML/CSS/JS |

---

## Setup

### 1. Clone and create virtual environment
```bash
git clone https://github.com/ritaaoki/judi_health_idp.git
cd judi_health_idp
python -m venv .idp
.idp\Scripts\activate   # Windows
source .idp/bin/activate  # Mac/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up Google Document AI
1. Create a GCP project at [console.cloud.google.com](https://console.cloud.google.com)
2. Enable the Document AI API
3. Create a Form Parser processor
4. Authenticate locally:
```bash
gcloud auth application-default login
```

### 4. Configure environment variables
```bash
cp .env.example .env
```
Fill in your values:
```
ANTHROPIC_API_KEY=sk-ant-...
GCP_PROJECT_ID=your-project-id
GCP_LOCATION=us
GCP_PROCESSOR_ID=your-processor-id
```

### 5. Run
```bash
python server.py
```
Open [http://localhost:5000](http://localhost:5000) and upload a pre-auth form.

---

## Project structure

```
judi_health_idp/
├── server.py               # Flask server — pipeline orchestrator
├── ocr.py                  # Layer 1: Google Document AI + fallback chain
├── llm.py                  # Layer 2: Claude Haiku extraction
├── validate.py             # Medication database cross-reference
├── index.html              # Frontend
├── requirements.txt
└──  dummy_data/            # Test forms across 5 payers
```

---

## Known limitations

- **Image quality** — OCR accuracy degrades below 300 DPI. Production would need image preprocessing or higher-quality input.
- **Handwriting variation** — tested on relatively neat handwriting. Real prescribers write under time pressure; accuracy drops without fine-tuning on real prescriber data.
- **Confidence thresholds not empirically validated** — set based on reasoning, not a calibration study against labeled ground truth.
- **Payer form variation** — built for the RxBenefits pre-auth form. 
- **Local only** — production deployment would require encrypted storage, a BAA with Google Document AI, audit logging, and strict access controls.


---

*Built in ~1 day as interview preparation. Optimized for learning, not clean code.*
