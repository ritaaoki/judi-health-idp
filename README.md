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

**EasyOCR failed badly.** I started with Docling + EasyOCR — open-source, no API cost. It missed most fields on a phone photo of a handwritten form. Returned "1030 N" instead of "2030 N Adams Street", got the date of birth wrong, with a reported confidence of 94%.

The core insight: **self-reported confidence scores reflect how certain a model is about what it saw — not whether what it saw is correct.** EasyOCR was very confident it read "3030". It was wrong.

**Google Document AI worked.** The same photo that broke EasyOCR was correctly parsed by Document AI's Form Parser — address, date of birth, prescriber details, medication all extracted correctly.

**Confidence calibration is the real problem.** EasyOCR reported 95% confidence on a wrong address. Document AI reported 60% confidence on a gender checkbox and got it right. You cannot set a universal confidence threshold across models or field types. The right approach is an empirical calibration study: run labeled forms through the pipeline, plot confidence vs actual accuracy per field type, and derive thresholds from data rather than intuition.

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
├── .env.example            # Environment variable template
└── INTERVIEW_PREP_OCR.md   # OCR evaluation notes and training strategy
```

---

## Known limitations

- **Image quality** — OCR accuracy degrades below 300 DPI. Production would need image preprocessing or higher-quality input.
- **Handwriting variation** — tested on neat handwriting only. Real prescribers write under time pressure; accuracy would drop without fine-tuning on real prescriber data.
- **Confidence thresholds not empirically validated** — set based on reasoning, not a calibration study against labeled ground truth.
- **Not HIPAA compliant** — production deployment would require encrypted storage, a BAA with Google, audit logging, and strict access controls. This prototype runs locally only.
- **Single form type assumed** — built for one specific pre-auth form. Mixed document types would require a classification layer first.

---

## If I had more time

- Run a confidence calibration study on 500+ labeled forms
- Fine-tune Document AI on real prescriber handwriting (redacted forms)
- Build a human review queue where every correction becomes a training example
- Evaluate AWS Textract and Azure Document Intelligence head-to-head
- Add image preprocessing for low-quality inputs

---

*Built in ~1 day as interview preparation. Optimized for learning, not clean code.*
