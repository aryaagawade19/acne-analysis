# Shat:aayu Skin AI Service

FastAPI service for skin image analysis, designed to integrate with the main Shat:aayu website.

## Endpoints

- `GET /health`
- `POST /analyze` (backward-compatible)
- `POST /v1/skin-analyze` (recommended for new integrations)
- `POST /v1/skin-condition-classify` (image disease-class classifier)

All image endpoints accept multipart form data with key: `file`.

## Response Shape: Skin Analyze

```json
{
  "message": "Analysis successful",
  "scores": {
    "redness_score": 0.0123,
    "acne_score": 0.0345,
    "acne_count": 3,
    "texture_score": 0.0567
  },
  "classification": {
    "redness_level": "Low",
    "acne_level": "Mild",
    "texture_level": "Normal"
  },
  "dosha_analysis": {
    "dosha_scores": { "vata": 0.2, "pitta": 0.5, "kapha": 0.3 },
    "primary_dosha": "Pitta"
  },
  "analysis_confidence": "Medium",
  "quality": {
    "score": 0.78,
    "level": "High",
    "warnings": []
  },
  "acne_tracking": {
    "severity": "Mild",
    "inflammation_density": 0.012,
    "texture_lesion_density": 0.008,
    "recommended_frequency_days": 3
  }
}
```

## Response Shape: Skin Condition Classify

```json
{
  "message": "Classification successful",
  "classifier": {
    "predicted_label": "eczema",
    "raw_top_label": "eczema",
    "confidence": 0.74,
    "threshold": 0.6,
    "is_confident": true,
    "top_predictions": [
      { "label": "eczema", "probability": 0.74 },
      { "label": "psoriasis", "probability": 0.14 },
      { "label": "fungal", "probability": 0.07 }
    ],
    "all_probabilities": [],
    "disclaimer": "Informational AI output only. Not a medical diagnosis."
  }
}
```

If confidence is below threshold, `predicted_label` becomes `uncertain`.

## Environment Variables

Core:
- `APP_NAME` (default: `Shat:aayu Skin AI Service`)
- `APP_VERSION` (default: `1.0.0`)
- `MAX_UPLOAD_MB` (default: `8`)
- `AI_API_KEY` (optional, enables header auth via `x-ai-api-key`)
- `CORS_ORIGINS` (comma-separated, default: `http://localhost:3000`)
- `DEBUG_SAVE_SKIN` (`true`/`false`, default: `false`)
- `DEBUG_OUTPUT_PATH` (default: `debug_skin_output.jpg`)

Classifier:
- `CLASSIFIER_MODEL_PATH` (default points to your `skin_ai_model` keras file)
- `CLASSIFIER_IMG_SIZE` (default: `224`)
- `CLASSIFIER_CONF_THRESHOLD` (default: `0.60`)
- `CLASSIFIER_UNKNOWN_LABEL` (default: `uncertain`)
- `CLASSIFIER_LABELS_JSON` (default: `["acne","eczema","fungal","normal","psoriasis","vitiligo"]`)

## Local Run

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Merge with Main Website (Next.js)

1. Set backend URL in frontend env:
   - `NEXT_PUBLIC_AI_API_URL=http://localhost:8000`

2. For tracking scan page call:
   - `POST ${AI_API_URL}/v1/skin-analyze`

3. For condition classifier page call:
   - `POST ${AI_API_URL}/v1/skin-condition-classify`

4. Send multipart `file`.

5. If `AI_API_KEY` is set, pass:
   - `x-ai-api-key: <same-key>`

6. Persist for trends:
   - Analyze: `scores.acne_score`, `scores.acne_count`, `scores.redness_score`, `acne_tracking.severity`, `quality.score`
   - Classifier: `predicted_label`, `confidence`, `top_predictions`

## Evaluation Quality Note (from your confusion matrix)

Current classifier performance is not clinical-grade yet.
- Approx overall accuracy: ~49% (503/1023)
- Strong class confusion between eczema/fungal/psoriasis/vitiligo
- `normal` appears strong but has very low support (~20 in test), so it is unstable

Use this classifier as assistive triage only, not diagnosis.

## Production Notes

- Keep `DEBUG_SAVE_SKIN=false` in production.
- Put the AI service behind HTTPS + rate limiting.
- Use Next.js server routes as proxy to avoid exposing secrets.
- Show medical disclaimer on every AI output.
