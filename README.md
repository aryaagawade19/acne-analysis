# Shat:aayu Skin AI Service

FastAPI service for acne-focused skin image analysis, designed to integrate with the main Shat:aayu website.

## Endpoints

- `GET /health`
- `POST /analyze` (backward-compatible)
- `POST /v1/skin-analyze` (recommended)

All image endpoints accept multipart form data with key: `file`.

## Response Shape

```json
{
  "message": "Analysis successful",
  "scores": {
    "redness_score": 0.0123,
    "acne_score": 0.0345,
    "acne_count": 3,
    "texture_score": 0.0567,
    "oiliness_score": 0.0412
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
  "detected_concerns": ["Inflammatory"],
  "zone_analysis": [{ "zone": "Left Cheek", "score": 55, "severity": "moderate" }],
  "acne_tracking": {
    "severity": "Mild",
    "inflammation_density": 0.012,
    "texture_lesion_density": 0.008,
    "recommended_frequency_days": 3
  }
}
```

## Environment Variables

- `APP_NAME` (default: `Shat:aayu Skin AI Service`)
- `APP_VERSION` (default: `1.0.0`)
- `MAX_UPLOAD_MB` (default: `8`)
- `AI_API_KEY` (optional, enables header auth via `x-ai-api-key`)
- `CORS_ORIGINS` (comma-separated, default: `http://localhost:3000`)
- `DEBUG_SAVE_SKIN` (`true`/`false`, default: `false`)
- `DEBUG_OUTPUT_PATH` (default: `debug_skin_output.jpg`)

## Local Run

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Frontend Integration

For acne analysis page call:
- `POST ${ACNE_AI_API_URL}/v1/skin-analyze`

Send multipart key:
- `file`

If `AI_API_KEY` is set, pass:
- `x-ai-api-key: <same-key>`

Persist for trends:
- `scores.acne_score`
- `scores.acne_count`
- `scores.redness_score`
- `scores.oiliness_score`
- `acne_tracking.severity`
- `quality.score`
- `zone_analysis`

## Render Deployment

Build command:
```bash
pip install -r requirements.txt
```

Start command:
```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## Production Notes

- Keep `DEBUG_SAVE_SKIN=false` in production.
- Put the service behind HTTPS and basic rate limiting.
- Use your Next.js server routes as the public-facing proxy so the backend URL stays server-side.
- Show a medical disclaimer in the frontend because this is tracking support, not diagnosis.
