from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Shat:aayu Skin AI Service")
    app_version: str = os.getenv("APP_VERSION", "1.0.0")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "8"))
    debug_save_skin: bool = os.getenv("DEBUG_SAVE_SKIN", "false").lower() == "true"
    debug_output_path: str = os.getenv("DEBUG_OUTPUT_PATH", "debug_skin_output.jpg")
    ai_api_key: str = os.getenv("AI_API_KEY", "")
    cors_origins: list[str] = field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
            if origin.strip()
        ]
    )

    # Retained only as metadata for the analysis service.
    labels_json: str = os.getenv(
        "LABELS_JSON",
        json.dumps(["redness", "acne", "texture", "oiliness"]),
    )


settings = Settings()
