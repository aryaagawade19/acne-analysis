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

    # Skin condition classifier settings
    classifier_model_path: str = os.getenv(
        "CLASSIFIER_MODEL_PATH",
        r"C:\Users\ARYA\Downloads\skin_ai_model\models\skin_disease_model.keras",
    )
    classifier_img_size: int = int(os.getenv("CLASSIFIER_IMG_SIZE", "224"))
    classifier_conf_threshold: float = float(os.getenv("CLASSIFIER_CONF_THRESHOLD", "0.60"))
    classifier_unknown_label: str = os.getenv("CLASSIFIER_UNKNOWN_LABEL", "uncertain")

    classifier_labels: list[str] = field(
        default_factory=lambda: json.loads(
            os.getenv(
                "CLASSIFIER_LABELS_JSON",
                '["acne","eczema","fungal","normal","psoriasis","vitiligo"]',
            )
        )
    )


settings = Settings()
