from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.config import settings


class ClassifierError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load_model():
    model_path = Path(settings.classifier_model_path)
    if not model_path.exists():
        raise ClassifierError(f"Classifier model not found: {model_path}")

    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover
        raise ClassifierError(
            "TensorFlow not installed. Install it in this venv to enable classification."
        ) from exc

    try:
        return tf.keras.models.load_model(str(model_path))
    except Exception as exc:  # pragma: no cover
        raise ClassifierError(f"Failed to load classifier model: {exc}") from exc


def _prepare(image: np.ndarray) -> np.ndarray:
    size = int(settings.classifier_img_size)
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr


def classify_skin_condition(image: np.ndarray) -> dict[str, Any]:
    model = _load_model()
    x = _prepare(image)

    probs = model.predict(x, verbose=0)[0]
    labels = settings.classifier_labels

    if len(labels) != len(probs):
        raise ClassifierError(
            f"Label count ({len(labels)}) does not match model outputs ({len(probs)})"
        )

    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    top_label = labels[top_idx]

    ranked = sorted(
        [{"label": labels[i], "probability": round(float(p), 4)} for i, p in enumerate(probs)],
        key=lambda x: x["probability"],
        reverse=True,
    )

    confident = top_prob >= settings.classifier_conf_threshold
    predicted_label = top_label if confident else settings.classifier_unknown_label

    return {
        "predicted_label": predicted_label,
        "raw_top_label": top_label,
        "confidence": round(top_prob, 4),
        "threshold": settings.classifier_conf_threshold,
        "is_confident": confident,
        "top_predictions": ranked[:3],
        "all_probabilities": ranked,
        "disclaimer": "Informational AI output only. Not a medical diagnosis.",
    }
