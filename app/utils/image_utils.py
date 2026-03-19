from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np


class ImageValidationError(ValueError):
    pass


def validate_image_upload(content_type: str | None, payload: bytes, max_upload_mb: int) -> None:
    if not payload:
        raise ImageValidationError("Uploaded file is empty")

    if content_type and not content_type.startswith("image/"):
        raise ImageValidationError("Only image uploads are supported")

    max_bytes = max_upload_mb * 1024 * 1024
    if len(payload) > max_bytes:
        raise ImageValidationError(f"Image exceeds {max_upload_mb}MB limit")


def decode_image(payload: bytes) -> np.ndarray:
    nparr = np.frombuffer(payload, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageValidationError("Invalid image data")
    return image


def maybe_save_debug_image(image: np.ndarray, output_path: str) -> Dict[str, Any]:
    ok = cv2.imwrite(output_path, image)
    return {
        "debug_image_saved_as": output_path if ok else None,
        "debug_image_saved": bool(ok),
    }
