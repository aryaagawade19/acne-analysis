from __future__ import annotations

import cv2
import numpy as np


def create_skin_mask(image: np.ndarray, landmarks: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    face_outline_indices = [
        10, 338, 297, 332, 284, 251, 389, 356, 454,
        323, 361, 288, 397, 365, 379, 378, 400,
        377, 152, 148, 176, 149, 150, 136, 172,
        58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    ]

    face_outline = np.array([landmarks[i] for i in face_outline_indices], dtype=np.int32)
    cv2.fillPoly(mask, [face_outline], 255)

    left_eye = np.array([landmarks[i] for i in [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155]], dtype=np.int32)
    right_eye = np.array([landmarks[i] for i in [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382]], dtype=np.int32)
    lips = np.array([landmarks[i] for i in [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318]], dtype=np.int32)

    cv2.fillPoly(mask, [left_eye], 0)
    cv2.fillPoly(mask, [right_eye], 0)
    cv2.fillPoly(mask, [lips], 0)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return cv2.bitwise_and(image, image, mask=mask)


def normalize_lighting(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
