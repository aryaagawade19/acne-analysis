from __future__ import annotations

import cv2
import numpy as np


def _skin_mask_from_image(skin: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    mask = (gray > 0).astype(np.uint8)
    if np.count_nonzero(mask) == 0:
        return mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def calculate_redness_score(skin: np.ndarray) -> float:
    skin_mask = _skin_mask_from_image(skin)
    total_pixels = int(np.count_nonzero(skin_mask))
    if total_pixels == 0:
        return 0.0

    denoised = cv2.bilateralFilter(skin, 7, 40, 40)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)

    a_channel = lab[:, :, 1].astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)

    a_pixels = a_channel[skin_mask > 0]
    sat_pixels = saturation[skin_mask > 0]
    val_pixels = value[skin_mask > 0]

    baseline_a = float(np.median(a_pixels))
    mad_a = float(np.median(np.abs(a_pixels - baseline_a)))
    local_baseline = cv2.GaussianBlur(a_channel, (0, 0), sigmaX=11, sigmaY=11)
    redness_delta = a_channel - local_baseline

    global_margin = max(5.5, mad_a * 2.0)
    local_margin = max(2.8, mad_a * 1.2)
    sat_floor = max(32.0, float(np.percentile(sat_pixels, 40)))
    val_floor = max(52.0, float(np.percentile(val_pixels, 20)))

    redness_mask = (
        (a_channel > (baseline_a + global_margin))
        & (redness_delta > local_margin)
        & (saturation >= sat_floor)
        & (value >= val_floor)
        & (value <= 240)
        & (skin_mask > 0)
    )

    red_pixels = int(np.count_nonzero(redness_mask))
    if red_pixels == 0:
        return 0.0

    redness_strength = np.clip((a_channel[redness_mask] - local_baseline[redness_mask]) / 16.0, 0.0, 1.0)
    coverage = red_pixels / total_pixels
    severity = float(np.mean(redness_strength)) if redness_strength.size else 0.0

    if coverage < 0.008 and severity < 0.22:
        return 0.0

    score = min((coverage * 0.55) + (severity * 0.45), 1.0)
    return round(float(score), 4)
