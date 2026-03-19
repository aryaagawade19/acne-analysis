from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

from app.services.acne_detector import calculate_acne_metrics, calculate_texture_score
from app.services.face_detector import extract_face_landmarks
from app.services.redness_detector import calculate_redness_score
from app.services.skin_mask import create_skin_mask, normalize_lighting


class AnalysisError(ValueError):
    pass


def generate_skin_profile(redness: float, acne_score: float, texture: float) -> Dict[str, str]:
    return {
        "redness_level": "Low" if redness < 0.035 else "Moderate" if redness < 0.08 else "High",
        "acne_level": "Clear" if acne_score < 0.015 else "Mild" if acne_score < 0.06 else "Moderate",
        "texture_level": "Smooth" if texture < 0.025 else "Normal" if texture < 0.07 else "Rough",
    }


def calculate_dosha_profile(redness: float, texture: float, acne_score: float) -> Dict[str, Any]:
    redness_norm = min(redness / 0.6, 1.0)
    texture_norm = min(texture / 0.3, 1.0)
    acne_norm = min(acne_score / 0.6, 1.0)

    vata_score = texture_norm * 0.7
    pitta_score = redness_norm * 0.6 + acne_norm * 0.4
    kapha_score = acne_norm * 0.6

    total = vata_score + pitta_score + kapha_score + 1e-6
    scores = {
        "vata": round(vata_score / total, 4),
        "pitta": round(pitta_score / total, 4),
        "kapha": round(kapha_score / total, 4),
    }

    return {
        "dosha_scores": scores,
        "primary_dosha": max(scores, key=scores.get).capitalize(),
    }


def assess_scan_quality(image: np.ndarray, skin: np.ndarray) -> Dict[str, Any]:
    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_skin = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    skin_mask = (gray_skin > 0).astype(np.uint8)

    if np.count_nonzero(skin_mask) == 0:
        return {
            "quality_score": 0.0,
            "quality_level": "Low",
            "warnings": ["Low face/skin visibility"],
            "brightness": 0.0,
            "contrast": 0.0,
            "sharpness": 0.0,
            "face_coverage": 0.0,
        }

    skin_pixels = gray_skin[skin_mask > 0]

    brightness = float(np.mean(skin_pixels))
    contrast = float(np.std(skin_pixels))
    sharpness = float(cv2.Laplacian(gray_full, cv2.CV_64F).var())
    face_coverage = float(np.count_nonzero(skin_mask) / skin_mask.size)

    brightness_score = 1.0 - min(abs(brightness - 125.0) / 125.0, 1.0)
    contrast_score = min(contrast / 55.0, 1.0)
    sharpness_score = min(sharpness / 180.0, 1.0)
    coverage_score = min(face_coverage / 0.32, 1.0)

    quality_score = (
        (brightness_score * 0.25)
        + (contrast_score * 0.2)
        + (sharpness_score * 0.3)
        + (coverage_score * 0.25)
    )

    warnings: list[str] = []
    if brightness < 70:
        warnings.append("Image too dark")
    elif brightness > 190:
        warnings.append("Image too bright")

    if sharpness < 80:
        warnings.append("Image appears blurry")

    if face_coverage < 0.15:
        warnings.append("Move face closer to camera")

    if contrast < 18:
        warnings.append("Low contrast lighting")

    if quality_score >= 0.75:
        quality_level = "High"
    elif quality_score >= 0.5:
        quality_level = "Medium"
    else:
        quality_level = "Low"

    return {
        "quality_score": round(float(quality_score), 4),
        "quality_level": quality_level,
        "warnings": warnings,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
        "face_coverage": round(face_coverage, 4),
    }


def calculate_confidence(total_pixels: int, quality_score: float) -> str:
    pixel_conf = 1.0 if total_pixels > 50000 else 0.65 if total_pixels > 25000 else 0.35
    combined = (pixel_conf * 0.6) + (quality_score * 0.4)

    if combined >= 0.75:
        return "High"
    if combined >= 0.5:
        return "Medium"
    return "Low"


def _recommended_frequency_days(severity: str) -> int:
    if severity == "High":
        return 3
    if severity == "Moderate":
        return 5
    if severity == "Mild":
        return 7
    return 10


def analyze_skin_image(image: np.ndarray) -> tuple[Dict[str, Any], np.ndarray]:
    landmarks = extract_face_landmarks(image)
    if not landmarks:
        raise AnalysisError("No face detected")

    raw_skin = create_skin_mask(image, landmarks)
    skin = normalize_lighting(raw_skin)

    redness = calculate_redness_score(raw_skin)
    acne_metrics = calculate_acne_metrics(raw_skin)
    acne_score = float(acne_metrics["acne_score"])
    acne_count = int(acne_metrics["acne_count"])
    texture = calculate_texture_score(skin)

    profile = generate_skin_profile(redness, acne_score, texture)
    dosha = calculate_dosha_profile(redness, texture, acne_score)

    total_pixels = int(np.count_nonzero(cv2.cvtColor(raw_skin, cv2.COLOR_BGR2GRAY)))
    quality = assess_scan_quality(image, raw_skin)
    confidence = calculate_confidence(total_pixels, float(quality["quality_score"]))
    recommended_frequency_days = _recommended_frequency_days(str(acne_metrics["severity"]))

    result = {
        "message": "Analysis successful",
        "scores": {
            "redness_score": redness,
            "acne_score": acne_score,
            "acne_count": acne_count,
            "texture_score": texture,
            "oiliness_score": acne_metrics["oiliness_score"],
        },
        "classification": profile,
        "dosha_analysis": dosha,
        "analysis_confidence": confidence,
        "quality": {
            "score": quality["quality_score"],
            "level": quality["quality_level"],
            "warnings": quality["warnings"],
        },
        "detected_concerns": acne_metrics["acne_types"],
        "zone_analysis": acne_metrics["zone_analysis"],
        "acne_tracking": {
            "severity": acne_metrics["severity"],
            "inflammation_density": acne_metrics["inflammation_density"],
            "texture_lesion_density": acne_metrics["texture_lesion_density"],
            "recommended_frequency_days": recommended_frequency_days,
        },
    }

    return result, raw_skin
