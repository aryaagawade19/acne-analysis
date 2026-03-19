from __future__ import annotations

import cv2
import numpy as np


ZONE_DEFINITIONS = (
    ("Forehead", (0.20, 0.02, 0.80, 0.24)),
    ("Nose / T-Zone", (0.40, 0.22, 0.60, 0.58)),
    ("Left Cheek", (0.10, 0.30, 0.36, 0.66)),
    ("Right Cheek", (0.64, 0.30, 0.90, 0.66)),
    ("Chin", (0.36, 0.72, 0.64, 0.92)),
    ("Jawline", (0.12, 0.76, 0.88, 0.96)),
)


def _skin_binary_mask(skin: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    mask = (gray > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _severity_from_score(score: float, lesion_count: int) -> str:
    if lesion_count == 0 and score < 0.015:
        return "Clear"
    if score < 0.04:
        return "Mild"
    if score < 0.09:
        return "Moderate"
    return "High"


def _zone_severity(score: int) -> str:
    if score >= 60:
        return "severe"
    if score >= 30:
        return "moderate"
    if score >= 10:
        return "mild"
    return "clear"


def _glare_mask(hsv: np.ndarray, skin_mask: np.ndarray) -> np.ndarray:
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)

    sat_pixels = saturation[skin_mask > 0]
    val_pixels = value[skin_mask > 0]
    if sat_pixels.size == 0:
        return np.zeros_like(skin_mask)

    sat_cap = min(45.0, float(np.percentile(sat_pixels, 22)))
    value_floor = max(210.0, float(np.percentile(val_pixels, 92)))

    glare = ((value >= value_floor) & (saturation <= sat_cap)).astype(np.uint8) * 255
    glare = cv2.bitwise_and(glare, skin_mask)
    glare = cv2.GaussianBlur(glare, (5, 5), 0)
    glare = cv2.threshold(glare, 80, 255, cv2.THRESH_BINARY)[1]
    return glare


def _inflammation_mask(bgr: np.ndarray, hsv: np.ndarray, skin_mask: np.ndarray, glare_mask: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    a_pixels = a_channel[skin_mask > 0]
    sat_pixels = sat[skin_mask > 0]
    val_pixels = val[skin_mask > 0]

    if a_pixels.size == 0:
        return np.zeros_like(skin_mask)

    baseline_a = float(np.median(a_pixels))
    mad_a = float(np.median(np.abs(a_pixels - baseline_a)))
    local_baseline = cv2.GaussianBlur(a_channel, (0, 0), sigmaX=9, sigmaY=9)
    redness_delta = a_channel - local_baseline

    global_margin = max(6.0, mad_a * 2.2)
    local_margin = max(3.2, mad_a * 1.3)
    sat_floor = max(42.0, float(np.percentile(sat_pixels, 55)))
    val_floor = max(55.0, float(np.percentile(val_pixels, 25)))

    inflammation_mask = (
        (a_channel > (baseline_a + global_margin))
        & (redness_delta > local_margin)
        & (sat >= sat_floor)
        & (val >= val_floor)
        & (val <= 242)
        & (skin_mask > 0)
    ).astype(np.uint8) * 255

    inflammation_mask = cv2.bitwise_and(inflammation_mask, cv2.bitwise_not(glare_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inflammation_mask = cv2.medianBlur(inflammation_mask, 5)
    inflammation_mask = cv2.morphologyEx(inflammation_mask, cv2.MORPH_OPEN, kernel)
    inflammation_mask = cv2.morphologyEx(inflammation_mask, cv2.MORPH_CLOSE, kernel)
    return inflammation_mask


def _texture_masks(gray: np.ndarray, skin_mask: np.ndarray, glare_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray_base = cv2.bilateralFilter(gray, 7, 35, 35)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray_base)

    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tophat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    blackhat = cv2.morphologyEx(gray_eq, cv2.MORPH_BLACKHAT, blackhat_kernel)
    tophat = cv2.morphologyEx(gray_eq, cv2.MORPH_TOPHAT, tophat_kernel)

    blackhat = cv2.bitwise_and(blackhat, blackhat, mask=skin_mask)
    tophat = cv2.bitwise_and(tophat, tophat, mask=skin_mask)

    blackhat_values = blackhat[skin_mask > 0]
    tophat_values = tophat[skin_mask > 0]
    blackhat_thresh = int(max(24, np.percentile(blackhat_values, 94))) if blackhat_values.size else 24
    tophat_thresh = int(max(22, np.percentile(tophat_values, 96))) if tophat_values.size else 22

    _, comedone_mask = cv2.threshold(blackhat, blackhat_thresh, 255, cv2.THRESH_BINARY)
    _, whitehead_mask = cv2.threshold(tophat, tophat_thresh, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    comedone_mask = cv2.morphologyEx(comedone_mask, cv2.MORPH_OPEN, kernel)
    whitehead_mask = cv2.morphologyEx(whitehead_mask, cv2.MORPH_OPEN, kernel)

    inverse_glare = cv2.bitwise_not(glare_mask)
    comedone_mask = cv2.bitwise_and(comedone_mask, inverse_glare)
    whitehead_mask = cv2.bitwise_and(whitehead_mask, inverse_glare)
    return comedone_mask, whitehead_mask


def _bounding_rect(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def _zone_masks(skin_mask: np.ndarray) -> dict[str, np.ndarray]:
    x, y, w, h = _bounding_rect(skin_mask)
    if w <= 1 or h <= 1:
        return {}

    zone_masks: dict[str, np.ndarray] = {}
    for zone_name, (x1r, y1r, x2r, y2r) in ZONE_DEFINITIONS:
        zone_mask = np.zeros_like(skin_mask)
        zx1 = int(x + w * x1r)
        zy1 = int(y + h * y1r)
        zx2 = int(x + w * x2r)
        zy2 = int(y + h * y2r)
        cv2.rectangle(zone_mask, (zx1, zy1), (zx2, zy2), 255, -1)
        zone_masks[zone_name] = cv2.bitwise_and(zone_mask, skin_mask)

    # Make jawline exclude the chin center so chin lesions don't get misreported as jawline.
    if "Jawline" in zone_masks and "Chin" in zone_masks:
        zone_masks["Jawline"] = cv2.bitwise_and(zone_masks["Jawline"], cv2.bitwise_not(zone_masks["Chin"]))

    return zone_masks


def _assign_zone(lesion_mask: np.ndarray, zone_masks: dict[str, np.ndarray]) -> str | None:
    best_zone: str | None = None
    best_overlap = 0

    for zone_name, zone_mask in zone_masks.items():
        overlap = int(np.count_nonzero(cv2.bitwise_and(lesion_mask, zone_mask)))
        if overlap > best_overlap:
            best_overlap = overlap
            best_zone = zone_name

    return best_zone if best_overlap > 0 else None


def _zone_analysis(lesions: list[dict[str, float | str | np.ndarray]], skin_mask: np.ndarray) -> list[dict[str, float | str]]:
    zone_masks = _zone_masks(skin_mask)
    if not zone_masks:
        return []

    zone_totals = {zone_name: 0.0 for zone_name in zone_masks.keys()}
    total_weight = max(sum(float(lesion["weight"]) for lesion in lesions), 1.0)

    for lesion in lesions:
        lesion_mask = lesion.get("mask")
        if lesion_mask is None:
            continue
        zone_name = _assign_zone(lesion_mask, zone_masks)
        if zone_name is not None:
            zone_totals[zone_name] += float(lesion["weight"])

    output: list[dict[str, float | str]] = []
    for zone_name in ("Forehead", "Nose / T-Zone", "Left Cheek", "Right Cheek", "Chin", "Jawline"):
        zone_weight = zone_totals.get(zone_name, 0.0)
        score = int(min(100, round((zone_weight / total_weight) * 100)))
        output.append({
            "zone": zone_name,
            "score": score,
            "severity": _zone_severity(score),
        })

    return output


def _classify_lesions(
    contours: list[np.ndarray],
    inflammation_mask: np.ndarray,
    comedone_mask: np.ndarray,
    whitehead_mask: np.ndarray,
    image_area: int,
) -> tuple[list[dict[str, float | str | np.ndarray]], dict[str, int], float]:
    lesions: list[dict[str, float | str | np.ndarray]] = []
    lesion_type_counts = {"Inflammatory": 0, "Comedonal": 0, "Cystic": 0}
    lesion_weight_sum = 0.0
    min_area = max(18, int(image_area * 0.00002))
    max_area = max(180, int(image_area * 0.0035))

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        circularity = (4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / max(h, 1)
        contour_mask = np.zeros(inflammation_mask.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)

        contour_pixels = max(np.count_nonzero(contour_mask), 1)
        redness_ratio = float(np.count_nonzero(cv2.bitwise_and(inflammation_mask, contour_mask)) / contour_pixels)
        comedone_ratio = float(np.count_nonzero(cv2.bitwise_and(comedone_mask, contour_mask)) / contour_pixels)
        whitehead_ratio = float(np.count_nonzero(cv2.bitwise_and(whitehead_mask, contour_mask)) / contour_pixels)
        texture_ratio = comedone_ratio + whitehead_ratio
        solidity = area / max(float(w * h), 1.0)

        if area > max(60, image_area * 0.00045) and redness_ratio > 0.5 and solidity > 0.35:
            lesion_type = "Cystic"
            weight = 2.1
        elif redness_ratio > 0.32 and circularity > 0.25 and solidity > 0.3:
            lesion_type = "Inflammatory"
            weight = 1.35
        elif texture_ratio > 0.34 and circularity > 0.2 and solidity > 0.28 and 0.55 <= aspect_ratio <= 1.8:
            lesion_type = "Comedonal"
            weight = 1.0
        else:
            continue

        lesion_type_counts[lesion_type] += 1
        lesion_weight_sum += weight
        lesions.append({
            "type": lesion_type,
            "cx": x + (w / 2.0),
            "cy": y + (h / 2.0),
            "weight": weight,
            "mask": contour_mask,
        })

    return lesions, lesion_type_counts, lesion_weight_sum


def _calculate_oiliness_score(hsv: np.ndarray, skin_mask: np.ndarray, glare_mask: np.ndarray) -> float:
    skin_pixels = int(np.count_nonzero(skin_mask))
    if skin_pixels == 0:
        return 0.0

    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)
    sat_pixels = saturation[skin_mask > 0]
    val_pixels = value[skin_mask > 0]

    sat_median = float(np.median(sat_pixels)) if sat_pixels.size else 0.0
    val_high = float(np.percentile(val_pixels, 90)) if val_pixels.size else 0.0
    highlight_ratio = float(np.count_nonzero(glare_mask) / skin_pixels)

    shine_strength = np.clip((val_high - 170.0) / 70.0, 0.0, 1.0)
    low_sat_factor = np.clip((50.0 - sat_median) / 50.0, 0.0, 1.0)

    score = min((highlight_ratio / 0.12) * 0.65 + shine_strength * 0.2 + low_sat_factor * 0.15, 1.0)
    if highlight_ratio < 0.01 and shine_strength < 0.18:
        return 0.0

    return round(float(score), 4)


def calculate_acne_metrics(skin: np.ndarray) -> dict[str, float | int | str | list[dict[str, float | str]] | list[str]]:
    filtered_skin = cv2.bilateralFilter(skin, 7, 35, 35)
    hsv = cv2.cvtColor(filtered_skin, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(filtered_skin, cv2.COLOR_BGR2GRAY)
    skin_mask = _skin_binary_mask(filtered_skin)
    total_skin_pixels = int(np.count_nonzero(skin_mask))

    if total_skin_pixels == 0:
        return {
            "acne_score": 0.0,
            "acne_count": 0,
            "inflammation_density": 0.0,
            "texture_lesion_density": 0.0,
            "severity": "Clear",
            "acne_types": [],
            "zone_analysis": [],
            "oiliness_score": 0.0,
        }

    glare_mask = _glare_mask(hsv, skin_mask)
    inflammation_mask = _inflammation_mask(filtered_skin, hsv, skin_mask, glare_mask)
    comedone_mask, whitehead_mask = _texture_masks(gray, skin_mask, glare_mask)

    combined_mask = cv2.bitwise_or(inflammation_mask, comedone_mask)
    combined_mask = cv2.bitwise_or(combined_mask, whitehead_mask)
    combined_mask = cv2.bitwise_and(combined_mask, skin_mask)
    combined_mask = cv2.morphologyEx(
        combined_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    inflamed_pixels = int(np.count_nonzero(inflammation_mask))
    texture_pixels = int(np.count_nonzero(cv2.bitwise_or(comedone_mask, whitehead_mask)))

    inflammation_density = inflamed_pixels / total_skin_pixels
    texture_density = texture_pixels / total_skin_pixels
    oiliness_score = _calculate_oiliness_score(hsv, skin_mask, glare_mask)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = filtered_skin.shape[0] * filtered_skin.shape[1]
    lesions, lesion_type_counts, lesion_weight_sum = _classify_lesions(contours, inflammation_mask, comedone_mask, whitehead_mask, image_area)
    acne_count = len(lesions)

    weak_signal = acne_count < 2 and inflammation_density < 0.018 and texture_density < 0.028
    if weak_signal:
        acne_score = 0.0
        acne_count = 0
        lesion_type_counts = {"Inflammatory": 0, "Comedonal": 0, "Cystic": 0}
        lesions = []
    else:
        count_factor = min(acne_count / 55.0, 1.0)
        lesion_weight_factor = min(lesion_weight_sum / 70.0, 1.0)
        inflammation_factor = min(inflammation_density / 0.09, 1.0)
        texture_factor = min(texture_density / 0.075, 1.0)
        acne_score = min(
            (inflammation_factor * 0.36)
            + (texture_factor * 0.16)
            + (count_factor * 0.18)
            + (lesion_weight_factor * 0.30),
            1.0,
        )

    severity = _severity_from_score(acne_score, acne_count)
    acne_types = [key for key, count in lesion_type_counts.items() if count > 0]
    zone_analysis = _zone_analysis(lesions, skin_mask)

    return {
        "acne_score": round(float(acne_score), 4),
        "acne_count": int(acne_count),
        "inflammation_density": round(float(inflammation_density), 4),
        "texture_lesion_density": round(float(texture_density), 4),
        "severity": severity,
        "acne_types": acne_types,
        "zone_analysis": zone_analysis,
        "oiliness_score": oiliness_score,
    }


def calculate_acne_score(skin: np.ndarray) -> tuple[float, int]:
    metrics = calculate_acne_metrics(skin)
    return float(metrics["acne_score"]), int(metrics["acne_count"])


def calculate_texture_score(skin: np.ndarray) -> float:
    gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)

    if gray.size == 0:
        return 0.0

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    center = gray[1:-1, 1:-1]
    lbp = np.zeros_like(center, dtype=np.uint8)

    lbp |= ((gray[:-2, :-2] > center).astype(np.uint8) << 7)
    lbp |= ((gray[:-2, 1:-1] > center).astype(np.uint8) << 6)
    lbp |= ((gray[:-2, 2:] > center).astype(np.uint8) << 5)
    lbp |= ((gray[1:-1, 2:] > center).astype(np.uint8) << 4)
    lbp |= ((gray[2:, 2:] > center).astype(np.uint8) << 3)
    lbp |= ((gray[2:, 1:-1] > center).astype(np.uint8) << 2)
    lbp |= ((gray[2:, :-2] > center).astype(np.uint8) << 1)
    lbp |= ((gray[1:-1, :-2] > center).astype(np.uint8) << 0)

    high_variation = np.count_nonzero(lbp > 168)
    total_pixels = lbp.size

    if total_pixels == 0:
        return 0.0

    return round(float(high_variation / total_pixels), 4)
