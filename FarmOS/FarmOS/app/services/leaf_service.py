from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from app.services.groq_service import get_ai_explanation
from app.services.groundwater_service import get_groundwater_stats
from app.models.schemas import LeafAnalysisResponse, HealthLabel

logger = logging.getLogger(__name__)

# Threshold — tweak this if needed
GREEN_PIXEL_THRESHOLD = 0.40   # 40% of image must be green to be healthy

# HSV green range
# Hue 35-85 covers yellow-green to pure green to blue-green
# Saturation > 40, Value > 40 filters out grey/white/dark noise
HSV_GREEN_LOWER = np.array([35,  40,  40], dtype=np.uint8)
HSV_GREEN_UPPER = np.array([85, 255, 255], dtype=np.uint8)


def analyse_leaf(
    image_bytes:       bytes,
    wind_force_ms:     Optional[float] = None,
    humidity_pct:      Optional[float] = None,
    temperature_c:     Optional[float] = None,
    sunlight_lux:      Optional[float] = None,
    users_question:    Optional[str]   = None,
    selected_language: str             = "English",
    latitude:          Optional[float] = None,
    longitude:         Optional[float] = None,
) -> LeafAnalysisResponse:

    # 1. Decode image
    arr     = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image. Unsupported format or corrupted file.")

    h, w = img_bgr.shape[:2]
    total_pixels = h * w
    logger.info("Image loaded: %d x %d px (%d total pixels)", w, h, total_pixels)

    # 2. Count green pixels via HSV masking
    hsv         = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    green_mask  = cv2.inRange(hsv, HSV_GREEN_LOWER, HSV_GREEN_UPPER)
    green_count = int(cv2.countNonZero(green_mask))
    green_ratio = green_count / total_pixels

    logger.info("Green pixels: %d  (%.1f%% of image)", green_count, green_ratio * 100)

    # 3. One rule — green enough = healthy
    is_healthy    = green_ratio >= GREEN_PIXEL_THRESHOLD
    health_label  = HealthLabel.HEALTHY if is_healthy else HealthLabel.NOT_HEALTHY
    health_status = "HEALTHY" if is_healthy else "NOT HEALTHY"

    if is_healthy:
        recommendation = (
            f"Plant looks healthy. {green_ratio*100:.1f}% of the image is green. "
            "Maintain your current watering and fertilisation schedule."
        )
    else:
        recommendation = (
            f"Plant does not look healthy. Only {green_ratio*100:.1f}% of the image is green "
            f"(minimum threshold is {GREEN_PIXEL_THRESHOLD*100:.0f}%). "
            "Check for yellowing, browning, drought stress, or disease."
        )

    logger.info("Health verdict: %s (green=%.1f%%)", health_status, green_ratio * 100)

    # 4. Fetch groundwater stats if coordinates provided
    groundwater_stats = None
    if latitude is not None and longitude is not None:
        try:
            groundwater_stats = get_groundwater_stats(latitude, longitude)
            logger.info("Groundwater stats fetched for lat=%.4f lon=%.4f", latitude, longitude)
        except Exception as exc:
            logger.warning("Could not fetch groundwater stats: %s", exc)

    # 5. Groq AI explanation
    ai_result = get_ai_explanation(
        rule_recommendation=recommendation,
        health_label=health_label.value,
        health_score=round(green_ratio * 100, 2),
        greenness_pct=round(green_ratio * 100, 2),
        yellowing_pct=0.0,
        necrosis_pct=0.0,
        triggered_tags=["healthy"] if is_healthy else ["not-healthy", "colour-check"],
        rule_id="GREEN_CHECK",
        severity=1 if is_healthy else 3,
        lrtdc_vector="N/A",
        wind_force_ms=wind_force_ms,
        humidity_pct=humidity_pct,
        temperature_c=temperature_c,
        sunlight_lux=sunlight_lux,
        users_question=users_question,
        selected_language=selected_language,
        groundwater_stats=groundwater_stats,
    )

    # 6. Return response
    return LeafAnalysisResponse(
        image_width=w,
        image_height=h,
        total_pixels=total_pixels,
        green_pixel_count=green_count,
        green_ratio=round(green_ratio, 4),
        health_label=health_label,
        health_status=health_status,
        is_healthy=is_healthy,
        recommendation=recommendation,
        wind_force_ms=wind_force_ms,
        humidity_pct=humidity_pct,
        temperature_c=temperature_c,
        sunlight_lux=sunlight_lux,
        latitude=latitude,
        longitude=longitude,
        users_question=users_question,
        selected_language=selected_language,
        AIexplanation=ai_result["ai_explanation"],
        fallback_used=ai_result["fallback_used"],
    )