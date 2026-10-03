"""
FarmOS – Schemas (Simplified)
==============================
Simple healthy / not healthy leaf analysis response.
No LRTDC. No regions. Just colour check + AI explanation.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class HealthLabel(str, Enum):
    HEALTHY     = "healthy"
    NOT_HEALTHY = "not_healthy"


class LeafAnalysisResponse(BaseModel):
    """Response from POST /analyze/leaf"""

    # Image metadata
    image_width:       int   = Field(..., description="Image width in pixels")
    image_height:      int   = Field(..., description="Image height in pixels")
    total_pixels:      int   = Field(..., description="Total pixels in image")

    # Colour analysis
    green_pixel_count: int   = Field(..., description="Number of green pixels detected")
    green_ratio:       float = Field(..., description="Fraction of image that is green (0-1)")

    # Verdict — the one rule
    is_healthy:        bool        = Field(..., description="True if green_ratio >= 40%")
    health_label:      HealthLabel = Field(..., description="healthy or not_healthy")
    health_status:     str         = Field(..., description="HEALTHY or NOT HEALTHY")
    recommendation:    str         = Field(..., description="Plain rule-based recommendation")

    # Environmental inputs echoed back
    wind_force_ms:     Optional[float] = None
    humidity_pct:      Optional[float] = None
    temperature_c:     Optional[float] = None
    sunlight_lux:      Optional[float] = None

    # Location echoed back
    latitude:          Optional[float] = None
    longitude:         Optional[float] = None

    # Farmer input echoed back
    users_question:    Optional[str] = None
    selected_language: Optional[str] = "English"

    # Groq AI explanation
    AIexplanation:     Optional[str] = Field(
        None,
        description="Farmer-friendly explanation from gpt-oss-120b on Groq. Falls back to recommendation if the Groq call fails."
    )
    fallback_used:     bool = Field(
        False,
        description="True if the Groq call failed and AIexplanation is just the rule-based recommendation"
    )


class ErrorResponse(BaseModel):
    detail: str
    code:   str