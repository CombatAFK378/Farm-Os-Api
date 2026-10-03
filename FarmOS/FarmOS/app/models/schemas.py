"""
FarmOS – Schemas
=================
Leaf analysis response: the leaf is segmented from the background and its
pixels classified as healthy / yellow / brown, overall and per LRTDC region,
plus a rule-based recommendation and the AI explanation.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class HealthLabel(str, Enum):
    HEALTHY     = "healthy"
    NOT_HEALTHY = "not_healthy"


class RegionStats(BaseModel):
    """Colour make-up of one leaf region (L, R, T, D or C)."""
    hex_color:      str   = Field(..., description="Average colour of the region, #RRGGBB")
    green_ratio:    float = Field(..., description="Healthy share of the region (0-1)")
    yellow_ratio:   float = Field(..., description="Yellow (chlorotic) share of the region (0-1)")
    brown_ratio:    float = Field(..., description="Brown / dead share of the region (0-1)")
    coverage_ratio: float = Field(..., description="Share of the whole leaf in this region (0-1)")


class HealthSummary(BaseModel):
    """Grouped health values (kept for app versions that read health_summary)."""
    health_score:   float
    health_label:   str
    greenness_pct:  float
    yellowing_pct:  float
    necrosis_pct:   float
    recommendation: str


class LeafAnalysisResponse(BaseModel):
    """Response from POST /analyze/leaf"""

    # Image metadata
    image_width:       int   = Field(..., description="Image width in pixels")
    image_height:      int   = Field(..., description="Image height in pixels")
    total_pixels:      int   = Field(..., description="Total pixels in image")

    # Colour analysis — all ratios are of the detected LEAF, not the whole photo
    leaf_pixel_count:  int   = Field(0,   description="Pixels belonging to the detected leaf")
    green_pixel_count: int   = Field(..., description="Healthy (not yellow/brown) leaf pixels")
    green_ratio:       float = Field(..., description="Healthy share of the leaf (0-1)")
    yellow_ratio:      float = Field(0.0, description="Yellow (chlorotic) share of the leaf (0-1)")
    brown_ratio:       float = Field(0.0, description="Brown / dead share of the leaf (0-1)")

    # Verdict
    is_healthy:        bool        = Field(..., description="True if yellow + brown < 7% of the leaf")
    health_label:      HealthLabel = Field(..., description="healthy or not_healthy")
    health_status:     str         = Field(..., description="HEALTHY or NOT HEALTHY")
    health_score:      float       = Field(0.0, description="0-100: >=80 healthy, 50-80 needs attention, <50 critical")
    severity:          int         = Field(1,   description="1 (none) to 5 (severe)")
    detected_issues:   list[str]   = Field(default_factory=list, description="Findings, e.g. 'large brown dead patch'")
    recommendation:    str         = Field(..., description="Rule-based recommendation")

    # Per-region breakdown (L, R, T, D, C) and compact colour vector
    regions:           dict[str, RegionStats] = Field(default_factory=dict)
    lrtdc_vector:      str                    = Field("", description="e.g. (L,#5A7A3B)(R,#...)...")
    health_summary:    Optional[HealthSummary] = None

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