
from __future__ import annotations

import io
import logging
from typing import Annotated, Optional

from fastapi import APIRouter, HTTPException, Query, Form, status
from fastapi.responses import StreamingResponse, JSONResponse

from app.concurrency import run_blocking
from app.services.groundwater_service import (
    generate_map,
    get_groundwater_stats,
    get_groundwater_explanation,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/groundwater", tags=["Groundwater Analysis"])


# ---------------------------------------------------------------------------
# POST /groundwater/analyze  — JSON response with stats + AI explanation
# ---------------------------------------------------------------------------

@router.post(
    "/analyze",
    summary="Groundwater stats + AI farming impact explanation",
    description="""
Provide a **latitude** and **longitude** to receive:
- Full groundwater potential breakdown (very_high → very_low %)
- AI-generated explanation (via gpt-oss-120b on Groq) of how the groundwater
  situation affects crops and vegetation at that location
- Response in your chosen language (English / Hindi / Marathi)
""",
)
async def analyze_groundwater(
    latitude: Annotated[
        float,
        Form(description="Latitude of the farm location (e.g. 18.52 for Pune)", ge=-90, le=90),
    ],
    longitude: Annotated[
        float,
        Form(description="Longitude of the farm location (e.g. 73.85 for Pune)", ge=-180, le=180),
    ],
    selected_language: Annotated[
        Optional[str],
        Form(description="Language for AI response: English, Hindi, Marathi. Default: English"),
    ] = "English",
) -> JSONResponse:

    logger.info("Groundwater analyze request — lat=%.4f lon=%.4f lang=%s",
                latitude, longitude, selected_language)

    try:
        # 1. Get groundwater stats
        stats = await run_blocking(get_groundwater_stats, latitude, longitude)

        # 2. Get Groq AI explanation
        ai = await run_blocking(
            get_groundwater_explanation,
            stats=stats,
            selected_language=selected_language or "English",
        )

        # 3. Assemble JSON response — stats + AIexplanation only
        #    To get the PNG map image call: GET /groundwater/map?latitude=..&longitude=..
        response = {
            "lat":                   stats["lat"],
            "lon":                   stats["lon"],
            "groundwater_potential": stats["groundwater_potential"],
            "AIexplanation":         ai["AIexplanation"],
            "selected_language":     selected_language or "English",
            "ai_available":          ai["ai_available"],
            "fallback_used":         ai["fallback_used"],
            "map_image_url":         f"/groundwater/map?latitude={latitude}&longitude={longitude}",
        }
        return JSONResponse(content=response)

    except Exception as exc:
        logger.exception("Groundwater analysis failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Groundwater analysis failed: {exc}",
        )


# ---------------------------------------------------------------------------
# GET /groundwater/map  — returns PNG image
# ---------------------------------------------------------------------------

@router.get(
    "/map",
    summary="Groundwater potential map image (PNG)",
    response_description="PNG satellite + groundwater overlay map",
    responses={
        200: {"content": {"image/png": {}}, "description": "Groundwater map PNG"},
        500: {"description": "Map generation failed"},
    },
)
async def get_groundwater_map(
    latitude:  float = Query(..., ge=-90,  le=90,  description="Latitude  (e.g. 18.52)"),
    longitude: float = Query(..., ge=-180, le=180, description="Longitude (e.g. 73.85)"),
):
    """
    Returns a professional groundwater potential map PNG for the given coordinates.

    Example:
    ```
    GET /groundwater/map?latitude=18.52&longitude=73.85   ← Pune
    GET /groundwater/map?latitude=13.20&longitude=80.25   ← Chennai
    ```
    """
    logger.info("Groundwater map request — lat=%.4f lon=%.4f", latitude, longitude)
    try:
        image_bytes = await run_blocking(generate_map, latitude, longitude)
    except Exception as exc:
        logger.exception("Map generation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Map generation failed: {exc}",
        )

    return StreamingResponse(
        io.BytesIO(image_bytes),
        media_type="image/png",
        headers={
            "Content-Disposition": f'inline; filename="gw_map_{latitude}_{longitude}.png"',
            "X-Latitude":  str(latitude),
            "X-Longitude": str(longitude),
        },
    )