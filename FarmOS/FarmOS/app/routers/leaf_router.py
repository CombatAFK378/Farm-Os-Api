"""
FarmOS – /analyze/leaf router
==============================
Accepts a multipart form upload (leaf image + optional env scalars) and
returns the full LRTDC vector analysis.

Endpoints
---------
POST /analyze/leaf
    Upload a leaf image alongside optional environmental parameters.
    Returns LeafAnalysisResponse JSON.

GET  /analyze/leaf/schema
    Returns the JSON schema of LeafAnalysisResponse (useful for the mobile
    app / AI model to know what format to expect).
"""

from __future__ import annotations

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.concurrency import run_blocking
from app.models.schemas import ErrorResponse, LeafAnalysisResponse
from app.services.leaf_service import analyse_leaf

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["Leaf Analysis"])

# Accepted MIME types
ACCEPTED_MIME = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024   # 15 MB


# ---------------------------------------------------------------------------
# POST /analyze/leaf
# ---------------------------------------------------------------------------

@router.post(
    "/leaf",
    response_model=LeafAnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad image or invalid parameters"},
        413: {"model": ErrorResponse, "description": "File too large"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal processing error"},
    },
    summary="Analyse leaf image → health verdict, damage breakdown, AI explanation",
    description="""
Upload a **leaf photograph** (JPEG / PNG / WEBP, ≤ 15 MB). Works best with one
leaf filling most of the frame.

The service will:
1. Separate the leaf from the background (soil, table, other leaves, shadows).
2. Measure how much of the **leaf** is healthy, yellow (chlorosis) and brown
   (dead tissue), and the damage pattern (large patch, spots, edge burn).
3. Break the leaf into **five regions** — L (Left), R (Right), T (Top),
   D (Down), C (Center) — with their colour make-up and `(L,#RRGGBB)` vector.
4. Pick a rule-based recommendation using the optional weather inputs
   (`wind_force_ms`, `humidity_pct`, `temperature_c`, `sunlight_lux`).
5. Have the AI explain it to the farmer in the selected language.
""",
)
async def analyze_leaf_endpoint(
    leaf_image: Annotated[
        UploadFile,
        File(description="Leaf photograph — JPEG / PNG / WEBP, max 15 MB"),
    ],
    wind_force_ms: Annotated[
        Optional[float],
        Form(description="Wind speed in m/s", ge=0.0, le=200.0),
    ] = None,
    humidity_pct: Annotated[
        Optional[float],
        Form(description="Relative humidity 0–100 %", ge=0.0, le=100.0),
    ] = None,
    temperature_c: Annotated[
        Optional[float],
        Form(description="Air temperature in °C", ge=-50.0, le=70.0),
    ] = None,
    sunlight_lux: Annotated[
        Optional[float],
        Form(description="Sunlight illuminance in lux", ge=0.0, le=150_000.0),
    ] = None,
    users_question: Annotated[
        Optional[str],
        Form(description="Optional question from the farmer, e.g. 'Why is my leaf so dry?'"),
    ] = None,
    selected_language: Annotated[
        Optional[str],
        Form(description="Language for AI response. Options: English, Hindi, Marathi. Defaults to English."),
    ] = "English",
    latitude: Annotated[
        Optional[float],
        Form(description="Farm latitude (optional). If provided with longitude, groundwater data is included in AI analysis."),
    ] = None,
    longitude: Annotated[
        Optional[float],
        Form(description="Farm longitude (optional). If provided with latitude, groundwater data is included in AI analysis."),
    ] = None,
) -> LeafAnalysisResponse:

    # ── Validate MIME type ────────────────────────────────────────────────────
    if leaf_image.content_type not in ACCEPTED_MIME:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{leaf_image.content_type}'. "
                f"Accepted: {', '.join(ACCEPTED_MIME)}"
            ),
        )

    # ── Read bytes & size-guard ───────────────────────────────────────────────
    image_bytes = await leaf_image.read()

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {MAX_FILE_SIZE_BYTES // (1024*1024)} MB.",
        )

    logger.info(
        "Received leaf image — filename=%s size=%d bytes "
        "wind=%.2f humidity=%.2f temp=%.2f lux=%.2f question=%s",
        leaf_image.filename,
        len(image_bytes),
        wind_force_ms  or 0,
        humidity_pct   or 0,
        temperature_c  or 0,
        sunlight_lux   or 0,
        repr(users_question),
    )

    # ── Run analysis pipeline ─────────────────────────────────────────────────
    try:
        result = await run_blocking(
            analyse_leaf,
            image_bytes=image_bytes,
            wind_force_ms=wind_force_ms,
            humidity_pct=humidity_pct,
            temperature_c=temperature_c,
            sunlight_lux=sunlight_lux,
            users_question=users_question,
            selected_language=selected_language or "English",
            latitude=latitude,
            longitude=longitude,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during leaf analysis")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal processing error: {exc}",
        ) from exc

    return result


# ---------------------------------------------------------------------------
# GET /analyze/leaf/schema
# ---------------------------------------------------------------------------

@router.get(
    "/leaf/schema",
    summary="JSON schema of the LeafAnalysisResponse",
    description="Returns the full JSON Schema so clients can validate the response format.",
)
async def leaf_response_schema() -> JSONResponse:
    return JSONResponse(content=LeafAnalysisResponse.model_json_schema())