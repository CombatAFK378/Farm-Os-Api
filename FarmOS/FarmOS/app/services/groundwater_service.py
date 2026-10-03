

from __future__ import annotations

import io
import logging
import math
import urllib.request
from typing import Optional

import numpy as np
from PIL import Image
from io import BytesIO
from scipy.ndimage import gaussian_filter

from app.services.groq_service import call_groq

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tile helpers
# ---------------------------------------------------------------------------

def _deg2tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def _fetch_tile(x: int, y: int, zoom: int) -> Image.Image:
    url = (
        f"https://server.arcgisonline.com/ArcGIS/rest/services/"
        f"World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "FarmOS-GroundwaterAPI/1.0"})
    with urllib.request.urlopen(req, timeout=12) as resp:
        return Image.open(BytesIO(resp.read())).convert("RGB")


def fetch_terrain_image(lat: float, lon: float, zoom: int = 13, radius: int = 2) -> Image.Image:
    cx, cy = _deg2tile(lat, lon, zoom)
    tiles = []
    for dy in range(-radius, radius + 1):
        row = []
        for dx in range(-radius, radius + 1):
            try:
                tile = _fetch_tile(cx + dx, cy + dy, zoom)
            except Exception:
                tile = Image.new("RGB", (256, 256), (150, 130, 110))
            row.append(tile)
        tiles.append(row)

    cols      = len(tiles[0])
    rows_count = len(tiles)
    stitched  = Image.new("RGB", (cols * 256, rows_count * 256))
    for r, row in enumerate(tiles):
        for c, tile in enumerate(row):
            stitched.paste(tile, (c * 256, r * 256))
    return stitched


def generate_map(lat: float, lon: float) -> bytes:
    """Fetch satellite tiles and return stitched PNG bytes. No overlays."""
    img   = fetch_terrain_image(lat, lon, zoom=13, radius=2)
    buf   = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def classify_terrain(img_arr: np.ndarray) -> np.ndarray:
    R   = img_arr[:, :, 0].astype(float) / 255.0
    G   = img_arr[:, :, 1].astype(float) / 255.0
    B   = img_arr[:, :, 2].astype(float) / 255.0
    eps = 1e-6

    NDVI       = (G - R) / (G + R + eps)
    NDWI       = (B - R) / (B + R + eps)
    brightness = (R + G + B) / 3.0
    cmax       = np.maximum(np.maximum(R, G), B)
    cmin       = np.minimum(np.minimum(R, G), B)
    saturation = np.where(cmax > 0, (cmax - cmin) / (cmax + eps), 0)

    gw = np.zeros_like(R)

    water_mask = (NDWI > 0.05) & (B > 0.3) & (B > R)
    gw[water_mask] = 0.85 + 0.15 * np.clip(NDWI[water_mask] * 2, 0, 1)

    veg_mask = (NDVI > 0.05) & ~water_mask
    gw[veg_mask] = 0.55 + 0.25 * np.clip(NDVI[veg_mask] * 2, 0, 1)

    moist_mask = (brightness < 0.35) & (saturation < 0.25) & ~water_mask & ~veg_mask
    gw[moist_mask] = 0.40 + 0.20 * (1 - brightness[moist_mask] / 0.35)

    dry_mask = (R > G) & (R > B) & ~water_mask & ~veg_mask & ~moist_mask
    gw[dry_mask] = 0.20 + 0.20 * (1 - np.clip((R[dry_mask] - G[dry_mask]) * 3, 0, 1))

    urban_mask = ~water_mask & ~veg_mask & ~moist_mask & ~dry_mask
    gw[urban_mask] = 0.10 + 0.15 * saturation[urban_mask]

    return gaussian_filter(gw, sigma=6)


def get_groundwater_stats(lat: float, lon: float) -> dict:
    img_arr  = np.array(fetch_terrain_image(lat, lon, zoom=13, radius=2))
    gw_score = classify_terrain(img_arr)
    total    = gw_score.size
    return {
        "lat": lat,
        "lon": lon,
        "groundwater_potential": {
            "very_high_pct": round(float(np.sum(gw_score > 0.85)                         / total * 100), 2),
            "high_pct":      round(float(np.sum((gw_score > 0.70) & (gw_score <= 0.85)) / total * 100), 2),
            "medium_pct":    round(float(np.sum((gw_score > 0.45) & (gw_score <= 0.70)) / total * 100), 2),
            "low_pct":       round(float(np.sum((gw_score > 0.20) & (gw_score <= 0.45)) / total * 100), 2),
            "very_low_pct":  round(float(np.sum(gw_score <= 0.20)                       / total * 100), 2),
        },
    }


def _build_groundwater_prompt(stats: dict, selected_language: str = "English") -> str:
    gw  = stats["groundwater_potential"]
    lat = stats["lat"]
    lon = stats["lon"]

    dominant_zone = max(
        {"Very High": gw["very_high_pct"], "High": gw["high_pct"],
         "Medium": gw["medium_pct"], "Low": gw["low_pct"], "Very Low": gw["very_low_pct"]},
        key=lambda k: {"Very High": gw["very_high_pct"], "High": gw["high_pct"],
                       "Medium": gw["medium_pct"], "Low": gw["low_pct"],
                       "Very Low": gw["very_low_pct"]}[k]
    )

    return f"""You are an expert agricultural advisor for farmers in India.

A satellite-based groundwater potential analysis has been completed for the farm location below.
Your job is to explain what this means for the farmer's crops and vegetation in simple, practical language.

You MUST answer ONLY in {selected_language} language.

--- LOCATION ---
Latitude:  {lat}
Longitude: {lon}

--- GROUNDWATER POTENTIAL BREAKDOWN ---
Very High potential zone : {gw['very_high_pct']:.1f}% of the area
High potential zone      : {gw['high_pct']:.1f}% of the area
Medium potential zone    : {gw['medium_pct']:.1f}% of the area
Low potential zone       : {gw['low_pct']:.1f}% of the area
Very Low potential zone  : {gw['very_low_pct']:.1f}% of the area
Dominant zone            : {dominant_zone}

--- YOUR TASK ---
Write 3-4 sentences for the farmer explaining:
1. What the groundwater situation looks like at their location
2. How this is likely affecting their crops and vegetation right now
3. One practical recommendation — e.g. whether they can rely on borewells,
   need rainwater harvesting, should use drip irrigation, etc.

Write in plain paragraph form. No bullet points. Warm and helpful tone.
Keep it under 120 words. Respond only in {selected_language}.
""".strip()


def get_groundwater_explanation(
    stats:             dict,
    selected_language: str = "English",
) -> dict:
    """Send groundwater stats to Groq. Never raises — falls back if the Groq call fails."""
    prompt = _build_groundwater_prompt(stats, selected_language)

    try:
        explanation = call_groq(prompt)
        logger.info("Groq groundwater explanation received (%d chars)", len(explanation))
        return {"AIexplanation": explanation, "ai_available": True, "fallback_used": False}

    except Exception as exc:
        logger.warning("Groq call failed for groundwater: %s. Using fallback.", exc)
        gw = stats["groundwater_potential"]
        fallback = (
            f"Groundwater analysis for ({stats['lat']}, {stats['lon']}): "
            f"{gw['very_high_pct'] + gw['high_pct']:.1f}% high or very high potential, "
            f"{gw['medium_pct']:.1f}% medium, "
            f"{gw['low_pct'] + gw['very_low_pct']:.1f}% low or very low. "
            "Plan irrigation accordingly."
        )
        return {"AIexplanation": fallback, "ai_available": False, "fallback_used": True}