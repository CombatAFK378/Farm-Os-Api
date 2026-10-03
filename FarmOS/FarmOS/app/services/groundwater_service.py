

from __future__ import annotations

import io
import logging
import math
import threading
import urllib.request
from collections import OrderedDict
from typing import Optional

import numpy as np
from PIL import Image
from io import BytesIO
from scipy.ndimage import gaussian_filter

from app.services.groq_service import call_groq

logger = logging.getLogger(__name__)

MAP_ZOOM     = 13
MAP_RADIUS   = 2      # 5x5 tiles around the farm (~24 km across at zoom 13)
JPEG_QUALITY = 85     # source tiles are JPEG already; PNG was ~6x larger for no gain


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


def _fetch_mosaic(
    cx: int, cy: int, zoom: int = MAP_ZOOM, radius: int = MAP_RADIUS
) -> tuple[Image.Image, bool]:
    """
    Stitch the tiles around tile (cx, cy). Returns (image, complete) —
    complete is False if any tile failed and was replaced by a placeholder.
    """
    complete = True
    tiles = []
    for dy in range(-radius, radius + 1):
        row = []
        for dx in range(-radius, radius + 1):
            try:
                tile = _fetch_tile(cx + dx, cy + dy, zoom)
            except Exception:
                tile = Image.new("RGB", (256, 256), (150, 130, 110))
                complete = False
            row.append(tile)
        tiles.append(row)

    cols      = len(tiles[0])
    rows_count = len(tiles)
    stitched  = Image.new("RGB", (cols * 256, rows_count * 256))
    for r, row in enumerate(tiles):
        for c, tile in enumerate(row):
            stitched.paste(tile, (c * 256, r * 256))
    return stitched, complete


# ---------------------------------------------------------------------------
# Per-location cache
# ---------------------------------------------------------------------------
# Results depend only on which zoom-13 tile the coordinates fall in (the
# mosaic is built around it), so that tile is the cache key: every request
# from the same ~5 km tile reuses one entry. Results built from placeholder
# tiles (a download failed) are never cached, so a network blip can't stick.

class _LRUCache:
    """Small thread-safe LRU cache — requests run on worker threads."""

    def __init__(self, maxsize: int) -> None:
        self._data: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key, value) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            if len(self._data) > self._maxsize:
                self._data.popitem(last=False)


_potential_cache = _LRUCache(maxsize=2048)   # small dicts
_map_cache       = _LRUCache(maxsize=32)     # ~0.5 MB JPEG each, ~16 MB max


def _analyse_tile(tile: tuple[int, int]) -> tuple[dict, bytes]:
    """Download the mosaic once and build both the groundwater breakdown and the map JPEG."""
    img, complete = _fetch_mosaic(*tile)
    potential = _potential_breakdown(classify_terrain(np.array(img)))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    jpeg = buf.getvalue()

    if complete:
        _potential_cache.put(tile, potential)
        _map_cache.put(tile, jpeg)
    else:
        logger.warning("Some satellite tiles failed for tile %s; result not cached", tile)
    return potential, jpeg


def generate_map(lat: float, lon: float) -> bytes:
    """Return the stitched satellite mosaic around the farm as JPEG bytes. No overlays."""
    tile = _deg2tile(lat, lon, MAP_ZOOM)
    jpeg = _map_cache.get(tile)
    if jpeg is None:
        _, jpeg = _analyse_tile(tile)
    else:
        logger.info("Map cache hit for tile %s", tile)
    return jpeg


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


def _potential_breakdown(gw_score: np.ndarray) -> dict:
    total = gw_score.size
    return {
        "very_high_pct": round(float(np.sum(gw_score > 0.85)                         / total * 100), 2),
        "high_pct":      round(float(np.sum((gw_score > 0.70) & (gw_score <= 0.85)) / total * 100), 2),
        "medium_pct":    round(float(np.sum((gw_score > 0.45) & (gw_score <= 0.70)) / total * 100), 2),
        "low_pct":       round(float(np.sum((gw_score > 0.20) & (gw_score <= 0.45)) / total * 100), 2),
        "very_low_pct":  round(float(np.sum(gw_score <= 0.20)                       / total * 100), 2),
    }


def get_groundwater_stats(lat: float, lon: float) -> dict:
    tile      = _deg2tile(lat, lon, MAP_ZOOM)
    potential = _potential_cache.get(tile)
    if potential is None:
        potential, _ = _analyse_tile(tile)
    else:
        logger.info("Groundwater cache hit for tile %s", tile)
    return {
        "lat": lat,
        "lon": lon,
        "groundwater_potential": dict(potential),   # copy so callers can't mutate the cache
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