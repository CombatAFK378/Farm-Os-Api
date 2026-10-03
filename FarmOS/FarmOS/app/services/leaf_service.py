"""
FarmOS – Leaf analysis
=======================
1. Find the leaf: the connected green/yellow tissue at the centre of the
   photo, plus the holes and edge notches inside its outline (lesions) —
   unless their colours match the background (soil / wood between leaflets).
2. Classify every leaf pixel as healthy (green), yellow (chlorosis) or brown
   (dead / necrotic tissue, including dark spots).
3. Measure the damage pattern (one big patch, many spots, edge burn) and the
   colour of five leaf regions — L, R, T, D (left/right/top/down) and C.
4. leaf_rules turns that + the weather into a recommendation; the AI
   explains it to the farmer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.services.groq_service import get_ai_explanation
from app.services.groundwater_service import get_groundwater_stats
from app.services.leaf_rules import LeafSignals, recommend
from app.models.schemas import HealthLabel, HealthSummary, LeafAnalysisResponse, RegionStats

logger = logging.getLogger(__name__)

WORK_SIZE        = 320     # analyse at this longest side (px): enough for colour stats, cheap on CPU
DAMAGE_THRESHOLD = 7.0     # % of the leaf yellow/brown at which it stops counting as healthy
LESION_BROWN_PCT = 2.5     # ...or this much brown, with
LESION_PATCH_PCT = 1.0     # one distinct lesion at least this big (% of leaf)
LESION_SPOTS     = 10      # or at least this many separate spots
MIN_LEAF_SHARE   = 0.03    # the leaf must cover >= 3% of the photo to count as found
REGIONS          = ("L", "R", "T", "D", "C")


@dataclass
class LeafMeasurement:
    leaf_found:        bool
    leaf_share:        float               # leaf area / photo area
    healthy_pct:       float               # % of leaf area
    yellow_pct:        float
    brown_pct:         float
    largest_patch_pct: float
    spot_count:        int
    edge_share:        float
    regions:           dict[str, RegionStats]
    lrtdc_vector:      str

    @property
    def damage_pct(self) -> float:
        return self.yellow_pct + self.brown_pct


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def _hsv_channels(hsv: np.ndarray):
    return (hsv[..., i].astype(np.int16) for i in range(3))


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8) * 255, 1)
    flood  = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, flood, (0, 0), 128)            # everything reachable from outside
    return mask | (padded[1:-1, 1:-1] == 0)


def _component_at_centre(mask: np.ndarray, centre: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    overlap = np.bincount(labels[centre], minlength=n)
    overlap[0] = 0
    k = int(overlap.argmax()) if overlap.max() > 0 else int(stats[1:, cv2.CC_STAT_AREA].argmax()) + 1
    return labels == k


def _background_like(bgr: np.ndarray, background: np.ndarray):
    """
    (same_colour, shadow_of) masks for pixels resembling the `background` sample:
    same_colour = colour common in the background; shadow_of = same hue/chroma
    (Lab a,b) but darker than the typical background, i.e. a shadow cast on it.
    None if there's too little background to sample (the leaf fills the frame).
    """
    n = int(background.sum())
    if n < 50:
        return None
    lab  = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int32)
    full = (lab[..., 0] // 32) * 64 + (lab[..., 1] // 32) * 8 + lab[..., 2] // 32   # 8 levels each
    ab   = (lab[..., 1] // 16) * 16 + lab[..., 2] // 16                              # 16 levels each
    common_full = np.bincount(full[background], minlength=512) / n >= 0.01
    common_ab   = np.bincount(ab[background],   minlength=256) / n >= 0.01
    darker      = lab[..., 0] < np.median(lab[..., 0][background])
    return common_full[full], common_ab[ab] & darker


def _segment_leaf(bgr: np.ndarray, hsv: np.ndarray) -> np.ndarray:
    h, w = hsv.shape[:2]
    H, S, V = _hsv_channels(hsv)
    # Core: connected green (or strongly yellow) tissue at the centre — brown
    # never seeds it, so soil / wood can't start the leaf
    leafy = ((H >= 30) & (H <= 95) & (S > 35) & (V > 35)) | \
            ((H >= 22) & (H < 30) & (S > 80) & (V > 80))
    leafy = cv2.morphologyEx(leafy.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0

    yy, xx = np.ogrid[:h, :w]
    centre = ((xx - w / 2) / (0.3 * w)) ** 2 + ((yy - h / 2) / (0.3 * h)) ** 2 <= 1
    b = max(3, round(0.04 * min(h, w)))
    border = np.zeros((h, w), bool)
    border[:b] = border[-b:] = True
    border[:, :b] = border[:, -b:] = True

    core = _component_at_centre(leafy, centre)
    if not core.any():
        return core

    # Grow the core into connected pixels whose colour is rare around the photo's
    # edges: dead / yellow tissue and lesions join the leaf, while background
    # (soil, wood, table, other leaves, shadows) stays out because it *is* the
    # edge colour. The edge sample skips pixels right next to the core (lesions
    # touching the photo edge), and growth is capped at a distance from the core
    # so a busy scene can't leak everywhere.
    gap = max(3, round(0.04 * max(h, w)))
    near_core = cv2.dilate(core.astype(np.uint8), np.ones((2 * gap + 1, 2 * gap + 1), np.uint8)) > 0
    sample    = border & ~near_core
    bg = _background_like(bgr, sample) if sample.sum() >= 0.02 * h * w else None
    if bg is not None:
        same_colour, shadow = bg
        # Holes fully surrounded by the core are almost always lesions: only drop
        # them if they're plainly the background colour (a gap showing soil/wood).
        enclosed = _fill_holes(core) & ~core & ~same_colour
        reach = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5) <= 0.25 * max(h, w)
        grow  = core | enclosed | (~same_colour & ~shadow & reach)
        grow  = cv2.morphologyEx(grow.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        n, labels = cv2.connectedComponents((grow | core).astype(np.uint8), connectivity=8)
        leaf = np.isin(labels, np.unique(labels[core]))
    else:
        # Leaf fills the frame: nothing to compare against, so grow a short way
        # into adjacent lesion-coloured (yellow / brown) pixels, then fill holes
        yellow, brown = _classify(hsv, np.ones((h, w), bool))
        reach = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5) <= 0.06 * max(h, w)
        n, labels = cv2.connectedComponents((core | ((yellow | brown) & reach)).astype(np.uint8), connectivity=8)
        leaf = _fill_holes(np.isin(labels, np.unique(labels[core])))

    # Tiny enclosed holes are spots even if they share a background colour (black spot on black)
    holes = _fill_holes(leaf) & ~leaf
    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=8)
    small = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] <= 0.003 * leaf.sum())
    return leaf | (np.isin(labels, small[small > 0]))


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def _classify(hsv: np.ndarray, leaf: np.ndarray):
    """Split leaf pixels into yellow (chlorosis) and brown (dead tissue); the rest is healthy."""
    H, S, V = _hsv_channels(hsv)
    green  = (H >= 33) & (H <= 95) & (S >= 25) & (V >= 30)
    yellow = ~green & (H >= 21) & (H < 33) & (S >= 50) & (V >= 80)   # incl. lime / shaded yellow
    brown  = ~green & ~yellow & (
        (((H < 21) | (H >= 155)) & (S >= 40) & (V >= 30))   # orange / red / tan-brown dead tissue
        | ((H >= 21) & (H < 33) & (S >= 40))                 # dark olive dead tissue
        | (V < 30)                                           # black spots
        | ((S < 25) & (V < 170))                             # grey / tan dead tissue (brighter = glare)
    )
    return leaf & yellow, leaf & brown


def _regions(bgr: np.ndarray, leaf: np.ndarray, yellow: np.ndarray, brown: np.ndarray) -> dict[str, RegionStats]:
    ys, xs = np.nonzero(leaf)
    cy, cx = ys.mean(), xs.mean()
    half_h = max((ys.max() - ys.min()) / 2, 1)
    half_w = max((xs.max() - xs.min()) / 2, 1)
    dy, dx = (ys - cy) / half_h, (xs - cx) / half_w
    names = np.where(dx ** 2 + dy ** 2 < 0.45 ** 2, "C",
             np.where(np.abs(dy) >= np.abs(dx), np.where(dy < 0, "T", "D"),
                                                np.where(dx < 0, "L", "R")))
    pix = bgr[ys, xs]
    yel, brn = yellow[ys, xs], brown[ys, xs]
    out = {}
    for name in REGIONS:
        sel = names == name
        n = int(sel.sum())
        if n == 0:
            out[name] = RegionStats(hex_color="#888888", green_ratio=0, yellow_ratio=0,
                                    brown_ratio=0, coverage_ratio=0)
            continue
        b, g, r = pix[sel].mean(axis=0)
        y_r, b_r = float(yel[sel].mean()), float(brn[sel].mean())
        out[name] = RegionStats(
            hex_color=f"#{int(r):02X}{int(g):02X}{int(b):02X}",
            green_ratio=round(1 - y_r - b_r, 4),
            yellow_ratio=round(y_r, 4),
            brown_ratio=round(b_r, 4),
            coverage_ratio=round(n / len(ys), 4),
        )
    return out


def _prepare(img_bgr: np.ndarray):
    """Downscaled photo, its HSV, and the leaf mask on that grid."""
    h, w = img_bgr.shape[:2]
    scale = WORK_SIZE / max(h, w)
    small = cv2.resize(img_bgr, (max(1, round(w * scale)), max(1, round(h * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1 else img_bgr
    small = cv2.GaussianBlur(small, (3, 3), 0)           # tame JPEG noise / leaf hairs
    hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    return small, hsv, _segment_leaf(small, hsv)


def measure_leaf(img_bgr: np.ndarray) -> LeafMeasurement:
    small, hsv, leaf = _prepare(img_bgr)
    leaf_share = leaf.sum() / leaf.size
    leaf_found = leaf_share >= MIN_LEAF_SHARE
    if not leaf_found:                                   # still measure something sensible
        leaf = np.ones(leaf.shape, bool)
    # Measure a pixel inside the outline: edge pixels blend leaf and background
    inner = cv2.erode(leaf.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if inner.sum() >= 0.5 * leaf.sum():
        leaf = inner
    leaf_px = int(leaf.sum())

    yellow, brown = _classify(hsv, leaf)
    damage = yellow | brown
    pct = lambda m: 100.0 * int(m.sum()) / leaf_px

    # Brown patches / spots
    spots = cv2.morphologyEx(brown.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(spots, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([0])
    min_spot = max(4, 0.0005 * leaf_px)
    spot_count = int((areas >= min_spot).sum())
    largest_patch_pct = 100.0 * float(areas.max()) / leaf_px

    # How much of the brown sits on the leaf margin (edge burn)
    dist = cv2.distanceTransform(leaf.astype(np.uint8), cv2.DIST_L2, 5)
    band = leaf & (dist <= 0.07 * np.sqrt(leaf_px))
    edge_share = float((brown & band).sum()) / max(int(brown.sum()), 1)

    regions = _regions(small, leaf, yellow, brown)
    return LeafMeasurement(
        leaf_found=leaf_found,
        leaf_share=float(leaf_share),
        healthy_pct=100.0 - pct(damage),
        yellow_pct=pct(yellow),
        brown_pct=pct(brown),
        largest_patch_pct=largest_patch_pct,
        spot_count=spot_count,
        edge_share=edge_share,
        regions=regions,
        lrtdc_vector="".join(f"({k},{v.hex_color})" for k, v in regions.items()),
    )


def is_healthy(m: LeafMeasurement) -> bool:
    """Healthy unless enough of the leaf is yellow/brown, or it has a distinct brown lesion."""
    lesion = m.brown_pct >= LESION_BROWN_PCT and (m.largest_patch_pct >= LESION_PATCH_PCT
                                                  or m.spot_count >= LESION_SPOTS)
    return m.damage_pct < DAMAGE_THRESHOLD and not lesion


def health_score(damage_pct: float) -> float:
    """0–100, matching the app's grading: >= 80 healthy, 50–80 needs attention, < 50 critical."""
    if damage_pct <= DAMAGE_THRESHOLD:
        score = 100 - damage_pct * (20 / DAMAGE_THRESHOLD)
    elif damage_pct <= 30:
        score = 80 - (damage_pct - DAMAGE_THRESHOLD) * (30 / (30 - DAMAGE_THRESHOLD))
    else:
        score = 50 - (damage_pct - 30) * (50 / 70)
    return round(max(0.0, min(100.0, score)), 1)


def severity(damage_pct: float) -> int:
    return 1 if damage_pct < DAMAGE_THRESHOLD else 2 if damage_pct < 15 else \
           3 if damage_pct < 30 else 4 if damage_pct < 50 else 5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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

    # 2. Find the leaf and measure it
    m = measure_leaf(img_bgr)
    damage        = m.damage_pct
    healthy       = is_healthy(m)
    health_label  = HealthLabel.HEALTHY if healthy else HealthLabel.NOT_HEALTHY
    health_status = "HEALTHY" if healthy else "NOT HEALTHY"
    score         = health_score(damage) if healthy else min(health_score(damage), 79.0)
    sev           = severity(damage) if healthy else max(severity(damage), 2)

    # 3. Recommendation rules
    rec = recommend(LeafSignals(
        green_pct=m.healthy_pct, yellow_pct=m.yellow_pct, brown_pct=m.brown_pct,
        damage_pct=damage, largest_patch_pct=m.largest_patch_pct, spot_count=m.spot_count,
        edge_share=m.edge_share, leaf_found=m.leaf_found,
        temperature_c=temperature_c, humidity_pct=humidity_pct,
        wind_force_ms=wind_force_ms, sunlight_lux=sunlight_lux,
    ), healthy, DAMAGE_THRESHOLD)

    logger.info("Leaf: found=%s share=%.0f%% healthy=%.1f%% yellow=%.1f%% brown=%.1f%% "
                "patch=%.1f%% spots=%d edge=%.2f -> %s (%s)",
                m.leaf_found, m.leaf_share * 100, m.healthy_pct, m.yellow_pct, m.brown_pct,
                m.largest_patch_pct, m.spot_count, m.edge_share, health_status, rec.rule_id)

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
        rule_recommendation=rec.text,
        health_label=health_label.value,
        health_score=score,
        greenness_pct=round(m.healthy_pct, 1),
        yellowing_pct=round(m.yellow_pct, 1),
        necrosis_pct=round(m.brown_pct, 1),
        triggered_tags=rec.tags,
        rule_id=rec.rule_id,
        severity=sev,
        lrtdc_vector=m.lrtdc_vector,
        wind_force_ms=wind_force_ms,
        humidity_pct=humidity_pct,
        temperature_c=temperature_c,
        sunlight_lux=sunlight_lux,
        users_question=users_question,
        selected_language=selected_language,
        groundwater_stats=groundwater_stats,
    )

    # 6. Return response (pixel counts scaled back to the original photo)
    leaf_full = int(m.leaf_share * total_pixels) if m.leaf_found else total_pixels
    return LeafAnalysisResponse(
        image_width=w,
        image_height=h,
        total_pixels=total_pixels,
        leaf_pixel_count=leaf_full,
        green_pixel_count=int(leaf_full * m.healthy_pct / 100),
        green_ratio=round(m.healthy_pct / 100, 4),
        yellow_ratio=round(m.yellow_pct / 100, 4),
        brown_ratio=round(m.brown_pct / 100, 4),
        health_label=health_label,
        health_status=health_status,
        is_healthy=healthy,
        health_score=score,
        severity=sev,
        detected_issues=rec.tags,
        recommendation=rec.text,
        regions=m.regions,
        lrtdc_vector=m.lrtdc_vector,
        health_summary=HealthSummary(
            health_score=score,
            health_label=health_status.lower(),
            greenness_pct=round(m.healthy_pct, 1),
            yellowing_pct=round(m.yellow_pct, 1),
            necrosis_pct=round(m.brown_pct, 1),
            recommendation=rec.text,
        ),
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
