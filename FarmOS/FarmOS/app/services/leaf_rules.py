"""
FarmOS – Leaf recommendation rules
===================================
Turns the measured leaf signals (how much of the leaf is green / yellow /
brown, and the damage pattern) plus the weather into one farmer-facing
recommendation.

Rules are checked in priority order and the first match wins. Wording stays
cautious ("likely", "common causes") because a photo alone can't confirm a
disease — the AI explanation expands on this text, and it is also the
fallback shown when the AI is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LeafSignals:
    green_pct:         float            # % of the leaf area
    yellow_pct:        float
    brown_pct:         float
    damage_pct:        float            # yellow + brown
    largest_patch_pct: float            # biggest single brown patch, % of leaf
    spot_count:        int              # separate brown spots
    edge_share:        float            # share of the damage sitting on the leaf margin (0–1)
    leaf_found:        bool
    temperature_c:     Optional[float] = None
    humidity_pct:      Optional[float] = None
    wind_force_ms:     Optional[float] = None
    sunlight_lux:      Optional[float] = None


@dataclass
class Recommendation:
    rule_id: str
    text:    str
    tags:    list[str] = field(default_factory=list)


# Damage pattern thresholds (all % of leaf area unless noted)
LARGE_PATCH_PCT = 10.0     # one patch this big = blight / scorch-like lesion
MANY_SPOTS      = 6        # this many separate spots = leaf-spot pattern
EDGE_SHARE      = 0.6      # most damage on the margin = edge burn
SEVERE_PCT      = 50.0


def describe(s: LeafSignals, healthy: bool, damage_threshold: float) -> list[str]:
    """Human-readable findings, passed to the AI as 'Detected issues'."""
    tags: list[str] = []
    if not s.leaf_found:
        tags.append("no clear leaf found in the photo")
    if not healthy:
        if s.largest_patch_pct >= LARGE_PATCH_PCT:
            tags.append(f"large brown dead patch (~{s.largest_patch_pct:.0f}% of the leaf)")
        if s.spot_count >= MANY_SPOTS:
            tags.append(f"{s.spot_count} separate brown spots")
        if s.edge_share >= EDGE_SHARE and s.brown_pct >= 2:
            tags.append("browning concentrated along the leaf edges")
        elif s.largest_patch_pct < LARGE_PATCH_PCT and s.spot_count < MANY_SPOTS and s.brown_pct >= 2:
            tags.append(f"brown lesion (~{s.largest_patch_pct:.0f}% of the leaf)")
        if s.yellow_pct >= damage_threshold * 0.7:
            tags.append(f"yellowing on ~{s.yellow_pct:.0f}% of the leaf")
    if s.humidity_pct is not None and s.humidity_pct >= 80:
        tags.append("very humid weather")
    if _hot_dry(s):
        tags.append("hot and dry weather")
    if s.wind_force_ms is not None and s.wind_force_ms >= 12:
        tags.append("strong wind")
    return tags or ["no visible problems"]


def _hot_dry(s: LeafSignals) -> bool:
    return (s.humidity_pct is not None and s.humidity_pct <= 35) or \
           (s.temperature_c is not None and s.temperature_c >= 36)


def _warm_humid(s: LeafSignals) -> bool:
    humid = s.humidity_pct is not None and s.humidity_pct >= 80
    warm  = s.temperature_c is None or 18 <= s.temperature_c <= 32
    return humid and warm


def recommend(s: LeafSignals, healthy: bool, damage_threshold: float) -> Recommendation:
    tags = describe(s, healthy, damage_threshold)

    if not s.leaf_found:
        return Recommendation("NO_LEAF",
            "Couldn't find a clear leaf in this photo. Retake it in daylight with one leaf "
            "filling most of the frame, ideally against a plain background.", tags)

    if healthy and _warm_humid(s):
        return Recommendation("HEALTHY_FUNGAL_RISK",
            f"Leaf looks healthy (about {s.green_pct:.0f}% green, no significant yellowing or "
            "browning). The warm, very humid weather favours fungal disease, so consider a "
            "preventive neem oil spray and avoid wetting the leaves when watering.", tags)

    if healthy:
        return Recommendation("HEALTHY",
            f"Leaf looks healthy: about {s.green_pct:.0f}% of it is green with no significant "
            "yellowing or browning. Keep your current watering and fertiliser routine.", tags)

    if s.damage_pct >= SEVERE_PCT:
        return Recommendation("SEVERE",
            f"Severe damage: about {s.damage_pct:.0f}% of the leaf is yellow or brown. Remove "
            "badly affected leaves (don't compost them), check nearby plants for the same "
            "symptoms, and confirm the cause with your local Krishi Vigyan Kendra or "
            "agriculture officer before spraying.", tags)

    if s.largest_patch_pct >= LARGE_PATCH_PCT:
        if _hot_dry(s):
            return Recommendation("HEAT_SCORCH",
                "Large brown patch in hot or dry weather — likely sun/heat scorch or drought "
                "stress. Water deeply in the early morning, mulch the soil, and give shade "
                "during peak afternoon heat.", tags)
        return Recommendation("BLIGHT_LIKE",
            "Large brown dead patch on the leaf — typical of a fungal blight such as early or "
            "late blight. Remove affected leaves, avoid overhead watering, improve air flow "
            "between plants, and spray a copper-based or mancozeb fungicide as per the label.",
            tags)

    if s.spot_count >= MANY_SPOTS and s.brown_pct >= 2:
        return Recommendation("LEAF_SPOT",
            "Many small brown spots — typical of a fungal or bacterial leaf spot disease (or "
            "rust, if the spots are orange and powdery). Remove spotted leaves, keep the foliage "
            "dry, and use a copper-based fungicide; repeat after 7–10 days if new spots keep "
            "appearing.", tags)

    if s.yellow_pct >= max(1.5 * s.brown_pct, damage_threshold * 0.7):
        return Recommendation("CHLOROSIS",
            "Yellowing on the leaf (chlorosis). Common causes are nitrogen deficiency, "
            "over-watering or poor drainage. Make sure the soil isn't waterlogged; if drainage "
            "is fine, apply a nitrogen fertiliser such as urea in small doses.", tags)

    if s.edge_share >= EDGE_SHARE and s.brown_pct >= 2:
        if s.humidity_pct is not None and s.humidity_pct >= 75:
            return Recommendation("EDGE_BLIGHT",
                "Browning along the leaf edges in humid weather — this can be a fungal blight "
                "starting at the margins. Remove affected leaves, avoid wetting the foliage, and "
                "spray a copper-based or mancozeb fungicide; if the weather has been dry instead, "
                "check for water stress.", tags)
        return Recommendation("EDGE_SCORCH",
            "Browning concentrated along the leaf edges (leaf scorch). Common causes are water "
            "stress, potassium deficiency, or salt build-up from too much fertiliser. Check soil "
            "moisture, avoid over-fertilising, and apply potash (MOP) if a soil test shows low "
            "potassium.", tags)

    if s.brown_pct >= 2:
        return Recommendation("LESION",
            "Brown dead lesion(s) on the leaf — often the start of a fungal disease such as "
            "early blight or leaf spot. Remove the affected leaf, keep the foliage dry, and watch "
            "nearby leaves; spray a copper-based fungicide if more lesions appear.", tags)

    return Recommendation("MILD_DAMAGE",
        f"Some browning or yellowing found (about {s.damage_pct:.0f}% of the leaf). Watch this "
        "plant daily, remove affected leaves if it spreads, and keep watering consistent.", tags)
