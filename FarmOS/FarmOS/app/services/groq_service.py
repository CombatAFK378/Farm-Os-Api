"""
FarmOS – Groq AI Explanation Service
=====================================
Takes the structured leaf diagnosis + environmental parameters and sends them
to gpt-oss-120b hosted on Groq (https://groq.com).

Groq returns a descriptive, farmer-friendly paragraph that expands on the
rule-based recommendation with context, reasoning, and actionable next steps.

Flow:
  leaf_service / groundwater_service  →  groq_service  →  final response
          (rule-based verdict)             (LLM expand)

Usage:
  Put your Groq API key (https://console.groq.com/keys) in a .env file next
  to main.py, or export it as an environment variable:
    GROQ_API_KEY=gsk_...

  If the key is missing or Groq fails, the service gracefully falls back to
  the rule-based recommendation so the API never breaks.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from groq import Groq

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GROQ_MODEL            = "openai/gpt-oss-120b"
GROQ_TIMEOUT_S        = 30
GROQ_REASONING_EFFORT = "low"     # low = fastest; these are short explanations
# gpt-oss is a reasoning model: its hidden reasoning tokens count toward this
# limit, so leave headroom beyond the ~150-word answer (Hindi / Marathi text
# also needs more tokens per word than English).
GROQ_MAX_TOKENS       = 1024


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(
    # From recommendation engine
    rule_recommendation: str,
    health_label:        str,
    health_score:        float,
    greenness_pct:       float,
    yellowing_pct:       float,
    necrosis_pct:        float,
    triggered_tags:      list[str],
    rule_id:             str,
    severity:            int,
    lrtdc_vector:        str,
    # Environmental inputs
    wind_force_ms:       Optional[float],
    humidity_pct:        Optional[float],
    temperature_c:       Optional[float],
    sunlight_lux:        Optional[float],
    # Farmer's question (optional)
    users_question:      Optional[str] = None,
    # Language for the response
    selected_language:   str = "English",
    # Groundwater context (optional — provided when lat/lon given)
    groundwater_stats:   Optional[dict] = None,
) -> str:
    """
    Build the prompt sent to gpt-oss-120b.
    Keep it structured so the model can reason clearly and stay on-topic.
    """

    # Format env block — skip None values cleanly
    env_lines = []
    if temperature_c  is not None: env_lines.append(f"  - Temperature:      {temperature_c}°C")
    if humidity_pct   is not None: env_lines.append(f"  - Humidity:         {humidity_pct}%")
    if wind_force_ms  is not None: env_lines.append(f"  - Wind speed:       {wind_force_ms} m/s")
    if sunlight_lux   is not None: env_lines.append(f"  - Sunlight:         {sunlight_lux} lux")
    env_block = "\n".join(env_lines) if env_lines else "  - No environmental data provided"

    tags_str = ", ".join(triggered_tags) if triggered_tags else "none"

    # Build groundwater block if stats were provided
    if groundwater_stats:
        gw = groundwater_stats.get("groundwater_potential", {})
        lat = groundwater_stats.get("lat", "N/A")
        lon = groundwater_stats.get("lon", "N/A")
        gw_block = (
            f"  - Farm location:     Lat {lat}, Lon {lon}\n"
            f"  - Very High GW zone: {gw.get('very_high_pct', 0):.1f}% of area\n"
            f"  - High GW zone:      {gw.get('high_pct', 0):.1f}% of area\n"
            f"  - Medium GW zone:    {gw.get('medium_pct', 0):.1f}% of area\n"
            f"  - Low GW zone:       {gw.get('low_pct', 0):.1f}% of area\n"
            f"  - Very Low GW zone:  {gw.get('very_low_pct', 0):.1f}% of area"
        )
    else:
        gw_block = "  - No groundwater data provided"

    # Build task block FIRST before the prompt string
    if users_question:
        task_block = (
            f"The farmer has asked: \"{users_question}\"\n\n"
            "Answer their specific question directly using the leaf diagnosis and environmental "
            "data above as context. Then in 1–2 sentences give the most important action they "
            "should take today."
        )
    else:
        task_block = (
            "Write a clear, descriptive explanation (3–5 sentences) for the farmer that:\n"
            "1. Tells them what is happening to their plant in simple language\n"
            "2. Explains WHY this is happening based on the leaf data and weather conditions\n"
            "3. Gives the most important action they should take TODAY\n"
            "4. Mentions what to watch for over the next few days"
        )

    prompt = f"""You are an expert agricultural advisor for small and medium-scale farmers in India.

A computer vision system has analysed a leaf photograph and produced the following structured diagnosis. Your job is to explain this diagnosis to the farmer in clear, practical, and empathetic language.

You MUST answer ONLY in {selected_language} language. Do not use any other language in your response.

--- LEAF DIAGNOSIS ---
Health status:     {health_label.upper()} (score: {health_score:.1f} / 100)
Severity level:    {severity} out of 5
Greenness:         {greenness_pct:.1f}% of leaf is green
Yellowing:         {yellowing_pct:.1f}% of leaf is yellowing
Browning/necrosis: {necrosis_pct:.1f}% of leaf is brown or dead
LRTDC colour vector: {lrtdc_vector}
Detected issues:   {tags_str}
Rule triggered:    {rule_id}

--- ENVIRONMENTAL CONDITIONS ---
{env_block}

--- GROUNDWATER CONDITIONS ---
{gw_block}

--- SYSTEM RECOMMENDATION ---
{rule_recommendation}

--- YOUR TASK ---
{task_block}

Write in a warm, helpful tone. Do NOT use bullet points. Write in plain paragraph form.
Do NOT repeat the numbers from the diagnosis verbatim — interpret them naturally.
Keep it under 150 words. Remember: your entire response must be in {selected_language} only.
"""
    return prompt.strip()


# ---------------------------------------------------------------------------
# Groq API call
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _client() -> Groq:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return Groq(api_key=settings.groq_api_key, timeout=GROQ_TIMEOUT_S, max_retries=1)


def call_groq(prompt: str) -> str:
    """
    Send a single-turn prompt to Groq and return the model's answer text.
    Raises on any failure (missing key, network/API error, empty answer) —
    callers are expected to catch it and fall back.
    """
    completion = _client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,                  # low temp = factual, consistent
        max_completion_tokens=GROQ_MAX_TOKENS,
        reasoning_effort=GROQ_REASONING_EFFORT,
        include_reasoning=False,          # only return the final answer
    )

    choice = completion.choices[0]
    text   = (choice.message.content or "").strip()
    if not text:
        raise RuntimeError(f"Groq returned an empty answer (finish_reason={choice.finish_reason})")
    if choice.finish_reason == "length":
        logger.warning("Groq answer hit the %d-token limit and may be cut off", GROQ_MAX_TOKENS)
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_ai_explanation(
    # Recommendation engine output
    rule_recommendation: str,
    health_label:        str,
    health_score:        float,
    greenness_pct:       float,
    yellowing_pct:       float,
    necrosis_pct:        float,
    triggered_tags:      list[str],
    rule_id:             str,
    severity:            int,
    lrtdc_vector:        str,
    # Environmental inputs (optional)
    wind_force_ms:       Optional[float] = None,
    humidity_pct:        Optional[float] = None,
    temperature_c:       Optional[float] = None,
    sunlight_lux:        Optional[float] = None,
    # Farmer question (optional)
    users_question:      Optional[str] = None,
    # Language for the response
    selected_language:   str = "English",
    # Groundwater context (optional)
    groundwater_stats:   Optional[dict] = None,
) -> dict:
    """
    Call gpt-oss-120b on Groq and return a structured dict with:
      - ai_explanation : str   — the LLM-generated paragraph
      - model          : str   — model name used
      - ai_available   : bool  — False if the Groq call failed (fallback used)
      - fallback_used  : bool  — True if the rule recommendation was used instead

    This function NEVER raises — if Groq fails for any reason the
    rule-based recommendation is returned as the explanation so the
    main API response is never broken.
    """
    prompt = _build_prompt(
        rule_recommendation=rule_recommendation,
        health_label=health_label,
        health_score=health_score,
        greenness_pct=greenness_pct,
        yellowing_pct=yellowing_pct,
        necrosis_pct=necrosis_pct,
        triggered_tags=triggered_tags,
        rule_id=rule_id,
        severity=severity,
        lrtdc_vector=lrtdc_vector,
        wind_force_ms=wind_force_ms,
        humidity_pct=humidity_pct,
        temperature_c=temperature_c,
        sunlight_lux=sunlight_lux,
        users_question=users_question,
        selected_language=selected_language,
        groundwater_stats=groundwater_stats,
    )

    logger.debug("Groq prompt:\n%s", prompt)

    try:
        explanation = call_groq(prompt)
        logger.info("Groq response received (%d chars)", len(explanation))
        return {
            "ai_explanation": explanation,
            "model":          GROQ_MODEL,
            "ai_available":   True,
            "fallback_used":  False,
        }

    except Exception as exc:
        logger.warning("Groq call failed (%s). Using rule-based fallback.", exc)

    # Graceful fallback — return the rule recommendation as-is
    return {
        "ai_explanation": rule_recommendation,
        "model":          "rule-based-fallback",
        "ai_available":   False,
        "fallback_used":  True,
    }
