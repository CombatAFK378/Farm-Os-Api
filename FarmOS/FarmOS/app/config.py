
from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FARMOS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Server
    host: str  = "0.0.0.0"
    port: int  = 8000
    reload: bool = False

    # Logging
    log_level: str = "INFO"

    # API metadata
    app_title:       str = "FarmOS Leaf Vector API"
    app_version:     str = "1.0.0"
    app_description: str = (
        "AI-ready REST API that converts a leaf photograph into an LRTDC "
        "colour-vector payload for the FarmOS Deep Reinforcement Learning model."
    )

    # CORS — set to your Flutter app origin in production
    cors_origins: list[str] = ["*"]

    # Image processing limits
    max_image_bytes: int = 15 * 1024 * 1024   # 15 MB
    min_leaf_pixels: int = 50

    # Heavy jobs (leaf / groundwater analysis) allowed to run at once.
    # Keep at 2 on 512 MB instances; raise it on bigger plans.
    max_concurrent_jobs: int = 2

    # Groq API key for AI explanations — read from GROQ_API_KEY (or FARMOS_GROQ_API_KEY)
    groq_api_key: str = Field(
        "", validation_alias=AliasChoices("GROQ_API_KEY", "FARMOS_GROQ_API_KEY")
    )


# Singleton — import this everywhere
settings = Settings()