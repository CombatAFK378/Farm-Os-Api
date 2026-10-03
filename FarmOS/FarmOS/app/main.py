
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import leaf_router, groundwater_router

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_title,
        version=settings.app_version,
        description=settings.app_description,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── CORS ───────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ────────────────────────────────────────────────────────────
    app.include_router(leaf_router.router)
    app.include_router(groundwater_router.router)

    # ── Health check ───────────────────────────────────────────────────────
    # HEAD too: uptime monitors (e.g. UptimeRobot) often probe with HEAD
    @app.get("/health", tags=["System"])
    @app.head("/health", include_in_schema=False)
    async def health() -> dict:
        return {"status": "ok", "version": settings.app_version}

    if not settings.groq_api_key:
        logger.warning("GROQ_API_KEY is not set — AIexplanation will use rule-based fallback text")

    logger.info("FarmOS API ready on http://%s:%d", settings.host, settings.port)
    return app


app = create_app()