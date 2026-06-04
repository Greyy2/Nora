"""Grey Backend - FastAPI Server.

This is the main application entrypoint used by Docker and local dev:
  - Docker: `uvicorn main:app --app-dir backend --port 8000`
  - Local:  `cd Grey && uvicorn backend.main:app --reload`

Routers live under `backend/routes/*`.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse, urlunparse
from fastapi import FastAPI, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from core.config import settings
from utils.error_handler import MongoConnectionError, normalize_error_response
from routes import (
    ai_routes,
    backtest,
    campaigns,
    carlo,
    chart,
    data,
    sheets,
    single_core,
    sova_routes,
    wfa,
    quanta
)
# Integrated Quanta V2 Engine
from routes.quanta_v2 import v2_engine

# Add engines to sys.path for internal imports
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _redact_uri(uri: str) -> str:
    """Mask Mongo credentials while keeping host/db visible for diagnostics."""
    try:
        parsed = urlparse(uri)
        if not parsed.scheme or "@" not in parsed.netloc:
            return uri

        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"
        return urlunparse(parsed._replace(netloc=f"***:***@{host}"))
    except Exception:
        return "<redacted>"

# Paths relative to backend/main.py
BACKEND_DIR = Path(__file__).resolve().parent
GREY_ROOT = BACKEND_DIR.parent
ENGINES_DIR = BACKEND_DIR / "engines"
FRONTEND_DIR = GREY_ROOT / "frontend"

if str(ENGINES_DIR / "quanta") not in sys.path:
    sys.path.insert(0, str(ENGINES_DIR / "quanta"))
if str(ENGINES_DIR / "sova") not in sys.path:
    sys.path.insert(0, str(ENGINES_DIR / "sova"))

app = FastAPI(
    title=settings.APP_NAME,
    description="Backtesting engine + AI routes",
    version=settings.VERSION,
)

# ============================================
# MIDDLEWARE: CORS
# ============================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# EXCEPTION HANDLERS
# ============================================

@app.exception_handler(MongoConnectionError)
async def mongo_connection_error_handler(request: Request, exc: MongoConnectionError):
    """Handle MongoDB connection errors"""
    logger.error(f"MongoDB connection error: {exc.detail}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "error",
            "code": 503,
            "message": exc.detail,
            "hint": "Backend is unable to reach the database. Please check if MongoDB is running."
        }
    )


@app.exception_handler(ConnectionFailure)
async def connection_failure_handler(request: Request, exc: ConnectionFailure):
    """Handle PyMongo connection failures"""
    logger.error(f"PyMongo connection failure: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "error",
            "code": 503,
            "message": "Database connection failed",
            "error_type": "ConnectionFailure",
            "hint": "Please ensure MongoDB service is running and accessible"
        }
    )


@app.exception_handler(ServerSelectionTimeoutError)
async def server_selection_timeout_handler(request: Request, exc: ServerSelectionTimeoutError):
    """Handle MongoDB server selection timeout"""
    logger.error(f"MongoDB server selection timeout: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "error",
            "code": 503,
            "message": "Database server unreachable",
            "error_type": "ServerSelectionTimeoutError",
            "hint": "MongoDB may be starting up. Please try again in a moment."
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions"""
    error_type = type(exc).__name__
    logger.error(f"Unhandled exception [{error_type}]: {str(exc)}", exc_info=True)
    
    # Don't expose internal details in production
    detail = str(exc) if settings.DEBUG else "Internal server error"
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "code": 500,
            "message": detail,
            "error_type": error_type
        }
    )

# ============================================
# ROUTERS
# ============================================
app.include_router(single_core.router)
app.include_router(data.router)
app.include_router(chart.router)
app.include_router(campaigns.router)
app.include_router(backtest.router)
app.include_router(wfa.router)
app.include_router(carlo.router)
app.include_router(ai_routes.router)
app.include_router(quanta.router)
app.include_router(v2_engine.router, prefix="/api/ai/quanta/v2")
app.include_router(sova_routes.router)
app.include_router(sheets.router)


# ============================================
# STATIC ARTIFACTS
# ============================================
app.mount(settings.ARTIFACT_MOUNT_PATH, StaticFiles(directory=str(settings.GREY_RESULTS_DIR)), name="results")

# NOTE: The QuantaAlpha Client UI is now rendered inline by the main Grey
# frontend (src/pages/Client) — no separate static build is needed here.


# ============================================
# HEALTH CHECK ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Root endpoint - basic health check"""
    return {
        "status": "ok",
        "service": "Grey Backend API",
        "version": settings.VERSION,
        "environment": "production" if not settings.DEBUG else "development"
    }


@app.get("/health")
async def health():
    """Basic health check"""
    return {"status": "healthy"}


@app.get("/health/db")
async def health_db():
    """Database connection health check"""
    try:
        from database.mongo_service import MongoService
        mongo = MongoService()
        if not mongo.ping():
            raise RuntimeError(getattr(mongo, '_last_ping_error', 'MongoDB ping failed'))
        mongo.close()
        return {
            "status": "healthy",
            "database": "connected",
            "db": settings.MONGO_DB,
            "uri": _redact_uri(settings.MONGO_URI)
        }
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "database": "disconnected",
                "db": settings.MONGO_DB,
                "uri": _redact_uri(settings.MONGO_URI),
                "error": str(e)
            }
        )
