"""Enhanced logging and error handling documentation for production deployments."""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from furniture_ai import __version__
from furniture_ai.config import Settings, get_settings

# Configure structured logging for production observability
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def log_request_info(settings: Annotated[Settings, Depends(get_settings)]) -> Settings:
    """Middleware dependency to log request context without exposing secrets."""
    logger.debug(
        "Request context: environment=%s, auth_enabled=%s",
        settings.environment,
        settings.service_auth_enabled,
    )
    return settings


def configure_error_handlers(app: FastAPI) -> None:
    """Configure global error handlers with user-friendly messages and logging."""

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc: HTTPException):
        """Log HTTP exceptions without exposing implementation details."""
        logger.warning(
            "HTTP %d: %s (path=%s, client=%s)",
            exc.status_code,
            exc.detail,
            request.url.path,
            request.client.host if request.client else "unknown",
        )
        return {"error": str(exc.detail), "status_code": exc.status_code}

    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc: Exception):
        """Catch-all for unexpected errors; log fully but return generic response."""
        logger.exception(
            "Unhandled exception (path=%s, client=%s): %s",
            request.url.path,
            request.client.host if request.client else "unknown",
            exc,
        )
        return {
            "error": "Internal server error",
            "status_code": 500,
        }, 500
