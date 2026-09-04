from __future__ import annotations

from furniture_ai.api import app
from furniture_ai.api_v2 import router as api_v2_router

app.include_router(api_v2_router)

__all__ = ["app"]
