"""FastAPI application startup, lifespan, middleware and logging configuration."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app.config import APP_CONFIG, settings
from app.api.router import router, model_manager
from app.middleware.file_size import FileSizeMiddleware
from app.middleware.timeout import RequestTimeoutError
from app.middleware.error_handler import (
    global_exception_handler,
    validation_exception_handler,
    timeout_exception_handler,
    StructuredFormatter,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload models on startup if PRELOAD_MODELS is configured."""
    if settings.PRELOAD_MODELS:
        model_names = [m.strip() for m in settings.PRELOAD_MODELS.split(",") if m.strip()]
        await model_manager.preload(model_names)
    yield


app = FastAPI(
    title=APP_CONFIG["title"],
    lifespan=lifespan,
)

# Middleware
app.add_middleware(FileSizeMiddleware, max_size_mb=settings.MAX_FILE_SIZE_MB)

# Exception handlers
app.add_exception_handler(RequestTimeoutError, timeout_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)

# Router
app.include_router(router)

# Logging setup - connect StructuredFormatter to root logger
handler = logging.StreamHandler()
handler.setFormatter(StructuredFormatter())
logging.root.addHandler(handler)
logging.root.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
