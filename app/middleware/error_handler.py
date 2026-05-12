"""Global exception handlers and structured logging formatter.

Provides standalone exception handler functions to be registered in main.py,
and a StructuredFormatter for JSON-based log output.
"""

import json
import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import settings
from app.middleware.timeout import RequestTimeoutError
from app.services.file_processor import (
    FileProcessingError,
    InvalidFileError,
    TooManyRowsError,
    UnsupportedFormatError,
    ZipPasswordRequiredError,
    ZipWrongPasswordError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP status code mapping for FileProcessingError subclasses
# ---------------------------------------------------------------------------
_FILE_ERROR_STATUS: dict[type, int] = {
    UnsupportedFormatError: 422,
    TooManyRowsError: 422,
    InvalidFileError: 400,
    ZipPasswordRequiredError: 400,
    ZipWrongPasswordError: 400,
}


# ---------------------------------------------------------------------------
# Exception handlers (registered via app.exception_handler in main.py)
# ---------------------------------------------------------------------------

async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unhandled exceptions.

    Returns a consistent error response and logs the details.
    """
    # Handle FileProcessingError subtypes with appropriate status codes
    if isinstance(exc, FileProcessingError):
        status_code = _FILE_ERROR_STATUS.get(type(exc), 400)
        logger.warning("File processing error: %s", exc.message)
        return JSONResponse(
            status_code=status_code,
            content={
                "error_code": exc.error_code,
                "message": exc.message,
            },
        )

    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_ERROR",
            "message": "Beklenmeyen bir hata oluştu",
            "detail": str(exc),
        },
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return Pydantic / FastAPI validation errors as HTTP 400."""
    logger.warning("Validation error: %s", exc.errors())
    return JSONResponse(
        status_code=400,
        content={
            "error_code": "INVALID_PARAMETER",
            "message": "Parametre doğrulama hatası",
            "detail": exc.errors(),
        },
    )


async def timeout_exception_handler(
    request: Request, exc: RequestTimeoutError
) -> JSONResponse:
    """Return request timeout as HTTP 408."""
    logger.warning("Request timeout exceeded (%ss)", settings.REQUEST_TIMEOUT_SECONDS)
    return JSONResponse(
        status_code=408,
        content={
            "error_code": "REQUEST_TIMEOUT",
            "message": "İstek zaman aşımına uğradı",
            "detail": f"Maksimum işleme süresi: {settings.REQUEST_TIMEOUT_SECONDS}s",
        },
    )


# ---------------------------------------------------------------------------
# Structured JSON log formatter
# ---------------------------------------------------------------------------

class StructuredFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "timestamp": self.formatTime(record),
                "level": record.levelname,
                "message": record.getMessage(),
                "module": record.module,
                "request_id": getattr(record, "request_id", None),
            },
            ensure_ascii=False,
        )
