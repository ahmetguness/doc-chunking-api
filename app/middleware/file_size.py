"""File size middleware for checking Content-Length header against configured limit."""

from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class FileSizeMiddleware:
    """ASGI middleware that rejects requests exceeding the configured file size limit.

    Checks the Content-Length header and returns HTTP 413 if the limit is exceeded.
    """

    def __init__(self, app: ASGIApp, max_size_mb: int):
        self.app = app
        self.max_size_mb = max_size_mb
        self.max_size_bytes = max_size_mb * 1024 * 1024

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            content_length = headers.get(b"content-length")
            if content_length and int(content_length) > self.max_size_bytes:
                response = JSONResponse(
                    status_code=413,
                    content={
                        "error_code": "FILE_TOO_LARGE",
                        "message": f"Dosya boyutu limiti: {self.max_size_mb}MB",
                    },
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)
