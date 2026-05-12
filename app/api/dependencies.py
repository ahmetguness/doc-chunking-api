"""Concurrency control and authentication dependencies for the API layer."""

import asyncio
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import HTTPException, Request


class ConcurrencyLimiter:
    def __init__(self, max_concurrent: int, acquire_timeout: float = 30.0):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent
        self._acquire_timeout = acquire_timeout
        self._waiting = 0

    @asynccontextmanager
    async def throttle(self):
        self._waiting += 1
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(), timeout=self._acquire_timeout
            )
        except asyncio.TimeoutError:
            raise
        finally:
            self._waiting -= 1
        try:
            yield
        finally:
            self._semaphore.release()

    @property
    def stats(self) -> dict:
        active = self._max_concurrent - self._semaphore._value
        return {"active": max(0, active), "waiting": self._waiting}


# ---------------------------------------------------------------------------
# Token-based authentication
# ---------------------------------------------------------------------------

def _get_chunking_token() -> str:
    return os.environ.get("CHUNKING_AUTH_TOKEN", "")


async def verify_chunking_token(request: Request) -> None:
    token_expected = _get_chunking_token()
    if not token_expected:
        return

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header gereklidir.")

    token = auth_header[7:]
    if not secrets.compare_digest(token, token_expected):
        raise HTTPException(status_code=401, detail="Geçersiz token.")
