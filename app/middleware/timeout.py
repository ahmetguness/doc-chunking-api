"""İstek zaman aşımı mekanizması.

asyncio.wait_for ile coroutine'leri timeout sarmalama.
"""

import asyncio


class RequestTimeoutError(Exception):
    """İstek zaman aşımına uğradığında fırlatılır."""
    pass


async def process_with_timeout(coro, timeout_seconds: float):
    """Verilen coroutine'i timeout ile çalıştırır.

    Args:
        coro: Çalıştırılacak coroutine.
        timeout_seconds: Maksimum bekleme süresi (saniye).

    Returns:
        Coroutine'in sonucu.

    Raises:
        RequestTimeoutError: Timeout aşılırsa.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise RequestTimeoutError()
