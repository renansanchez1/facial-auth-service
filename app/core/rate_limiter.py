import asyncio
import time
from collections import defaultdict, deque
from typing import Tuple

from app.core.config import settings


class RateLimiter:
    """
    Sliding window rate limiter em memória.
    Para produção com múltiplos workers, substitua por Redis (ex: redis-py + ZADD/ZCOUNT).
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._windows: dict[str, deque] = defaultdict(deque)

    async def is_allowed(self, key: str) -> Tuple[bool, int]:
        """Retorna (permitido, retry_after_segundos)."""
        now = time.monotonic()
        window = settings.RATE_LIMIT_WINDOW_SECONDS
        limit = settings.RATE_LIMIT_REQUESTS

        async with self._lock:
            dq = self._windows[key]
            # Remove timestamps fora da janela
            while dq and dq[0] < now - window:
                dq.popleft()

            if len(dq) >= limit:
                retry_after = int(window - (now - dq[0])) + 1
                return False, retry_after

            dq.append(now)
            return True, 0
