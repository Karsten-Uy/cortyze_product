"""Token-bucket rate limiter for outbound scraper requests.

Pure-Python, in-process. Single-worker constraint on the FastAPI service
makes a per-process bucket sufficient — Redis-backed coordination only
becomes necessary if/when the scheduler moves to a separate Railway
worker (see plan §5).

Usage:
    bucket = TokenBucket(rate_per_min=30, burst=10)
    if bucket.try_consume(1):
        # do the request
    else:
        # back off, log, retry next pass
"""

from __future__ import annotations

import time
from threading import Lock


class TokenBucket:
    """Classic token-bucket: capacity = `burst`, refill rate inferred from
    `rate_per_min`. Calls are non-blocking; callers handle the "no token"
    case themselves (typically: log + skip this scrape pass).
    """

    def __init__(self, *, rate_per_min: float, burst: int | None = None) -> None:
        if rate_per_min <= 0:
            raise ValueError(
                f"rate_per_min must be positive, got {rate_per_min!r}"
            )
        self._rate_per_sec = rate_per_min / 60.0
        self._capacity = float(burst if burst is not None else max(1, int(rate_per_min)))
        self._tokens = self._capacity
        self._last_refill = time.monotonic()
        self._lock = Lock()

    def try_consume(self, n: int = 1) -> bool:
        """Return True if `n` tokens were available and consumed."""
        with self._lock:
            self._refill_locked()
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False

    def available(self) -> float:
        """Snapshot of current token count (after a virtual refill)."""
        with self._lock:
            self._refill_locked()
            return self._tokens

    # ----------------------------------------------------------- internals

    def _refill_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed <= 0:
            return
        self._tokens = min(
            self._capacity, self._tokens + elapsed * self._rate_per_sec
        )
        self._last_refill = now
