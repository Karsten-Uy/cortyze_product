"""Shared slowapi rate-limiter instance.

Keyed on the client IP. The limit for /analyze is configurable via
RATE_LIMIT_ANALYZE (default: 20/minute). All other routes are unlimited
unless they add their own @limiter.limit() decorator.
"""

import os

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

# Read at import time — .env is loaded in api/main.py before any routes are
# imported, so os.environ is already populated when this module is first used.
ANALYZE_RATE: str = os.environ.get("RATE_LIMIT_ANALYZE", "20/minute")
