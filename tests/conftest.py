"""Shared test fixtures and environment setup.

The API now gates protected routes behind Supabase JWT auth. Tests
should not have to mint real tokens — set AUTH_DISABLED=true so the
auth dependency returns a sentinel user_id and tests can hit endpoints
end-to-end the way they did before Stage 3 auth landed.
"""

from __future__ import annotations

import os

# Set BEFORE pytest discovers any modules — api.auth reads it at request
# time so this just needs to be in os.environ before the first request.
os.environ.setdefault("AUTH_DISABLED", "true")
