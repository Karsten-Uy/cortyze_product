"""Shared test fixtures and environment setup.

The API now gates protected routes behind Supabase JWT auth. Tests
should not have to mint real tokens — set AUTH_DISABLED=true so the
auth dependency returns a sentinel user_id and tests can hit endpoints
end-to-end the way they did before Stage 3 auth landed.

We also force-clear DATABASE_URL so the v2 `/runs` flow uses the
in-memory store regardless of whether the developer's shell has the
real Supabase DSN exported. Without this, running `uv run pytest` from
a shell that has loaded production secrets would write test runs into
the live database — a clear footgun.
"""

from __future__ import annotations

import os

# Set BEFORE pytest discovers any modules — api.auth + persistence
# read these at import time so the env values must be in place by then.
os.environ.setdefault("AUTH_DISABLED", "true")
os.environ["DATABASE_URL"] = ""
