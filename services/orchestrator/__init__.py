"""Phase 1→2→3→4 orchestrator for the v2 (`/runs`) pipeline.

Public surface:

  * `start_run(...)`  — kick off a run, returns immediately with the
                        run record. The pipeline runs in a background
                        task and writes back to the persistence layer.
  * `subscribe(run_id)` — async iterator of progress events over SSE.

Implementation details live in `runner.py` and `events.py`.
"""

from __future__ import annotations

from .events import EVENT_BUS, RunEvent  # noqa: F401
from .runner import start_run  # noqa: F401
