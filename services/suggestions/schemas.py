"""Re-exports of Event/Moment from core.schemas.

These types are part of the public BrainReport surface so they live in
`core.schemas` (per §6.2 one-way-dep rule). This module remains as a
backward-compatible import path for `services.suggestions` consumers.
"""

from core.schemas import Event, Moment

__all__ = ["Event", "Moment"]
