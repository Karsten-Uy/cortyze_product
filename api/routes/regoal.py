"""POST /reports/{request_id}/regoal — re-run the suggestion engine
under a different goal, without re-running TRIBE inference.

The expensive part of `/analyze` is TRIBE v2 inference (~30-700s
depending on input). The downstream pipeline (atlas mapping, calibration,
goal-weighted overall, suggestion engine) is goal-dependent but cheap.
This endpoint reuses the cached predictions / region_scores and only
recomputes:

  - `overall_score` for the new goal (already cached in `overall_by_goal`,
    but we recompute deterministically for safety)
  - `suggestions` via `diagnose()` with the new goal — one Anthropic
    call (~$0.01) when SUGGESTION_LLM_MODE is paid

Behavior: a NEW report row is created with a fresh request_id, copying
everything goal-independent and overwriting the goal-dependent bits.
The original report is untouched. The new row appears in the sidebar
beside the original so the user can compare side-by-side via /compare.
"""

from __future__ import annotations

import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from core.schemas import BrainReport
from core.scoring.goals import Goal, overall_score
from services.persistence.reports import get_store
from services.suggestions import diagnose, is_enabled as suggestions_enabled

from ..auth import require_user

router = APIRouter()
_log = logging.getLogger(__name__)


class RegoalRequest(BaseModel):
    goal: Goal


@router.post("/reports/{request_id}/regoal", response_model=BrainReport)
def regoal(
    request_id: str,
    body: RegoalRequest,
    user_id: str = Depends(require_user),
) -> BrainReport:
    store = get_store()
    if store is None:
        raise HTTPException(
            status_code=501,
            detail="Reports persistence not configured. Set DATABASE_URL env var.",
        )

    original = store.get(request_id)
    if original is None:
        raise HTTPException(status_code=404, detail=f"Report {request_id} not found")
    if original.user_id and original.user_id != user_id:
        raise HTTPException(status_code=404, detail=f"Report {request_id} not found")

    if original.goal == body.goal:
        # No-op regoal — return the original. Saves a Claude call when the
        # frontend accidentally re-fires the current goal.
        return original

    # Goal-independent state we copy verbatim.
    new_overall = overall_score(original.region_scores, body.goal)
    new_request_id = str(uuid4())

    new_report = original.model_copy(
        update={
            "request_id": new_request_id,
            "goal": body.goal,
            "overall_score": new_overall,
            "suggestions": [],
            # Title gets a marker so the sidebar shows the regoal'd run
            # distinctly from its parent.
            "title": (
                f"{original.title} ({body.goal.value})"
                if original.title
                else f"Regoaled · {body.goal.value}"
            ),
            # Don't carry created_at; let DB default fire.
            "created_at": None,
            # The brain image is goal-independent — same predictions →
            # same heatmap. Reuse the original's R2-stored URI so the
            # frontend doesn't re-render or re-upload. The presign in
            # /report/{id} re-mints a fresh URL at load-time.
            "brain_image_uri": original.brain_image_uri,
        }
    )

    # Re-run the suggestion engine with the new goal. Wrapped so that
    # if the LLM errors, we still persist the new report (with empty
    # suggestions) — the user gets the new overall + region scores
    # immediately and can retry suggestions later.
    if suggestions_enabled():
        try:
            new_report.suggestions = diagnose(
                new_report,
                image_count=original.image_count or 0,
                seconds_per_image=original.seconds_per_image or 2.5,
                has_audio=bool(original.audio_url),
                additional_context=original.additional_context,
            )
            _log.info(
                "regoal request_id=%s -> %s, goal=%s, %d suggestions",
                request_id,
                new_request_id,
                body.goal.value,
                len(new_report.suggestions),
            )
        except Exception as e:
            _log.warning(
                "regoal request_id=%s suggestion engine errored: %s",
                request_id,
                e,
            )

    store.insert(new_report)
    return new_report
