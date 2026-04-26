"""API schemas for BrainScore.

Extension fields (`user_id`, `model_version`, `raw_predictions_uri`) are
baked in from Stage 1 even though they are populated or required only in
later stages — see IMPLEMENTATION_PLAN.md §6.4. Adding optional fields later
is free; threading them through every layer in Stage 4 is not.
"""

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from .scoring.goals import Goal


class AnalyzeRequest(BaseModel):
    content_url: str
    content_type: Literal["video", "image", "text"]
    goal: Goal
    user_id: str | None = None
    request_id: str = Field(default_factory=lambda: str(uuid4()))


class BrainReport(BaseModel):
    request_id: str
    region_scores: dict[str, float]
    overall_score: float
    goal: Goal
    user_id: str | None = None
    model_version: str
    raw_predictions_uri: str | None = None
    elapsed_ms: int
