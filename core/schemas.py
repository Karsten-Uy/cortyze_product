"""API schemas for BrainScore.

Extension fields (`user_id`, `model_version`, `raw_predictions_uri`) are
baked in from Stage 1 even though they are populated or required only in
later stages — see IMPLEMENTATION_PLAN.md §6.4. Adding optional fields later
is free; threading them through every layer in Stage 4 is not.

`Moment` and `Event` live here (rather than in services/) because they're
part of the public BrainReport surface — the frontend types against them
and the §6.2 one-way-dep rule says core/ doesn't import from services/.
The services/suggestions module re-exports them for backward import paths.
"""

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from .scoring.goals import Goal


class AnalyzeRequest(BaseModel):
    """Inputs to /analyze.

    Two shapes are supported:

    - **`content_type="video"`**: single MP4 URL via `content_url`. The
      audio + transcription + frames are all derived from that one URL
      by tribev2's pipeline.

    - **`content_type="post"`**: static social-media post(s) with 1-20
      images. `image_urls` is the ordered list of image URLs (a single
      image is still a list of length 1). Each image is held for
      `seconds_per_image` seconds in the synthesized video. `audio_url`
      (optional) supplies an audio track that plays continuously across
      all images. `caption` (optional) is the written caption,
      converted to synthetic Word events at reading-rate. At least one
      of `audio_url` / `caption` must be supplied.

    The post flow is image-count-agnostic at the data layer — the
    suggestion engine differentiates single-image posts from carousels
    when picking prompts (see `services/suggestions/prompts.py`), but
    the brain pipeline treats them uniformly.

    `image` and `text` content types remain reserved for future
    single-modality flows.
    """

    content_type: Literal["video", "image", "text", "post"] = "video"
    goal: Goal
    user_id: str | None = None
    request_id: str = Field(default_factory=lambda: str(uuid4()))

    # Video flow
    content_url: str | None = None

    # Post flow (1-20 images + optional audio + optional caption)
    image_urls: list[str] | None = None
    seconds_per_image: float = 2.5
    audio_url: str | None = None
    caption: str | None = None

    # Optional brand / campaign context the user types in alongside the
    # upload. Plumbed into the suggestion engine's user prompt so Claude
    # can ground fixes in the actual brand/audience instead of inventing
    # generic ones. Empty string treated same as None.
    additional_context: str | None = None

    # Optional campaign grouping. When set, the resulting BrainReport is
    # filed under this campaign in the sidebar and listing endpoints.
    campaign_id: str | None = None

    # Optional human-readable label for the run. Surfaced in the sidebar
    # so creators can find runs without recognizing UUIDs. Falls back to
    # caption / filename / goal if unset.
    title: str | None = None

    @model_validator(mode="after")
    def _validate_inputs(self) -> "AnalyzeRequest":
        if self.content_type == "video":
            if not self.content_url:
                raise ValueError("content_type='video' requires content_url")
        elif self.content_type == "post":
            if not self.image_urls or len(self.image_urls) < 1:
                raise ValueError(
                    "content_type='post' requires image_urls with at least 1 entry"
                )
            if len(self.image_urls) > 20:
                raise ValueError(
                    f"content_type='post' supports up to 20 images "
                    f"(got {len(self.image_urls)})"
                )
            if not (self.audio_url or self.caption):
                # Pure image-only post works but loses 2 of 3 modalities;
                # warn loudly via validation rather than silently underperform.
                raise ValueError(
                    "content_type='post' requires at least one of "
                    "audio_url or caption (image-only posts produce very "
                    "weak Engagement/Brand Recall scores)"
                )
            if not (0.5 <= self.seconds_per_image <= 10.0):
                raise ValueError(
                    "seconds_per_image must be in [0.5, 10.0] "
                    f"(got {self.seconds_per_image})"
                )
        return self


class Event(BaseModel):
    """A type-tagged span of input content with optional text.

    Mirrors what tribev2 emits in `model.get_events_dataframe()`. The GPU
    worker serializes these and the API decodes them so the suggestion
    engine can explain WHEN problems happen and WHAT was on screen.
    """

    type: Literal["Word", "Sentence", "Audio", "Video", "Unknown"]
    start_s: float
    duration_s: float
    text: str | None = None


class Moment(BaseModel):
    """A window where a brain region's per-timestep score crossed a threshold.

    Dips identify *where* a content problem happens; peaks identify
    strengths. Both are pre-computed in api/predict.py so the Stage 2
    suggestion engine can build timestamp-anchored prompts and the
    frontend can render dip/peak chips on each region card.
    """

    region: str
    type: Literal["dip", "peak"]
    start_s: float
    end_s: float
    avg_score: float
    context: str = ""
    events: list[Event] = []


class Suggestion(BaseModel):
    """One actionable diagnosis attached to a region in the BrainReport.

    Generated by the Stage 2 suggestion engine
    (services.suggestions.diagnose). Each suggestion is anchored to a
    region + (optional) timestamp window and includes a short fix the
    creator can act on. Stage 2 also matches each suggestion to high-
    scoring reference ads — those land in `examples` once
    services.examples.library is wired in.
    """

    region: str
    priority: Literal["critical", "important", "minor"]
    title: str
    fix: str
    why: str
    # Anchoring for video / post-with-audio: real timestamps in seconds.
    timestamp_start_s: float | None = None
    timestamp_end_s: float | None = None
    # Anchoring for galleries: 1-indexed image position(s) the suggestion
    # targets. Both ends inclusive ("image 2" → start=2, end=2; "images
    # 2-3" → start=2, end=3). Mutually exclusive with timestamp fields in
    # practice — a fix references either a moment in time or an image,
    # not both.
    image_index_start: int | None = None
    image_index_end: int | None = None
    examples: list[str] = []  # reference ad names; full lookup in services.examples


class BrainReport(BaseModel):
    request_id: str
    region_scores: dict[str, float]
    overall_score: float
    goal: Goal
    # Mirrors AnalyzeRequest.content_type so downstream consumers
    # (suggestion engine, frontend, Stage 4 audience profiles) know which
    # content shape produced this report. Defaults to "video" so older
    # reports deserialized without this field stay valid.
    content_type: Literal["video", "image", "text", "post"] = "video"
    user_id: str | None = None
    model_version: str
    raw_predictions_uri: str | None = None
    brain_image_b64: str | None = None
    # R2 URI / presigned URL for the rendered brain heatmap PNG. Persisted
    # so past runs can re-render without recomputing. brain_image_b64 is
    # the inline data-uri-friendly form returned to the frontend; either
    # may be set independently (b64 inline for fresh runs, URL for loads).
    brain_image_uri: str | None = None
    # request_id whose R2 key holds the brain image. Equals `request_id` for
    # ordinary runs; for a regoal'd run it points at the parent's id, since
    # the regoal reuses the parent's PNG rather than re-rendering. Lets
    # /report/{id} fetch / re-presign against the correct key.
    brain_image_request_id: str | None = None
    elapsed_ms: int
    # Stage 2 fields — populated when temporal pipeline runs (always today;
    # keeping defaults so existing JSON consumers tolerate older payloads).
    region_timeseries: dict[str, list[float]] | None = None
    moments: list[Moment] = []
    suggestions: list[Suggestion] = []
    # Stage-3 account-aware fields — mirror AnalyzeRequest so the report
    # can be re-rendered with the same context that produced it.
    additional_context: str | None = None
    campaign_id: str | None = None
    title: str | None = None
    thumbnail_url: str | None = None
    caption_text: str | None = None
    created_at: str | None = None  # ISO-8601 timestamp; populated by persistence layer

    # The same weighted-sum overall, recomputed under all four goals.
    # Lets the frontend swap the goal lens without re-hitting the backend
    # — region_scores are goal-independent; only weights change.
    # `goal` above is the *suggestion-engine* goal (the one the LLM
    # responses were generated against).
    overall_by_goal: dict[str, float] | None = None

    # Inputs persisted so /reports/{id}/regoal can correctly re-run the
    # suggestion engine with the same context. Without these, regoal would
    # lose audio-presence (the phantom-audio fix) and carousel-shape info.
    audio_url: str | None = None
    image_count: int | None = None
    seconds_per_image: float | None = None


class Campaign(BaseModel):
    """An ad-campaign grouping. One creator can have many campaigns; each
    campaign is a folder in the sidebar that aggregates multiple runs for
    a single product launch / brief / experiment.
    """

    id: str
    user_id: str
    name: str
    description: str | None = None
    created_at: str  # ISO-8601


class CampaignSummary(BaseModel):
    """Lightweight campaign view for the sidebar."""

    id: str
    name: str
    description: str | None = None
    run_count: int
    last_run_at: str | None = None  # ISO-8601 of most recent run, or None if empty


class ReportSummary(BaseModel):
    """Compact projection of a BrainReport for the sidebar list. Avoids
    shipping the full ~MB report when all the sidebar needs is a label and
    a score.
    """

    request_id: str
    title: str | None = None
    thumbnail_url: str | None = None
    overall_score: float
    goal: Goal
    content_type: Literal["video", "image", "text", "post"] = "video"
    campaign_id: str | None = None
    created_at: str  # ISO-8601


class ComparisonResult(BaseModel):
    """Output of POST /compare. Returns both reports plus the diff so the
    frontend can render side-by-side without a second round-trip.
    """

    report_a: BrainReport
    report_b: BrainReport
    overall_delta: float  # b.overall - a.overall (positive = B wins)
    per_region_delta: dict[str, float]  # b.score - a.score per region
    winner: Literal["a", "b", "tie"]
    llm_summary: str  # human-readable "why one wins" from Anthropic
