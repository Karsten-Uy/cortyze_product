"""Protocol + payload shape for Phase 2 (social context).

Real implementations will hit a knowledge graph and a trend firehose;
the protocol stays narrow on purpose so the orchestrator doesn't care
which backend is wired in.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Protocol

from pydantic import BaseModel, Field


class TrendReference(BaseModel):
    """One reference campaign discovered by the GraphRAG layer.

    Surfaces in the Results view as the optional `reference` card under
    a suggestion — see `core.schemas_v2.Reference` for the API-side
    rendering. We keep two near-identical shapes (this one + the API
    one) so the trends layer stays decoupled from the synthesis layer:
    Phase 3 picks which references attach to which suggestion.
    """

    brand: str
    campaign: str
    note: str
    # Two scores typically come back from the graph: one region-specific
    # ("Memory: 82") and one overall ("Overall: 91").
    score_region: int
    label_region: str  # e.g. "Memory"
    score_overall: int
    label_overall: str = "Overall"


class Entity(BaseModel):
    """One entity (brand / topic / person / event) lifted out of the
    user's brief or caption and cross-referenced against the rolling
    48-hour social-context graph.

    Surfaces only inside `TrendContext` for the time being — the frontend
    keeps reading the legacy `references` view, while Phase 3 (Claude)
    and any future audit tooling can pivot off the full entity list.
    """

    name: str
    type: Literal["BRAND", "TOPIC", "PERSON", "EVENT"]
    salience: float = 0.0
    # 0..1, slope of last-12h vs last-48h mention count for this entity.
    trend_velocity: float = 0.0
    sentiment_polarity: float = 0.0  # -1..1
    sarcasm_flag: bool = False
    # Per-platform peak mention rate, max-normalized across sources.
    platform_peaks: dict[str, float] = Field(default_factory=dict)


class TrendContext(BaseModel):
    """Phase 2 output — joined into Phase 3's prompt.

    `summary` is a 1-3 sentence plain-English snapshot Claude can quote.
    `references` is a small set of comparable campaigns that Phase 3
    may attach to specific suggestions.

    The remaining fields (entities, dominant_topic, brand_risk_score,
    cultural_moment, snapshot_timestamp, fallback_reason) are the v2
    GraphRAG payload — additive over the old shape so consumers that
    still read just `{summary, references}` keep working unchanged.
    """

    summary: str
    references: list[TrendReference] = []
    entities: list[Entity] = []
    dominant_topic: str | None = None
    brand_risk_score: float = 0.0  # 0..1
    cultural_moment: str | None = None
    snapshot_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    # Set when the GraphRAG client falls back to mock data — empty graph,
    # stale snapshot, Neo4j unreachable, etc. Honest audit trail without
    # blocking the pipeline.
    fallback_reason: str | None = None


class TrendClient(Protocol):
    """Single-method interface for Phase 2 implementations."""

    def fetch(
        self,
        *,
        brief: str,
        caption: str,
        goal: str,
        request_id: str | None = None,
    ) -> TrendContext: ...
