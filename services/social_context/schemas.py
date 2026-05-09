"""Internal Pydantic models for the GraphRAG pipeline.

These cover the shapes that flow between scrapers, the knowledge graph,
and the query layer. The user-facing `TrendContext` shape (rendered into
Phase 3's prompt and persisted as `trend_context.payload`) lives in
`services/trends/protocol.py` and is the only contract surface this
module owns end-to-end.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

# Re-export the public Entity model so internal callers can write
# `from services.social_context.schemas import Entity` without crossing
# the protocol boundary. Entities are the unit of knowledge in the
# graph — every node corresponds to one.
from services.trends.protocol import Entity  # noqa: F401

EdgeKind = Literal["SENTIMENT", "TRENDING_ON", "CO_OCCURS_WITH", "MENTIONED_IN"]


class EntityEdge(BaseModel):
    """A directed edge in the knowledge graph between two entities or
    between an entity and a context node (platform, snapshot, etc.).

    `weight` is interpretation-dependent: for SENTIMENT it's the polarity
    (-1..1), for TRENDING_ON it's the trend velocity at the time of
    ingestion, for CO_OCCURS_WITH it's the joint mention count.
    """

    src: str
    dst: str
    kind: EdgeKind
    weight: float = 0.0
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


SourceKind = Literal["reddit", "news", "trends", "x"]


class SourceSnapshot(BaseModel):
    """One raw item lifted off an external source — a Reddit post, a
    news article, a Google Trends rising-query bucket. Each scraper
    pass produces a list of these; the graph builder turns them into
    entities + edges."""

    source: SourceKind
    source_id: str  # platform-stable id; used for de-duping
    title: str = ""
    body: str = ""
    url: str | None = None
    author: str | None = None
    score: float = 0.0  # platform-native engagement signal (upvotes, etc.)
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    extra: dict[str, object] = Field(default_factory=dict)


class SentimentScore(BaseModel):
    """VADER polarity (-1..1) plus a cheap rule-based sarcasm flag.

    `subjectivity` is an optional secondary axis; we don't currently use
    it but the field is here so we can promote it without a migration.
    """

    polarity: float
    subjectivity: float = 0.0
    sarcasm_flag: bool = False


class IngestStats(BaseModel):
    """Per-source counters for the most recent scrape pass. Surfaced via
    `GET /health/social_context` so a Railway alarm can fire on degrade.
    """

    source: SourceKind
    snapshots_ingested: int = 0
    entities_added: int = 0
    edges_added: int = 0
    errors: int = 0
    latency_ms: int = 0
    finished_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
