"""Protocol + payload shape for Phase 2 (social context).

Real implementations will hit a knowledge graph and a trend firehose;
the protocol stays narrow on purpose so the orchestrator doesn't care
which backend is wired in.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


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


class TrendContext(BaseModel):
    """Phase 2 output — joined into Phase 3's prompt.

    `summary` is a 1-3 sentence plain-English snapshot Claude can quote.
    `references` is a small set of comparable campaigns that Phase 3
    may attach to specific suggestions.
    """

    summary: str
    references: list[TrendReference] = []


class TrendClient(Protocol):
    """Single-method interface for Phase 2 implementations."""

    def fetch(self, *, brief: str, caption: str, goal: str) -> TrendContext:
        ...
