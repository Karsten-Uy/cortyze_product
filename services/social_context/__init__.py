"""Phase 2 GraphRAG implementation — social context pipeline.

Maintains a rolling 48-hour knowledge graph of trending entities,
sentiment, and platform activity. The orchestrator's `_run_phase_2`
calls into this package via `services.trends.GraphRAGTrendClient`,
which itself lives in `client.py` (PR #3).

Public surface is intentionally narrow:
  * `KnowledgeGraph`       — storage abstraction (NetworkX, Neo4j later)
  * `extract_entities`     — spaCy-backed NER over user briefs / scraped text
  * `score_sentiment`      — VADER polarity + heuristic sarcasm flag
  * `get_trend_context`    — query-time assembly of `TrendContext` (PR #3)
  * `start_scheduler`      — APScheduler boot hook (PR #4)

Imports are lazy at module level so a backend with `TRENDS_MODE=mock`
doesn't pay the cost of importing spaCy / VADER / NetworkX. Each helper
is only loaded when its respective surface is touched.
"""

from __future__ import annotations

# Re-export the commonly-imported names. Heavy modules are NOT eagerly
# imported here — callers should `from services.social_context.graph
# import NetworkXGraph` etc. when they actually need them.

__all__ = [
    "schemas",
    "graph",
    "entities",
    "sentiment",
]
