"""Query-time assembly of `TrendContext` from the knowledge graph.

Pipeline, end to end:
  1. Extract entities from `{brief}\n{caption}\n{goal}` via spaCy NER.
  2. For each entity, walk the graph one hop out to gather sentiment +
     trending edges.
  3. Aggregate per-entity stats — `trend_velocity`, `sentiment_polarity`,
     `sarcasm_flag`, `platform_peaks` — from the gathered edges.
  4. Roll up to `TrendContext` fields: `dominant_topic`,
     `brand_risk_score`, `cultural_moment`, `summary` (templated).
  5. Synthesize the legacy `references` view from the top entities so
     the frontend keeps rendering reference cards unchanged.

Pure, deterministic, no LLM calls. Phase 3 owns LLM cost.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from services.trends.protocol import Entity, TrendContext, TrendReference

from .graph import KnowledgeGraph
from .schemas import EdgeKind  # noqa: F401  (re-exported for callers)

_log = logging.getLogger("cortyze.social_context.query")

# When two SENTIMENT edges contradict, sarcasm tips the average toward
# the negative side — a "great" caption rated -0.6 by VADER (because of
# context) shouldn't get washed out by a single +0.1 vote.
_SARCASM_DAMPING = 1.4

# An entity counts as "trending" if it has a non-trivial fraction of
# the recent total mention count. Used by `_velocity_from_attrs`.
_TRENDING_RECENCY_HOURS = 12


def get_trend_context(
    *,
    brief: str,
    caption: str,
    goal: str,
    graph: KnowledgeGraph,
    request_id: str | None = None,
) -> TrendContext:
    """Assemble a `TrendContext` for one user run.

    Caller is responsible for handling the empty-graph case before
    invoking this — see `client.GraphRAGTrendClient.fetch`. This
    function will happily return a TrendContext with no entities if
    none of the user's text matches the graph; that's a valid signal
    Phase 3 can interpret.
    """
    # Lazy import so an environment without spaCy doesn't fail to load
    # this module — the regex fallback in entities.py kicks in.
    from .entities import extract_entities

    text = "\n".join(filter(None, [brief, caption, goal]))
    seeds = extract_entities(text)

    if not seeds:
        return _empty_context(reason=None)

    # Match each seed entity to a graph node (best-effort lexical),
    # collect graph nodes + their neighbors, deduplicate by lowercased
    # name. Each unique entity gets its raw graph attrs assembled into
    # the `Entity` shape Phase 3 reads.
    by_name: dict[str, Entity] = {}
    for seed in seeds:
        matches = graph.query_entities_for_text(seed.name, k=3)
        for m in matches:
            key = m.name.lower()
            if key not in by_name:
                by_name[key] = m
        # Neighbors give us context — co-occurring topics, sentiment
        # nodes, platform peaks the user's text didn't mention directly.
        for m in matches:
            for nb in graph.neighbors(m.name, depth=1):
                nb_key = nb.name.lower()
                if nb_key not in by_name:
                    by_name[nb_key] = nb

    if not by_name:
        return _empty_context(reason=None)

    enriched = [_enrich_with_neighbors(e, graph) for e in by_name.values()]

    # Sort by salience × velocity for the "what matters most" axis
    # Phase 3 will read first.
    enriched.sort(
        key=lambda e: e.salience * (1.0 + e.trend_velocity),
        reverse=True,
    )

    dominant = enriched[0] if enriched else None
    risk = _compute_brand_risk(enriched)
    cultural = _detect_cultural_moment(enriched, graph)

    # Templated summary — deterministic, no LLM. Phase 3 (Claude) takes
    # the structured fields and writes the actual user-facing prose.
    summary = _summarize(enriched, dominant=dominant, goal=goal)

    references = _references_from_entities(enriched, goal=goal)

    ctx = TrendContext(
        summary=summary,
        references=references,
        entities=enriched,
        dominant_topic=dominant.name if dominant else None,
        brand_risk_score=risk,
        cultural_moment=cultural,
        snapshot_timestamp=datetime.now(timezone.utc),
        fallback_reason=None,
    )
    if request_id:
        _log.info(
            "trend_context request_id=%s entities=%d dominant=%r risk=%.2f",
            request_id,
            len(enriched),
            dominant.name if dominant else None,
            risk,
        )
    return ctx


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _enrich_with_neighbors(
    entity: Entity, graph: KnowledgeGraph
) -> Entity:
    """Walk one hop out from `entity` and use the edges to populate
    trend_velocity / sentiment_polarity / sarcasm_flag.

    The graph stores raw attrs (mention counts, platform counts, last
    seen). We compute the derived signals here so the graph layer stays
    a dumb store.
    """
    # platform_peaks already populated by `_attrs_to_entity` on read.
    velocity = _velocity_from_neighbors(entity, graph)
    polarity, sarcasm = _sentiment_from_neighbors(entity, graph)
    return entity.model_copy(
        update={
            "trend_velocity": velocity,
            "sentiment_polarity": polarity,
            "sarcasm_flag": sarcasm,
        }
    )


def _velocity_from_neighbors(
    entity: Entity, graph: KnowledgeGraph
) -> float:
    """Crude trend velocity: ratio of recent neighbors (<12h) to total.

    Real implementation would hit edge timestamps directly; this
    approximation works because each ingest pass adds a fresh node +
    edges, so recent neighbor count tracks recent mention frequency.
    """
    neighbors = graph.neighbors(entity.name, depth=1)
    if not neighbors:
        return 0.0
    cutoff = datetime.now(timezone.utc) - timedelta(
        hours=_TRENDING_RECENCY_HOURS
    )
    recent = sum(
        1
        for nb in neighbors
        # `last_seen` was lost in the projection — fall back to peak
        # heuristic: an entity with strong platform_peaks is recent.
        if max(nb.platform_peaks.values(), default=0.0) > 0.5
    )
    velocity = recent / len(neighbors)
    # Boost by raw mention salience so a hot but isolated entity isn't
    # capped at zero.
    return min(velocity + entity.salience * 0.2, 1.0)


def _sentiment_from_neighbors(
    entity: Entity, graph: KnowledgeGraph
) -> tuple[float, bool]:
    """Best-effort sentiment from edge weights.

    Stage-2 PR #2 ships without scraper-driven sentiment edges yet —
    so this returns (0.0, False) until PR #4 starts feeding the graph.
    Once SENTIMENT edges land, this function picks them up because it
    only inspects the graph (graph store doesn't change shape).
    """
    # Hook for the future — keep the signature stable.
    del entity, graph
    return 0.0, False


def _compute_brand_risk(entities: list[Entity]) -> float:
    """`brand_risk_score = max(|min(polarity)| * sarcasm_factor)` clipped to [0,1]."""
    if not entities:
        return 0.0
    worst = min((e.sentiment_polarity for e in entities), default=0.0)
    sarcasm_factor = (
        _SARCASM_DAMPING if any(e.sarcasm_flag for e in entities) else 1.0
    )
    return max(0.0, min(abs(worst) * sarcasm_factor, 1.0))


def _detect_cultural_moment(
    entities: list[Entity], graph: KnowledgeGraph
) -> str | None:
    """Pick the highest-velocity neighbor cluster name, if any.

    Heuristic: among the entities, find the highest-velocity one whose
    type is EVENT — that maps cleanly to "cultural moment" (e.g.,
    "Super Bowl", "Olympics", "Coachella"). Falls back to None.
    """
    del graph
    events = [
        e
        for e in entities
        if e.type == "EVENT" and e.trend_velocity > 0.5
    ]
    if not events:
        return None
    events.sort(key=lambda e: e.trend_velocity, reverse=True)
    return events[0].name


def _summarize(
    entities: list[Entity],
    *,
    dominant: Entity | None,
    goal: str,
) -> str:
    """Deterministic 1–3 sentence summary. Claude rewrites this anyway,
    but Phase 3's prompt cache treats stable input strings as a hit, so
    keeping this templated (and short) buys real cache savings.
    """
    if not entities or dominant is None:
        return (
            f"No comparable trending campaigns surfaced for the {goal} "
            "goal in the last 48 hours. The synthesis layer will fall "
            "back to region-gap analysis only."
        )
    sentiment_word = (
        "warm"
        if dominant.sentiment_polarity > 0.2
        else "tense"
        if dominant.sentiment_polarity < -0.2
        else "neutral"
    )
    sarcasm_note = (
        " Conversation around it carries a sarcastic tone, so cultural "
        "framing matters more than usual."
        if dominant.sarcasm_flag
        else ""
    )
    return (
        f"\"{dominant.name}\" is the dominant {dominant.type.lower()} in the "
        f"current {goal} window with {sentiment_word} sentiment.{sarcasm_note} "
        f"{len(entities)} related entities are active across "
        f"{', '.join(sorted(set(p for e in entities for p in e.platform_peaks)))}."
    )


def _references_from_entities(
    entities: list[Entity], *, goal: str
) -> list[TrendReference]:
    """Synthesize the legacy `references` view from the top entities.

    Best-effort match against the static reference-ad library so the
    frontend keeps rendering Reference cards. If no library match,
    returns an empty list — Phase 3 will simply not attach a reference
    card to that suggestion.
    """
    try:
        from services.examples.library import top_n_for_region
    except Exception:  # noqa: BLE001 — defensive against missing data dir
        return []

    if not entities:
        return []

    out: list[TrendReference] = []
    seen_brands: set[str] = set()
    # Pull a handful of high-quality references from the library; pick
    # the ones whose region keys match the spirit of the goal.
    region_for_goal = _region_for_goal(goal)
    for ad in top_n_for_region(region_for_goal, n=4):
        brand = ad.get("display_name") or ad.get("name") or ""
        if not brand or brand in seen_brands:
            continue
        seen_brands.add(brand)
        rs = ad.get("region_scores") or {}
        ob = ad.get("overall_by_goal") or {}
        out.append(
            TrendReference(
                brand=brand,
                campaign=ad.get("description", "")[:60] or "—",
                note="Recent comparable; matched on region peak.",
                score_region=int(rs.get(region_for_goal, 0)),
                label_region=region_for_goal.title(),
                score_overall=int(ob.get(goal, 0)),
                label_overall="Overall",
            )
        )
    return out


_GOAL_TO_REGION: dict[str, str] = {
    "brand_recall": "hippocampus",
    "purchase_intent": "reward",
    "emotional_resonance": "amygdala",
    "trust": "fusiform_face",
    "attention": "visual_cortex",
}


def _region_for_goal(goal: str) -> str:
    return _GOAL_TO_REGION.get(goal, "visual_cortex")


def _empty_context(*, reason: str | None) -> TrendContext:
    return TrendContext(
        summary=(
            "No matching entities in the rolling 48-hour graph. Synthesis "
            "will proceed on region-gap analysis alone."
        ),
        references=[],
        entities=[],
        dominant_topic=None,
        brand_risk_score=0.0,
        cultural_moment=None,
        snapshot_timestamp=datetime.now(timezone.utc),
        fallback_reason=reason,
    )
