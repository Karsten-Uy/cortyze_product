"""Knowledge-graph storage abstraction for Phase 2 GraphRAG.

`KnowledgeGraph` is the narrow interface every scrape pass and every
query-time call uses; it has two concrete implementations:

  * `NetworkXGraph` — in-process `nx.MultiDiGraph` wrapped behind an
    `RLock`. Default backend (`GRAPH_BACKEND=networkx`). Periodically
    pickled to disk + mirrored to Postgres `trend_snapshots.payload`
    for crash recovery.

  * `Neo4jGraph` — Neo4j AuraDB-backed. Lands in PR #5.
    Triggered by `GRAPH_BACKEND=neo4j`.

The interface intentionally mirrors what `query.get_trend_context`
needs — no leaky NetworkX-specific helpers leak through. That keeps
the Neo4j swap to a one-line factory change.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from threading import RLock
from typing import Any

from .schemas import EdgeKind, Entity, EntityEdge, SourceSnapshot

_log = logging.getLogger("cortyze.social_context.graph")


class KnowledgeGraph(ABC):
    """Single-method-per-concept storage interface.

    Implementations must be thread-safe — the scheduler ingests on a
    background thread while the API serves `get_trend_context` queries
    on the event loop.
    """

    @abstractmethod
    def add_entity(self, entity: Entity, snapshot: SourceSnapshot) -> None: ...

    @abstractmethod
    def add_edge(self, edge: EntityEdge) -> None: ...

    @abstractmethod
    def query_entities_for_text(
        self, text: str, k: int = 10
    ) -> list[Entity]: ...

    @abstractmethod
    def neighbors(
        self, entity_id: str, depth: int = 1
    ) -> list[Entity]: ...

    @abstractmethod
    def prune_older_than(self, cutoff: datetime) -> int: ...

    @abstractmethod
    def node_count(self) -> int: ...

    @abstractmethod
    def last_ingest_at(self) -> datetime | None: ...

    @abstractmethod
    def healthcheck(self) -> bool: ...


# ---------------------------------------------------------------------------
# NetworkX backend
# ---------------------------------------------------------------------------


def _entity_id(name: str) -> str:
    """Canonical node id — lowercased + whitespace-collapsed.

    Two entities with the same lemma collapse to the same node, which is
    what we want: "Nike" mentioned in two articles is one graph node
    with two MENTIONED_IN edges, not two duplicate entity nodes.
    """
    return " ".join(name.strip().lower().split())


class NetworkXGraph(KnowledgeGraph):
    """In-process knowledge graph backed by `networkx.MultiDiGraph`.

    Multi-edge so the same `(src, dst)` pair can carry multiple
    `EntityEdge`s of different kinds (or the same kind across
    snapshots). All public methods take the lock.
    """

    def __init__(self) -> None:
        # Lazy import — networkx is in the [social-context] extras.
        # When the extras aren't installed we fail loudly here rather
        # than at module import time so a `TRENDS_MODE=mock` deployment
        # never trips on it.
        import networkx as nx  # noqa: WPS433  (intentional lazy import)

        self._g: Any = nx.MultiDiGraph()
        self._lock = RLock()
        self._last_ingest_at: datetime | None = None

    # ----------------------------------------------------------- mutation

    def add_entity(self, entity: Entity, snapshot: SourceSnapshot) -> None:
        eid = _entity_id(entity.name)
        with self._lock:
            existing = self._g.nodes.get(eid)
            if existing is None:
                self._g.add_node(
                    eid,
                    name=entity.name,
                    type=entity.type,
                    salience=entity.salience,
                    first_seen=snapshot.ingested_at,
                    last_seen=snapshot.ingested_at,
                    mention_count=1,
                    platform_counts={snapshot.source: 1},
                )
            else:
                existing["last_seen"] = snapshot.ingested_at
                existing["mention_count"] = (
                    existing.get("mention_count", 0) + 1
                )
                existing["salience"] = max(
                    existing.get("salience", 0.0), entity.salience
                )
                pc = existing.setdefault("platform_counts", {})
                pc[snapshot.source] = pc.get(snapshot.source, 0) + 1
            # Snapshot bookkeeping — used by last_ingest_at().
            if (
                self._last_ingest_at is None
                or snapshot.ingested_at > self._last_ingest_at
            ):
                self._last_ingest_at = snapshot.ingested_at

    def add_edge(self, edge: EntityEdge) -> None:
        with self._lock:
            self._g.add_edge(
                _entity_id(edge.src),
                _entity_id(edge.dst),
                kind=edge.kind,
                weight=edge.weight,
                ts=edge.ts,
            )

    # -------------------------------------------------------------- query

    def query_entities_for_text(
        self, text: str, k: int = 10
    ) -> list[Entity]:
        """Best-effort lookup: substring match against canonical node ids.

        We don't use embeddings here yet (deferred to PR #6/#7) — this is
        the lexical fallback that mirrors `services/examples/library.py`'s
        approach, sufficient for v1.
        """
        if not text:
            return []
        needles = [w for w in text.lower().split() if len(w) >= 3]
        if not needles:
            return []
        with self._lock:
            matches: list[tuple[float, str, dict[str, Any]]] = []
            for nid, attrs in self._g.nodes(data=True):
                hit_count = sum(1 for n in needles if n in nid)
                if hit_count == 0:
                    continue
                # Score combines lexical hits + recency + raw mention count.
                score = (
                    hit_count * 1.0
                    + attrs.get("salience", 0.0) * 0.5
                    + min(attrs.get("mention_count", 0), 50) / 50.0
                )
                matches.append((score, nid, attrs))
            matches.sort(key=lambda x: x[0], reverse=True)
            return [
                _attrs_to_entity(nid, attrs)
                for _, nid, attrs in matches[:k]
            ]

    def neighbors(
        self, entity_id: str, depth: int = 1
    ) -> list[Entity]:
        eid = _entity_id(entity_id)
        with self._lock:
            if eid not in self._g:
                return []
            visited = {eid}
            frontier = {eid}
            for _ in range(max(depth, 1)):
                next_frontier: set[str] = set()
                for node in frontier:
                    for nb in self._g.successors(node):
                        if nb not in visited:
                            visited.add(nb)
                            next_frontier.add(nb)
                    for nb in self._g.predecessors(node):
                        if nb not in visited:
                            visited.add(nb)
                            next_frontier.add(nb)
                frontier = next_frontier
                if not frontier:
                    break
            visited.discard(eid)
            return [
                _attrs_to_entity(nid, self._g.nodes[nid])
                for nid in visited
            ]

    # -------------------------------------------------------- maintenance

    def prune_older_than(self, cutoff: datetime) -> int:
        """Delete every node whose `last_seen` is older than `cutoff`,
        and any edge attached to a deleted node. Returns count removed.
        """
        with self._lock:
            stale: list[str] = []
            for nid, attrs in self._g.nodes(data=True):
                last_seen = attrs.get("last_seen")
                if (
                    isinstance(last_seen, datetime)
                    and last_seen < cutoff
                ):
                    stale.append(nid)
            for nid in stale:
                self._g.remove_node(nid)
            return len(stale)

    def node_count(self) -> int:
        with self._lock:
            return self._g.number_of_nodes()

    def last_ingest_at(self) -> datetime | None:
        with self._lock:
            return self._last_ingest_at

    def healthcheck(self) -> bool:
        """In-memory graph is healthy as long as the lock and the
        underlying object are responsive. Always True for NetworkX —
        the meaningful health check is `last_ingest_at()` plus the
        scheduler's per-source counters.
        """
        with self._lock:
            try:
                _ = self._g.number_of_nodes()
                return True
            except Exception:  # noqa: BLE001
                _log.exception("networkx graph healthcheck failed")
                return False


# ---------------------------------------------------------------------------
# Neo4j backend
# ---------------------------------------------------------------------------


class Neo4jGraph(KnowledgeGraph):
    """Neo4j AuraDB-backed knowledge graph.

    Wraps the official `neo4j` Python driver (Bolt protocol). Same
    surface as `NetworkXGraph` so swapping backends is a one-line
    factory change. Schema is a single node label (`Entity`) with
    typed relationships per `EdgeKind`.

    Required env vars (resolved at construction time):
      * `NEO4J_URI`        — `neo4j+s://<host>:7687` for AuraDB
      * `NEO4J_USER`       — typically `neo4j`
      * `NEO4J_PASSWORD`   — Aura instance password
      * `NEO4J_DATABASE`   — optional, defaults to `neo4j`

    The driver is constructed eagerly so a misconfigured deployment
    fails at boot rather than on the first scrape pass. The client
    layer's healthcheck path catches connection drops at runtime and
    triggers the mock fallback.
    """

    def __init__(
        self,
        *,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ) -> None:
        # Lazy import — keeps the driver out of the dev install path.
        try:
            from neo4j import GraphDatabase  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover - explicit error
            raise RuntimeError(
                "Neo4jGraph requires the [social-context-neo4j] extras. "
                "Install via `uv sync --extra social-context-neo4j`."
            ) from exc

        import os

        self._uri = uri or os.environ.get("NEO4J_URI") or ""
        self._user = user or os.environ.get("NEO4J_USER", "neo4j")
        self._password = password or os.environ.get("NEO4J_PASSWORD") or ""
        self._database = (
            database or os.environ.get("NEO4J_DATABASE", "neo4j")
        )
        if not (self._uri and self._password):
            raise RuntimeError(
                "Neo4jGraph requires NEO4J_URI and NEO4J_PASSWORD env vars."
            )
        self._driver = GraphDatabase.driver(
            self._uri, auth=(self._user, self._password)
        )
        self._lock = RLock()
        self._last_ingest_at: datetime | None = None

        # Best-effort schema bootstrap. Idempotent — safe to call on
        # every cold start.
        self._ensure_constraints()

    # --------------------------------------------------------- mutation

    def add_entity(self, entity: Entity, snapshot: SourceSnapshot) -> None:
        eid = _entity_id(entity.name)
        ts = snapshot.ingested_at
        with self._lock:
            with self._driver.session(database=self._database) as sess:
                sess.run(
                    """
                    MERGE (e:Entity {id: $id})
                    ON CREATE SET
                        e.name = $name,
                        e.type = $type,
                        e.salience = $salience,
                        e.first_seen = $ts,
                        e.last_seen = $ts,
                        e.mention_count = 1,
                        e.platform_counts = {}
                    ON MATCH SET
                        e.last_seen = $ts,
                        e.mention_count = coalesce(e.mention_count, 0) + 1,
                        e.salience = CASE
                            WHEN $salience > coalesce(e.salience, 0.0)
                            THEN $salience ELSE e.salience END
                    WITH e
                    SET e.platform_counts =
                        apoc.map.merge(
                            coalesce(e.platform_counts, {}),
                            {`%(src)s`:
                                coalesce(
                                    e.platform_counts.`%(src)s`, 0) + 1}
                        )
                    """ % {"src": snapshot.source},
                    id=eid,
                    name=entity.name,
                    type=entity.type,
                    salience=float(entity.salience),
                    ts=ts.isoformat(),
                )
            if (
                self._last_ingest_at is None
                or ts > self._last_ingest_at
            ):
                self._last_ingest_at = ts

    def add_edge(self, edge: EntityEdge) -> None:
        with self._lock, self._driver.session(database=self._database) as sess:
            sess.run(
                """
                MATCH (a:Entity {id: $src}), (b:Entity {id: $dst})
                MERGE (a)-[r:%(kind)s {ts: $ts}]->(b)
                ON CREATE SET r.weight = $weight
                ON MATCH SET r.weight = $weight
                """ % {"kind": edge.kind},
                src=_entity_id(edge.src),
                dst=_entity_id(edge.dst),
                weight=float(edge.weight),
                ts=edge.ts.isoformat(),
            )

    # ------------------------------------------------------------- query

    def query_entities_for_text(
        self, text: str, k: int = 10
    ) -> list[Entity]:
        if not text:
            return []
        needles = [w for w in text.lower().split() if len(w) >= 3]
        if not needles:
            return []
        with self._lock, self._driver.session(database=self._database) as sess:
            # Lexical OR-match on lowercased id substring. AuraDB has
            # full-text indexes available; for v1 we keep parity with
            # the NetworkX backend's substring scoring.
            result = sess.run(
                """
                MATCH (e:Entity)
                WHERE ANY(n IN $needles WHERE e.id CONTAINS n)
                RETURN e.id AS id, properties(e) AS attrs
                LIMIT $k
                """,
                needles=needles,
                k=k * 2,  # over-fetch; we re-rank in Python.
            )
            rows = [(rec["id"], dict(rec["attrs"])) for rec in result]
        # Re-rank using the same formula as NetworkXGraph for parity.
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for nid, attrs in rows:
            hit_count = sum(1 for n in needles if n in nid)
            score = (
                hit_count * 1.0
                + attrs.get("salience", 0.0) * 0.5
                + min(attrs.get("mention_count", 0), 50) / 50.0
            )
            scored.append((score, nid, attrs))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [_attrs_to_entity(nid, attrs) for _, nid, attrs in scored[:k]]

    def neighbors(
        self, entity_id: str, depth: int = 1
    ) -> list[Entity]:
        eid = _entity_id(entity_id)
        max_depth = max(depth, 1)
        with self._lock, self._driver.session(database=self._database) as sess:
            result = sess.run(
                f"""
                MATCH (a:Entity {{id: $id}})
                MATCH (a)-[*1..{max_depth}]-(b:Entity)
                WHERE b.id <> $id
                RETURN DISTINCT b.id AS id, properties(b) AS attrs
                """,
                id=eid,
            )
            return [
                _attrs_to_entity(rec["id"], dict(rec["attrs"]))
                for rec in result
            ]

    # -------------------------------------------------------- maintenance

    def prune_older_than(self, cutoff: datetime) -> int:
        cutoff_iso = cutoff.isoformat()
        with self._lock, self._driver.session(database=self._database) as sess:
            result = sess.run(
                """
                MATCH (e:Entity)
                WHERE e.last_seen < $cutoff
                WITH collect(e) AS stale
                CALL {
                    WITH stale
                    UNWIND stale AS s
                    DETACH DELETE s
                    RETURN count(*) AS removed
                }
                RETURN removed
                """,
                cutoff=cutoff_iso,
            )
            rec = result.single()
            return int(rec["removed"]) if rec else 0

    def node_count(self) -> int:
        with self._lock, self._driver.session(database=self._database) as sess:
            rec = sess.run("MATCH (e:Entity) RETURN count(e) AS n").single()
            return int(rec["n"]) if rec else 0

    def last_ingest_at(self) -> datetime | None:
        with self._lock:
            return self._last_ingest_at

    def healthcheck(self) -> bool:
        try:
            with self._driver.session(database=self._database) as sess:
                sess.run("RETURN 1").single()
            return True
        except Exception:  # noqa: BLE001
            _log.exception("Neo4j healthcheck failed")
            return False

    # ------------------------------------------------------------- close

    def close(self) -> None:
        try:
            self._driver.close()
        except Exception:  # noqa: BLE001
            _log.exception("Neo4j driver close raised; ignoring")

    # --------------------------------------------------------- internals

    def _ensure_constraints(self) -> None:
        try:
            with self._driver.session(database=self._database) as sess:
                sess.run(
                    "CREATE CONSTRAINT entity_id IF NOT EXISTS "
                    "FOR (e:Entity) REQUIRE e.id IS UNIQUE"
                )
        except Exception:  # noqa: BLE001
            _log.exception(
                "Neo4j constraint bootstrap failed; continuing"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _attrs_to_entity(node_id: str, attrs: dict[str, Any]) -> Entity:
    """Project a graph node back into an `Entity` for the query layer.

    `trend_velocity`, `sentiment_polarity`, `sarcasm_flag`, and
    `platform_peaks` are derived in `query.py` from edge data — the
    graph itself only stores raw mention counts and snapshot pointers.
    """
    platform_counts = attrs.get("platform_counts", {}) or {}
    total = sum(platform_counts.values()) or 1
    platform_peaks = {
        src: count / total for src, count in platform_counts.items()
    }
    return Entity(
        name=attrs.get("name") or node_id,
        type=attrs.get("type", "TOPIC"),
        salience=attrs.get("salience", 0.0),
        trend_velocity=0.0,  # filled in by query.py
        sentiment_polarity=0.0,
        sarcasm_flag=False,
        platform_peaks=platform_peaks,
    )
