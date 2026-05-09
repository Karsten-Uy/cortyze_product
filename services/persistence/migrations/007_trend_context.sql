-- Phase 2 GraphRAG support tables.
--
-- The `trend_context` table itself was created in 002_runs_v2.sql with
-- a generic `payload jsonb` column, so we don't need to alter it — the
-- richer TrendContextV2 payload (entities, dominant_topic, brand_risk_score,
-- cultural_moment, snapshot_timestamp, fallback_reason) deserializes into
-- the same column. We add an index for the audit-trail / fallback-rate
-- queries the social_context health endpoint will run.
--
-- The two new tables back the rolling 48-hour knowledge graph and the
-- (deferred) embedding cache that `query.py` may activate later.

create index if not exists trend_context_fetched_idx
  on trend_context (fetched_at desc);

-- Per-source raw snapshots. The scheduler writes a row per ingest;
-- entities/edges are derived into the in-memory NetworkX graph (and,
-- in prod, mirrored to Neo4j). `ttl_at` is set on insert to
-- `ingested_at + GRAPH_TTL_HOURS`; the prune job deletes rows past it.
-- `(source, source_id)` is unique so re-fetching the same Reddit post
-- or news article doesn't double-count.
create table if not exists trend_snapshots (
  snapshot_id   uuid primary key default gen_random_uuid(),
  source        text not null,                   -- 'reddit' | 'news' | 'trends' | 'x'
  source_id     text not null,                   -- platform's stable id (post id, url hash, etc.)
  payload       jsonb not null,                  -- raw snapshot, post-NER + sentiment
  ingested_at   timestamptz not null default now(),
  ttl_at        timestamptz not null,
  unique (source, source_id)
);
create index if not exists trend_snapshots_ttl_idx
  on trend_snapshots (ttl_at);
create index if not exists trend_snapshots_source_idx
  on trend_snapshots (source, ingested_at desc);

-- Content-addressed embedding cache. Reserved for the deferred
-- similarity step in services/social_context/query.py — the column
-- shape is fixed now so we don't need a follow-up migration when we
-- activate it. `cache_key = sha256(model + ':' + text)` per
-- the cache.py implementation.
create table if not exists embedding_cache (
  cache_key    text primary key,
  model        text not null,
  vector       bytea not null,
  created_at   timestamptz not null default now()
);
create index if not exists embedding_cache_created_idx
  on embedding_cache (created_at);
