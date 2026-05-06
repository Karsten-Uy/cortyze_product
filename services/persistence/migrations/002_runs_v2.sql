-- v2 (`/runs`) pipeline tables. Companion to the v1 reports/campaigns
-- tables in 001_reports.sql; these are independent — the legacy
-- `/analyze` flow and the new `/runs` flow share no data.

-- One row per user-initiated analysis. Holds inputs, status, timestamps.
create table if not exists runs (
  id            text primary key,
  user_id       uuid,
  name          text not null,
  goal          text not null,             -- snake_case GoalKey
  brief         text not null default '',
  caption       text not null default '',
  media_url     text,
  kind          text not null default 'Video',  -- 'Video' | 'Image'
  status        text not null default 'queued',
  created_at    timestamptz not null default now(),
  completed_at  timestamptz,
  error         text
);

create index if not exists runs_user_created_idx
  on runs (user_id, created_at desc);

create index if not exists runs_status_idx
  on runs (status);

-- Six rows per run (one per v2 region).
create table if not exists region_scores (
  run_id      text not null references runs(id) on delete cascade,
  region_key  text not null,               -- 'memory' | 'emotion' | ...
  score       numeric(5,2) not null,       -- 0..100
  benchmark   numeric(5,2) not null,       -- 0..100
  primary key (run_id, region_key)
);

-- One row per run, written after region_scores land.
create table if not exists composites (
  run_id      text primary key references runs(id) on delete cascade,
  score       numeric(5,2) not null,       -- 0..100
  benchmark   numeric(5,2) not null,
  delta       numeric(6,2) not null,       -- signed
  status      text not null                -- 'Needs work' | 'Solid' | ...
);

-- Phase 2 snapshot. Kept for audit + Phase 3 prompt provenance.
create table if not exists trend_context (
  run_id      text primary key references runs(id) on delete cascade,
  payload     jsonb not null,
  fetched_at  timestamptz not null default now()
);

-- Suggestions list. `lift` is updated in-place by Phase 4.
create table if not exists suggestions (
  run_id          text not null references runs(id) on delete cascade,
  ord             integer not null,        -- 1-based rank, stable within a run
  priority        text not null,           -- 'critical' | 'high' | 'medium'
  title           text not null,
  area            text not null,           -- v2 RegionKey
  lift            numeric(5,2) not null,
  explanation     text not null,
  reference_json  jsonb,                   -- nullable; matches Reference shape
  primary key (run_id, ord)
);

-- Sidebar projection. Materialized so the list endpoint stays cheap
-- even at 100k+ runs per user.
create materialized view if not exists past_runs_view as
  select
    r.id              as id,
    r.user_id         as user_id,
    r.name            as name,
    r.kind            as kind,
    coalesce(c.score, 0)::numeric(5,2) as score,
    r.created_at      as created_at
  from runs r
  left join composites c on c.run_id = r.id
  where r.status = 'complete';

-- Refresh policy is a deployment concern — see Stage-3 ops notes. A
-- trigger-driven refresh on `composites.insert` is the cheap default;
-- a periodic full refresh (every 60s) is the lazier alternative.
create unique index if not exists past_runs_view_id_idx on past_runs_view (id);
create index if not exists past_runs_view_user_idx
  on past_runs_view (user_id, created_at desc);
