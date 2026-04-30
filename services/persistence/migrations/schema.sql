-- Cortyze — full database schema (idempotent).
-- Run once in the Supabase SQL editor before first deploy, or any time
-- thereafter (safe on existing databases — all statements are no-ops when
-- the object already exists, except the backfill UPDATE at the bottom which
-- only touches rows that need it).

create extension if not exists pgcrypto;

-- ---------------------------------------------------------------------------
-- ad_campaigns
-- ---------------------------------------------------------------------------

create table if not exists ad_campaigns (
    id          uuid        primary key default gen_random_uuid(),
    user_id     uuid        not null,
    name        text        not null,
    description text        null,
    created_at  timestamptz not null default now()
);

create index if not exists idx_campaigns_user
    on ad_campaigns (user_id);
create index if not exists idx_campaigns_user_created
    on ad_campaigns (user_id, created_at desc);

-- ---------------------------------------------------------------------------
-- reports  (create with all columns for fresh databases)
-- ---------------------------------------------------------------------------

create table if not exists reports (
    -- core (migration 001)
    request_id          uuid        primary key,
    user_id             text        null,
    goal                text        not null,
    region_scores       jsonb       not null,
    overall_score       real        not null,
    model_version       text        not null,
    raw_predictions_uri text        null,
    created_at          timestamptz not null default now(),

    -- campaigns & context (migration 002)
    campaign_id         uuid        references ad_campaigns(id) on delete set null,
    additional_context  text        null,
    caption_text        text        null,
    thumbnail_url       text        null,
    title               text        null,
    content_type        text        null,

    -- scoring cache (migration 003)
    overall_by_goal     jsonb       null,
    audio_url           text        null,
    image_count         integer     null,
    seconds_per_image   real        null,

    -- brain image pointer (migrations 004 + 006)
    brain_image_uri         text    null,
    brain_image_request_id  uuid    null,

    -- persisted outputs (migration 005)
    suggestions         jsonb       null,
    moments             jsonb       null,
    region_timeseries   jsonb       null
);

-- Ensure all columns exist on already-migrated databases. Each statement
-- is a no-op if the column is already present.
alter table reports add column if not exists campaign_id        uuid references ad_campaigns(id) on delete set null;
alter table reports add column if not exists additional_context text null;
alter table reports add column if not exists caption_text       text null;
alter table reports add column if not exists thumbnail_url      text null;
alter table reports add column if not exists title              text null;
alter table reports add column if not exists content_type       text null;
alter table reports add column if not exists overall_by_goal    jsonb null;
alter table reports add column if not exists audio_url          text null;
alter table reports add column if not exists image_count        integer null;
alter table reports add column if not exists seconds_per_image  real null;
alter table reports add column if not exists brain_image_uri    text null;
alter table reports add column if not exists brain_image_request_id uuid null;
alter table reports add column if not exists suggestions        jsonb null;
alter table reports add column if not exists moments            jsonb null;
alter table reports add column if not exists region_timeseries  jsonb null;

create index if not exists idx_reports_user_id
    on reports (user_id);
create index if not exists idx_reports_created_at
    on reports (created_at desc);
create index if not exists idx_reports_campaign
    on reports (campaign_id);
create index if not exists idx_reports_user_created
    on reports (user_id, created_at desc);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------

alter table reports      enable row level security;
alter table ad_campaigns enable row level security;

-- Drop before recreate so re-running this file is safe.
drop policy if exists "users see own reports"              on reports;
drop policy if exists "users see own campaigns"            on ad_campaigns;
drop policy if exists "service role full access reports"   on reports;
drop policy if exists "service role full access campaigns" on ad_campaigns;

-- Users can only read/write their own rows (anon key path).
create policy "users see own reports" on reports
    for all
    using      (user_id = auth.uid()::text)
    with check (user_id = auth.uid()::text);

create policy "users see own campaigns" on ad_campaigns
    for all
    using      (user_id = auth.uid())
    with check (user_id = auth.uid());

-- The API server uses the service-role key for writes during /analyze so
-- it can bypass RLS (it owns the user_id enforcement itself via JWT sub).
create policy "service role full access reports" on reports
    for all to service_role using (true) with check (true);

create policy "service role full access campaigns" on ad_campaigns
    for all to service_role using (true) with check (true);

-- ---------------------------------------------------------------------------
-- Backfill: existing rows whose image lives under their own request_id
-- (safe to re-run — only touches rows that still need it)
-- ---------------------------------------------------------------------------

update reports
   set brain_image_request_id = request_id
 where brain_image_request_id is null
   and brain_image_uri is not null;
