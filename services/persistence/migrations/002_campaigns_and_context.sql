-- Stage 3 account-aware migration. Adds ad_campaigns table + extra columns
-- on reports for sidebar grouping, additional context, and lightweight
-- summary projection.
--
-- Apply via the Supabase SQL editor or `psql "$DATABASE_URL" -f 002_campaigns_and_context.sql`.

-- gen_random_uuid() lives in pgcrypto on older Postgres; on Supabase it's
-- available by default. Make the extension idempotent so this script is
-- safe to run on a fresh database.
create extension if not exists pgcrypto;

create table if not exists ad_campaigns (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null,
    name text not null,
    description text null,
    created_at timestamptz not null default now()
);
create index if not exists idx_campaigns_user on ad_campaigns (user_id);
create index if not exists idx_campaigns_user_created on ad_campaigns (user_id, created_at desc);

-- Extend reports with the columns the sidebar / suggestion engine need.
-- All columns nullable so existing rows from migration 001 stay valid.
alter table reports
    add column if not exists campaign_id uuid references ad_campaigns(id) on delete set null,
    add column if not exists additional_context text null,
    add column if not exists caption_text text null,
    add column if not exists thumbnail_url text null,
    add column if not exists title text null,
    add column if not exists content_type text null;

create index if not exists idx_reports_campaign on reports (campaign_id);
create index if not exists idx_reports_user_created on reports (user_id, created_at desc);

-- Row-Level Security so users only see their own data. The Supabase
-- JWT carries auth.uid() (the user's UUID), which we compare against
-- the report's user_id. NB: reports.user_id is `text` per migration 001
-- to allow non-Supabase identities; we cast for the policy.
alter table reports enable row level security;
alter table ad_campaigns enable row level security;

-- Drop old policies if they exist (no-op on first apply) so re-running
-- the migration doesn't fail with "policy already exists".
drop policy if exists "users see own reports" on reports;
drop policy if exists "users see own campaigns" on ad_campaigns;
drop policy if exists "service role full access reports" on reports;
drop policy if exists "service role full access campaigns" on ad_campaigns;

create policy "users see own reports" on reports
    for all using (user_id = auth.uid()::text)
    with check (user_id = auth.uid()::text);

create policy "users see own campaigns" on ad_campaigns
    for all using (user_id = auth.uid())
    with check (user_id = auth.uid());

-- Service role bypass: the API server uses the service-role key for
-- writes (it can't go through RLS without a user JWT for inserts during
-- /analyze). RLS is enforced for the anon key only.
create policy "service role full access reports" on reports
    for all to service_role using (true) with check (true);
create policy "service role full access campaigns" on ad_campaigns
    for all to service_role using (true) with check (true);
