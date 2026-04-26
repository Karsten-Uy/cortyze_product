-- Stage 1 reports table per IMPLEMENTATION_PLAN.md §5.2.
-- Apply once via the Supabase SQL editor or `psql "$DATABASE_URL" -f 001_reports.sql`.

create table if not exists reports (
    request_id uuid primary key,
    user_id text null,
    goal text not null,
    region_scores jsonb not null,
    overall_score real not null,
    model_version text not null,
    raw_predictions_uri text null,
    created_at timestamptz not null default now()
);

create index if not exists idx_reports_user_id on reports (user_id);
create index if not exists idx_reports_created_at on reports (created_at desc);
