-- Stage 3 follow-up. Cache the weighted overall under EACH of the four
-- goals at analyze-time so the frontend can re-weight without hitting
-- the backend. Costs ~80 bytes per row.
--
-- Apply via Supabase SQL Editor or `psql "$DATABASE_URL" -f 003_overall_by_goal.sql`.

alter table reports
    add column if not exists overall_by_goal jsonb null;

-- Also persist the post-input scalars so the regoal endpoint can
-- correctly call diagnose() with the same context as the original run
-- (without these, regoal would lose audio-presence and carousel-shape
-- info that the suggestion engine reads).
alter table reports
    add column if not exists audio_url text null,
    add column if not exists image_count integer null,
    add column if not exists seconds_per_image real null;
