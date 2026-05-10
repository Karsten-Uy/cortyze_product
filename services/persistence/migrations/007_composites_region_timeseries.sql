-- 007_composites_region_timeseries.sql
--
-- Adds the per-region 1Hz activation timeseries to the composites
-- row. Without this column, `_PostgresRunStore._write_result` had
-- nowhere to persist `SuggestionPlan.region_timeseries`, so demo runs
-- on prod (which go through Postgres, not the in-memory store) came
-- back from `GET /runs/{id}` with `region_timeseries: null`. The
-- frontend then fell back to the static-bar layout and the combined
-- chart never rendered.
--
-- jsonb (not json) so the column reads back as a parsed Python dict
-- via psycopg's default jsonb codec.

alter table composites
  add column if not exists region_timeseries jsonb null;
