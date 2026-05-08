-- Add the `examples_json` column to suggestions so the library example
-- slugs picked by services/synthesis at synthesis-time survive the
-- write-then-read round trip. Without this column the persistence
-- layer silently dropped Suggestion.examples on insert, and the
-- frontend always saw the Pydantic default of [].
--
-- The column is nullable jsonb to keep old rows readable; the read path
-- coalesces NULL → [].

alter table suggestions
  add column if not exists examples_json jsonb;
