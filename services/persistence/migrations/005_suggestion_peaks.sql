-- Add per-suggestion peak window columns. The frontend reads these to
-- render a small clip player inside each expanded suggestion card,
-- looping the section of the user's video where that region peaks.
-- Both nullable: image runs and pre-feature DB-cached suggestions
-- legitimately have no peak window.

alter table suggestions
  add column if not exists peak_start_s numeric(7, 2),
  add column if not exists peak_end_s numeric(7, 2);
