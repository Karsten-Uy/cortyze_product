-- Persist the R2 object key for the uploaded clip alongside the
-- presigned URL. The presigned `media_url` expires after ~1 hour; the
-- underlying object lives 7 days under the bucket lifecycle rule. With
-- the key on hand, GET /runs/:id can re-presign a fresh `media_url` on
-- every report load instead of handing the user a dead link.
--
-- Nullable: pre-feature uploads and runs without media_url have no key.

alter table runs
  add column if not exists media_object_key text;
