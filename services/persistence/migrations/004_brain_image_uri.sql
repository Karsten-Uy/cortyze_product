-- Stage 3 polish. Persist a pointer to the rendered brain heatmap PNG
-- in R2 so past runs (and regoals) can re-render the image without
-- having to re-run the visualization pipeline.
--
-- The image itself lives at r2://<predictions-bucket>/brain_images/<request_id>.png;
-- only the storage-key gets persisted here — the API mints a fresh
-- presigned GET URL per request.

alter table reports
    add column if not exists brain_image_uri text null;
