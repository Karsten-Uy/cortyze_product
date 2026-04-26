# cortyze_product

Stage 1 backend for [BrainScore](IMPLEMENTATION_PLAN.md) — a JSON API that turns content (video/image/text) into 8 brain-region scores plus a goal-weighted overall score.

## Quick start

```bash
uv sync
cp .env.example .env       # fill in tokens you have; the rest is optional
uv run pytest              # 48+ tests pass on a fresh checkout
uv run uvicorn api.main:app --reload
```

Then:

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/analyze \
  -H 'content-type: application/json' \
  -d '{"content_url":"https://example.com/x.mp4","content_type":"video","goal":"engagement"}' \
  | python -m json.tool
```

Auto-generated OpenAPI Swagger UI: <http://localhost:8000/docs>.

## Repo layout

- [`core/`](core/) — pure-Python library (atlas mapping, scoring, schemas). No I/O. Importable from anywhere.
- [`api/`](api/) — FastAPI app (`api.main:app`), routes, RunPod client.
- [`services/`](services/) — bounded contexts: `storage/r2.py` (Cloudflare R2), `persistence/reports.py` (Supabase Postgres). Stage 2 adds `suggestions/` and `examples/`.
- [`scripts/`](scripts/) — one-shot dev tools: `build_atlas_labels.py` (fsaverage5 DK label generator), `build_fixture.py` (TRIBE v2 on Mac CPU → real fixture).
- [`tests/`](tests/) — pytest suites (`test_goals.py`, `test_scoring.py`, `test_atlas.py`, `test_api.py`).

## Modes

The API auto-degrades based on env vars — every dependency is optional:

| Env var present | Behavior |
|---|---|
| (none required) | Mock mode — `MockRunPodClient` returns synthetic `(T, 20484)`; `/upload-url` and `/report/{id}` return 501 |
| `R2_*` | `/upload-url` returns presigned URLs; `/analyze` persists predictions to R2 |
| `DATABASE_URL` | `/analyze` writes a row to `reports`; `/report/{id}` returns saved reports |
| `RUNPOD_ENDPOINT_ID` | Real GPU inference (Stage 1.2 — not yet implemented; raises `NotImplementedError`) |
| `HF_TOKEN` | `scripts/build_fixture.py` can run TRIBE v2 on the Mac to generate a real fixture |

## Account setup (to leave mock mode)

See [IMPLEMENTATION_PLAN.md §11](IMPLEMENTATION_PLAN.md) for the full Stage 3 prerequisite list. Stage 1 needs three:

### 1. Hugging Face — `HF_TOKEN`

Required for `scripts/build_fixture.py` (the one-time real-data fixture run).

1. Sign up at <https://huggingface.co>
2. Settings → Access Tokens → "Create new token" with **Read** scope
3. Visit <https://huggingface.co/facebook/tribev2> and click **Agree and access repository** to accept gated terms
4. Add `HF_TOKEN=hf_...` to `.env`

### 2. Cloudflare R2 — object storage

Required for `/upload-url` and prediction persistence (Stage 5 training data).

1. Sign up at <https://dash.cloudflare.com> (free)
2. Add a payment method — R2 needs one even on free tier (10 GB free; you'll be at $0)
3. **R2 → Create bucket** twice: `cortyze-uploads` and `cortyze-predictions`
4. **R2 → Manage R2 API Tokens → Create API token**, scope: *Object Read & Write*. Note the Account ID, Access Key, and Secret Key.
5. (Recommended) On `cortyze-uploads`, add a CORS rule allowing `PUT` from your frontend origin and a 7-day lifecycle delete rule.
6. Add to `.env`:
   ```
   R2_ACCOUNT_ID=...
   R2_ACCESS_KEY=...
   R2_SECRET_KEY=...
   R2_BUCKET_UPLOADS=cortyze-uploads
   R2_BUCKET_PREDICTIONS=cortyze-predictions
   ```

### 3. Supabase — Postgres for the `reports` table

Required for `/report/{request_id}` and the Stage 4 audience-profile join key.

1. Sign up at <https://supabase.com> (free)
2. **New project**, set a strong DB password, wait ~2 min for provisioning
3. **Project Settings → Database → Connection string → URI** — copy. Add to `.env` as:
   ```
   DATABASE_URL=postgresql://postgres:<password>@db.<ref>.supabase.co:5432/postgres
   ```
4. **SQL Editor → New query** → paste contents of [`services/persistence/migrations/001_reports.sql`](services/persistence/migrations/001_reports.sql) → Run

After all three are set, restart the API. `/upload-url` and `/report/{id}` will start returning real data; `/analyze` will persist every prediction to both R2 and Postgres.

### Later — RunPod (Stage 1.2)

Deferred until the rest of Stage 1 is solid. Sign up at <https://runpod.io>, add ~$20 credit, generate an API key. Wiring lands in `gpu_worker/` per [IMPLEMENTATION_PLAN.md §4](IMPLEMENTATION_PLAN.md).

## Running tests

```bash
uv run pytest -v                 # all suites
uv run pytest tests/test_api.py  # API smoke tests only
```

## Generating the real fixture (optional, one-time)

Replace the synthetic mock-mode predictions with real TRIBE v2 output (10–20 min on Mac CPU, ~5 GB RAM):

```bash
export HF_TOKEN=hf_...
/Users/kirby/Documents/cortyze/tribev2/.venv/bin/python scripts/build_fixture.py
```

The mock client picks up any `tests/fixtures/golden_pred_*.npy` automatically. See [`tests/fixtures/README.md`](tests/fixtures/README.md) for details.
