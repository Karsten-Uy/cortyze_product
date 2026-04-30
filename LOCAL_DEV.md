# Local Development

How to run the full Cortyze stack on your Mac. The system is designed so you can iterate on **everything except real TRIBE inference** locally — UI, prompts, suggestion engine, gallery flow, brain visualization, persistence, even real Claude vision-grounded suggestions all work fine on an M3 8GB.

## TL;DR

Three terminals, one-time setup done:

```bash
# Terminal 1 — local object storage
cd cortyze_product
./scripts/dev_minio.sh start

# Terminal 2 — backend API (FastAPI)
make api-run

# Terminal 3 — frontend (Next.js)
cd ../cortyze_frontend
npm run dev
```

Open <http://localhost:3000>. Mock inference returns instantly; suggestions can be either templated (free) or real Claude (cents per call).

## What runs locally

| Component | Status | Notes |
|---|---|---|
| Backend FastAPI | ✓ | <100 MB RAM idle |
| Frontend (Next.js dev) | ✓ | <200 MB RAM |
| MinIO local storage | ✓ | <100 MB |
| Mock RunPod client | ✓ | Returns saved sintel fixture instantly |
| Mock LLM | ✓ | Templated, deterministic, free |
| Real Claude / OpenAI suggestion LLM | ✓ | Just an HTTPS call from your Mac |
| Real Anthropic with image attached (vision) | ✓ | When implemented — sends image bytes inline |
| Real TRIBE inference on M3 CPU | ⚠ | Works but ~2 hr per video. Use `scripts/build_fixture.py` for one-off fixture generation. |

For real GPU inference at usable speeds, deploy to RunPod ([RUNPOD_SESSION.md](RUNPOD_SESSION.md)) or use the mock fixture path during dev.

---

## First-time setup (~15 minutes)

### Prerequisites

```bash
# Python 3.11+ via uv (fast, deterministic dependency resolution)
brew install uv

# Node 20+ for the frontend
brew install node

# MinIO + mc for local S3-compatible storage
brew install minio/stable/minio minio/stable/mc

# OrbStack only if you'll be building Docker images for RunPod
# (not needed for daily local dev)
brew install --cask orbstack
```

### Backend deps

```bash
cd cortyze_product
uv sync                      # installs everything in pyproject.toml
uv run pytest                # 113 tests should pass
```

### Frontend deps

```bash
cd ../cortyze_frontend
npm install
```

### `.env` setup

Copy the example and fill in what you have:

```bash
cd cortyze_product
cp .env.example .env
```

Minimum to run mock-everything mode (no API keys needed):

```env
INFERENCE_MODE=mock
STORAGE_MODE=minio
S3_ENDPOINT_URL=http://localhost:9000
R2_ACCESS_KEY=minioadmin
R2_SECRET_KEY=minioadmin
R2_BUCKET_UPLOADS=cortyze-uploads
R2_BUCKET_PREDICTIONS=cortyze-predictions
ENABLE_SUGGESTIONS=true
SUGGESTION_LLM_MODE=mock
FRONTEND_ORIGINS=http://localhost:3000
```

To use **real Claude for suggestions** (~$0.001 per region triggered, recommended for prompt iteration):

```env
SUGGESTION_LLM_MODE=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-6
```

To **persist reports to Postgres** locally (optional — without it, `/report/{id}` returns 501 but `/analyze` still works):

Either set up a local Postgres and run [services/persistence/migrations/001_reports.sql](services/persistence/migrations/001_reports.sql), or use Supabase's free tier and put its connection string in `DATABASE_URL`.

### Toggle reference

The three independent toggles that decide what's real and what's mocked. Mix and match:

| Env var | Values | What it controls |
|---|---|---|
| **`INFERENCE_MODE`** | `mock` (default) · `runpod` | Where brain scores come from. `mock` = saved `.npy` fixture in `tests/fixtures/` (instant, free). `runpod` = real GPU via Pod or Serverless (~5–15 min, ~$0.02–0.10 per call). |
| **`SUGGESTION_LLM_MODE`** | `mock` (default) · `anthropic` · `openai_compatible` | What generates suggestion text. `mock` = templated, free, deterministic. `anthropic` / `openai_compatible` = real LLM, content-type-aware prompts, ~$0.001 per region triggered. |
| **`STORAGE_MODE`** | `off` · `minio` (recommended local) · `r2` | Where uploads + persisted predictions live. `off` = uploads disabled. `minio` = local at `localhost:9000`. `r2` = real Cloudflare R2. |

The toggles are independent. The most useful local combo is **`INFERENCE_MODE=mock` + `SUGGESTION_LLM_MODE=anthropic` + `STORAGE_MODE=minio`** — real upload UX, real Claude suggestions, instant brain scores from a saved fixture. See "Running TRIBE v2 locally" below for how to make the saved fixture come from your own content instead of sintel.

---

## Day-to-day workflow

### Three-terminal startup

**Terminal 1 — MinIO**:
```bash
cd cortyze_product
./scripts/dev_minio.sh start
```
First run creates the two buckets (`cortyze-uploads`, `cortyze-predictions`). Console at <http://localhost:9001> with `minioadmin` / `minioadmin`.

**Terminal 2 — backend**:
```bash
make api-run
```
Watches for code changes and auto-reloads. Boots in ~1s. Health: <http://localhost:8000/health>.

On startup you'll see a feature-flag banner showing which paths are active:
```
[cortyze] startup feature flags:
  inference:      mock
  object storage: minio (local)
  persistence:    off
  suggestions:    enabled (mode=mock)
```

**Terminal 3 — frontend**:
```bash
cd ../cortyze_frontend
npm run dev
```
Opens at <http://localhost:3000>. Hot-reload on save.

### Two-tab UI

The frontend has two top-level tabs:

- **Video** — drag-drop an MP4 (or paste a URL), pick a goal, run.
- **Post** — drop 1-20 images, optional audio + caption, pick a goal, run.

In mock mode every analysis returns the saved sintel fixture, so the brain map and region scores look identical regardless of input. **What's real**:
- Your image / audio / caption uploads to MinIO and renders correctly in the report
- Suggestion engine sees the actual goal + region scores from the fixture
- If `SUGGESTION_LLM_MODE=anthropic`, real Claude generates suggestions; switching content shapes (1 image vs 5 vs video) changes the prompt and the suggestion language

This is a great loop for iterating on UI, prompts, and report rendering — fast, cheap, and the only stationary thing is the brain scores.

---

## What works without RunPod

Most product iteration:

- **UI for video / post tabs** — pure frontend, no GPU
- **Prompt quality for video / post / carousel** — real LLM with mock brain scores
- **Vision-grounded suggestions** (when implemented) — real Claude + your real images, mock brain scores
- **Suggestion card / region card / per-image bar rendering** — frontend only
- **Storage flow** — MinIO mirrors R2 behavior for upload presigned URLs
- **Tests** — 113 backend, plus `npx next build` typechecks the frontend
- **Brain visualization** — nilearn renders PNG locally

### What you actually need RunPod for

Three things, and they're all rare:

1. **End-to-end product validation** with real brain scores from a real upload
2. **Generating a new fixture** for mock mode (or just use the committed sintel fixture)
3. **Benchmarking latency / cost** — see [docs/runpod_benchmark.md](docs/runpod_benchmark.md)

---

## Running TRIBE v2 locally on M3

The full GPU pipeline doesn't run interactively on the M3 — V-JEPA2-Giant takes ~30-90 min per inference on CPU. The pragmatic workflow is **generate a fixture once, replay it via mock mode forever**.

There are two fixture builders:

| Script | Use when |
|---|---|
| `scripts/build_fixture.py` | You have a **video** clip. Produces `tests/fixtures/golden_pred_<name>_T<n>.npy`. ~1.5–2 hours on M3. |
| `scripts/build_post_fixture.py` | You have a **post** (1 image + optional audio + optional caption). ~30–90 min depending on whether audio is included. |

Both apply CPU + bf16 monkey-patches before importing TRIBE so the model fits in 8 GB RAM. Output `.npy` files land in `tests/fixtures/`; mock mode auto-picks the alphabetically-first one when `INFERENCE_MODE=mock`.

### One-time setup

```bash
brew install ffmpeg                                  # required by build_post_fixture.py for image→video assembly

# Hugging Face access:
# 1. Token at https://huggingface.co/settings/tokens (Read scope)
# 2. Accept gated terms at:
#    - https://huggingface.co/facebook/tribev2
#    - https://huggingface.co/meta-llama/Llama-3.2-3B
export HF_TOKEN=hf_...

# Verify the tribev2 venv is healthy (it has neuralset / transformers / torch installed):
ls /Users/kirby/Documents/cortyze/tribev2/.venv/bin/python    # exists ✓
/Users/kirby/Documents/cortyze/tribev2/.venv/bin/python \
  -c "import tribev2.demo_utils, neuralset; print('ok')"
```

If the import fails, re-install tribev2 in its venv:
```bash
cd /Users/kirby/Documents/cortyze/tribev2
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
deactivate
```

### Free up RAM before running

TRIBE loads ~5 GB of weights into RAM in bf16 mode. On 8 GB Mac:

- Quit OrbStack / Docker (their VM eats 4 GB by default)
- Quit Slack / heavy browser tabs
- Confirm headroom: `vm_stat | head -5` — want ≥ 4 GB free

### Path A — Generate a video fixture

```bash
cd /Users/kirby/Documents/cortyze/cortyze_product
export HF_TOKEN=hf_...

/Users/kirby/Documents/cortyze/tribev2/.venv/bin/python scripts/build_fixture.py \
  --video-url https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4 \
  --output-stem tests/fixtures/golden_pred_sintel
```

Wall time on M3: ~1.5–2 hours. Output: `tests/fixtures/golden_pred_sintel_T<N>.npy` plus a `.meta.json` sidecar.

### Path B — Generate a post fixture (image + optional audio + caption)

```bash
cd /Users/kirby/Documents/cortyze/cortyze_product
export HF_TOKEN=hf_...

# Caption-only post (~30-50 min)
/Users/kirby/Documents/cortyze/tribev2/.venv/bin/python scripts/build_post_fixture.py \
  --image-url https://your-cdn.com/your-image.jpg \
  --caption "Check out our new product, available now"

# Image + audio + caption (~50-90 min)
/Users/kirby/Documents/cortyze/tribev2/.venv/bin/python scripts/build_post_fixture.py \
  --image-url https://your-cdn.com/your-image.jpg \
  --audio-url https://your-cdn.com/voiceover.mp3 \
  --caption "Real voiceover and reading text combined"
```

Output: `tests/fixtures/golden_pred_post_T<N>.npy` plus sidecar.

### Picking which fixture mock mode returns

`MockRunPodClient` picks the **first** fixture matching `golden_pred_*.npy` by sorted order. To force mock to return a specific one:

```bash
mkdir -p tests/fixtures/.archive

# Hide everything except the one you want active
mv tests/fixtures/golden_pred_*.npy tests/fixtures/.archive/
mv tests/fixtures/golden_pred_*.meta.json tests/fixtures/.archive/

# Restore just one
mv tests/fixtures/.archive/golden_pred_post_T*.npy tests/fixtures/
mv tests/fixtures/.archive/golden_pred_post_T*.meta.json tests/fixtures/
```

Or rename your preferred fixture to sort first (e.g. `golden_pred_active_T*.npy`).

### Connecting it to the toggles

The fixture you just generated only matters when `INFERENCE_MODE=mock`. With `INFERENCE_MODE=runpod`, the API ignores fixtures entirely and calls the deployed worker.

A couple of useful combos for testing the **full pipeline** with a real image:

**Combo 1 — UI-iteration mode** (recommended for everyday work)
```env
INFERENCE_MODE=mock
SUGGESTION_LLM_MODE=anthropic
ANTHROPIC_API_KEY=sk-ant-...
STORAGE_MODE=minio
ENABLE_SUGGESTIONS=true
```
- Drop any image into the frontend → real upload, real Claude suggestions, brain scores from your fixture
- Each `/analyze` is instant
- Use this 99% of the time

**Combo 2 — Validate scores look right for a specific image** (one-off)
1. Run `build_post_fixture.py` against the image you care about
2. Restart the API (`make api-run`) so it sees the new fixture
3. Open the frontend in `INFERENCE_MODE=mock`
4. Drop **any** image (the upload + UI is real, but mock returns your post fixture's scores)

If you want each new image to produce its own real scores, you need either:
- RunPod (`INFERENCE_MODE=runpod` — see [RUNPOD_SESSION.md](RUNPOD_SESSION.md))
- A local persistent CPU worker that runs full TRIBE per call (~30-90 min each — described below)

### End-to-end test flow with a real image

The most representative local test of the full pipeline:

```bash
# 1. Pre-generate one fixture from your image (one time, ~30-60 min)
cd /Users/kirby/Documents/cortyze/cortyze_product
export HF_TOKEN=hf_...
/Users/kirby/Documents/cortyze/tribev2/.venv/bin/python scripts/build_post_fixture.py \
  --image-url https://your-cdn.com/your-image.jpg \
  --caption "Your real caption"

# 2. Confirm .env has the right toggle combo
grep -E "INFERENCE_MODE|SUGGESTION_LLM_MODE|STORAGE_MODE|ENABLE_SUGGESTIONS" .env
# Want:
#   INFERENCE_MODE=mock          ← mock returns the fixture you just made
#   SUGGESTION_LLM_MODE=anthropic ← real Claude suggestions on real content shape
#   STORAGE_MODE=minio            ← real upload to MinIO
#   ENABLE_SUGGESTIONS=true

# 3. Three terminals
./scripts/dev_minio.sh start     # T1
make api-run                     # T2 — picks up the new fixture on boot
cd ../cortyze_frontend && npm run dev  # T3
```

Open <http://localhost:3000>, click **Post**, drop an image, type a caption, pick a goal, run. What's real:
- ✅ Image upload to MinIO
- ✅ Brain scores (from the fixture you generated against your own image)
- ✅ Per-region bars + thumbnail under each
- ✅ Suggestions generated by real Claude with the right per-content-type system prompt
- ✅ Brain heatmap rendered from the real `(T, 20484)` array
- ✅ Frontend rendering of every report element

### Optional: persistent local CPU worker

Skip unless you want each `/analyze` to run real TRIBE fresh. ~30-90 min per call. Good for one-off "what would scores look like for this brand-new image I haven't fixture-ized yet" runs without reaching for RunPod.

```bash
cd /Users/kirby/Documents/cortyze/cortyze_product

# One-time: write a wrapper that applies CPU patches before booting the worker
cat > gpu_worker/local_handler.py <<'PY'
"""Local CPU-mode entry point for the GPU worker. M3 8GB only."""
import sys
sys.path.insert(0, "/Users/kirby/Documents/cortyze/tribev2")

# Apply CPU + bf16 patches BEFORE any tribev2 / gpu_worker import
from scripts.build_fixture import apply_mac_cpu_patches
apply_mac_cpu_patches()

from gpu_worker.handler import _build_app
import uvicorn

if __name__ == "__main__":
    app = _build_app()
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
PY

# Run it (uses the tribev2 venv which has neuralset)
export HF_TOKEN=hf_...
/Users/kirby/Documents/cortyze/tribev2/.venv/bin/python gpu_worker/local_handler.py
```

Then point the cortyze API at it via `.env`:
```env
INFERENCE_MODE=runpod
RUNPOD_POD_URL=http://127.0.0.1:8001
RUNPOD_TIMEOUT_SECONDS=14400
```

Restart `make api-run`. Now every `/analyze` runs real TRIBE locally. Be patient — first call ~90s of model load + 30-90 min of inference per request.

---

## Common dev tasks

### Run tests

```bash
cd cortyze_product
uv run pytest                      # all 113 tests, ~15s
uv run pytest tests/test_api.py    # just API smoke tests
uv run pytest -k suggestions       # suggestion engine tests
```

```bash
cd cortyze_frontend
npx tsc --noEmit                   # typecheck
npx next build                     # full prod build (catches more)
```

### Flip toggles

All three are independent — mix freely. See the **Toggle reference** table near the top for what each value does.

```env
# Suggestion LLM
SUGGESTION_LLM_MODE=anthropic            # or `mock` (free) / `openai_compatible`
ANTHROPIC_API_KEY=sk-ant-...

# Inference
INFERENCE_MODE=mock                      # or `runpod` (real GPU)
# RUNPOD_POD_URL=http://127.0.0.1:8001   # only when INFERENCE_MODE=runpod
# RUNPOD_API_KEY=...                     # only when INFERENCE_MODE=runpod (Serverless)
# RUNPOD_ENDPOINT_ID=...                 # only when INFERENCE_MODE=runpod (Serverless)

# Storage
STORAGE_MODE=minio                       # `off` / `minio` / `r2`
S3_ENDPOINT_URL=http://localhost:9000    # only when STORAGE_MODE=minio
R2_ACCESS_KEY=minioadmin                 # `minioadmin` for minio, real key for r2
R2_SECRET_KEY=minioadmin
R2_ACCOUNT_ID=<account-id>               # only when STORAGE_MODE=r2
```

Restart `make api-run` after any change. The startup banner will show what's active:
```
[cortyze] startup feature flags:
  inference:      mock
  object storage: minio (local)
  persistence:    off
  suggestions:    enabled (mode=anthropic)
```

### Stop everything

```bash
# Ctrl-C the frontend (Terminal 3)
# Ctrl-C the API (Terminal 2)
./scripts/dev_minio.sh stop
```

---

## Troubleshooting

### MinIO won't start / 403 on uploads

```bash
./scripts/dev_minio.sh stop
rm -rf ~/cortyze-minio-data       # nukes local data
./scripts/dev_minio.sh start
```

Confirm it's healthy:
```bash
curl -s http://localhost:9000/minio/health/cluster   # 200 = healthy
```

Common cause: previously running a bare `minio server` (without the script's env vars) leaves the data dir in a state where `minioadmin` no longer authenticates. Wiping the data dir fixes it.

### Frontend can't reach backend (CORS error)

Make sure `FRONTEND_ORIGINS=http://localhost:3000` is in `.env` and you restarted the API after changing it. The startup banner doesn't show CORS — check the `Access-Control-Allow-Origin` response header on a `curl http://localhost:8000/health -H 'origin: http://localhost:3000'` call.

### `/upload-url` returns 501

Means `STORAGE_MODE=off` (or unset, with no R2 vars). For local dev set `STORAGE_MODE=minio`. Restart the API.

### `/report/{id}` returns 501

Means `DATABASE_URL` isn't set. Either set it (local Postgres or Supabase) and run [services/persistence/migrations/001_reports.sql](services/persistence/migrations/001_reports.sql), or just live without persistence — `/analyze` still works without it.

### Suggestions don't appear in the report

Check `ENABLE_SUGGESTIONS=true` is in `.env` and the API was restarted. Also confirm a region is actually low-scoring in the fixture — the engine only fires on regions where score < 50 and goal-weight ≥ 5%.

### `tsc --noEmit` errors after pulling

```bash
cd cortyze_frontend
rm -rf .next node_modules
npm install
npx tsc --noEmit
```

### Backend tests fail with `OSError: ... no module named 'numpy'`

```bash
cd cortyze_product
uv sync     # re-resolves and installs deps
```

If `uv` itself complains about Python version, install Python 3.11+: `brew install python@3.11`.

---

## Architecture cheat sheet

For a deeper tour: [HANDOFF.md](HANDOFF.md). For scaling decisions: [SCALING.md](SCALING.md). For RunPod deployment: [RUNPOD_SESSION.md](RUNPOD_SESSION.md). For measured GPU latency / cost: [docs/runpod_benchmark.md](docs/runpod_benchmark.md).

The single contract holding it all together is `np.ndarray (T, 20484)` — the cortical activations TRIBE produces. Mock mode just returns a saved version of that array. Everything from atlas mapping → region scores → goal weighting → brain PNG → suggestions runs on top of it identically whether the array came from sintel-on-M3 or real-A40-on-RunPod.

---

## When you're ready for real GPU

When you actually need real brain scores:

1. **One-off real run** — RunPod pod, ~$0.10/clip, 5-10 min wall. See [RUNPOD_SESSION.md](RUNPOD_SESSION.md).
2. **Save the result as a fixture** — `tests/fixtures/golden_pred_*.npy`. Mock mode picks it up automatically.
3. **Iterate locally on the saved fixture** — no more GPU bills until you need a different clip.

For production, deploy the GPU worker as a RunPod Serverless endpoint and flip `INFERENCE_MODE=runpod` + `RUNPOD_ENDPOINT_ID` + `RUNPOD_API_KEY` in your prod env. Same code, same frontend, just a different env var.
