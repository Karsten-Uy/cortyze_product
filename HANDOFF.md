# Cortyze Handoff

If you just inherited this project, read this first. ~10 minutes for orientation, then jump to the doc that matches your task.

---

## TL;DR

**Cortyze** is a neural content-prediction tool. Upload a video / image / text, pick a marketing goal (Conversion / Awareness / Engagement / Brand Recall), get back **8 brain-region scores plus a goal-weighted overall score**. Built on top of Meta's **TRIBE v2** (CC BY-NC, open-source brain-encoding foundation model).

**Current state:**
- **Backend engine** (FastAPI) — works end-to-end. Mock mode returns deterministic synthetic data for free local dev; flip `INFERENCE_MODE=runpod` to switch to real GPU.
- **Frontend** (Next.js 16 + Tailwind v4 + React 19) — drag-drop upload, goal selector, brain heatmap PNG, region cards with top-3 highlighted.
- **GPU worker** (RunPod) — deployed end-to-end on A40, full pipeline validated against real inference. See [docs/runpod_benchmark.md](docs/runpod_benchmark.md) for measured numbers.
- **Storage** — R2 client coded, MinIO running locally as a stand-in. Switch via `STORAGE_MODE` env var. Supabase Postgres provisioned, schema migrated.
- **Tests** — 59 passing. CPU-derived golden fixture (sintel trailer, 53 timesteps) drives the e2e regression test.

**Inference reality check:** TRIBE v2 is a research model designed for batched offline runs, not interactive inference. Warm latency is ~660 s for a 52 s clip on A40, dominated 87% by V-JEPA2-Giant video encoding. See [SCALING.md](SCALING.md) for the speedup tiers (Tier 1 free wins drop a 15 s clip to ~150 s effective with feature caching). Plan async UX, not real-time.

---

## Stage roadmap

Six product stages from the strategy doc. Roughly:

| Stage | What it ships | % done | What's blocked on |
|---|---|---|---|
| **1 — Engine** | TRIBE v2 → 8 region scores → goal-weighted overall via JSON API | ~95% | Cross-clip calibration; Tier 1 speedup work (see [SCALING.md](SCALING.md)) |
| **2 — Diagnostic layer** | Suggestion engine (Claude) + reference ad library + matched examples in UI | 5% (lib scaffolded, 1 ad registered) | Stage 1 done; needs 3–5 more reference ad fixtures |
| **3 — Landing page** | Marketing copy, waitlist signups, Vercel deploy at custom domain | 0% (frontend scaffold exists) | Stage 1 demo-ready; copy + design |
| **4 — Account linking** | Instagram/TikTok OAuth, audience-brain profiling | 0% | Stage 3 acquiring users first |
| **5 — Commercial license / own model** | Either Meta licensing deal or train Cortyze model on collected data | 0% | Stage 4 data flywheel |
| **6 — Full launch** | Stripe, paid tiers, rate limits, SOC2-lite | 0% | Everything before |

Long-term: **Mirofish integration** — Mirofish renders billboard / sponsorship / OOH scenes; calls our `/analyze`. Already supported by the API surface; no work needed today beyond exposing a `/predict_raw` endpoint when they want vertex-level data.

---

## Architecture

Three tiers. Each has a stable contract; the contract is what lets stages plug in additively without rewrites.

```
┌───────────┐  POST /analyze    ┌──────────────────┐  HTTP    ┌──────────────────────┐
│ cortyze-  │ ─────────────────▶│ cortyze-product  │ ────────▶│ gpu_worker (RunPod)  │
│ frontend  │ ◀──────────────── │  (FastAPI)       │ ◀────────│  TribeModel.predict  │
│ (Next.js) │   BrainReport     │                  │  (T,     │  → (T, 20484)        │
└───────────┘   JSON+brain.png  └────────┬─────────┘  20484)  └──────────┬───────────┘
                                         │                               │
                                ┌────────▼─────────┐         ┌───────────▼──────────┐
                                │ R2: predictions/ │         │ R2: uploads/         │
                                │  (server writes) │         │  (browser writes)    │
                                │ Supabase reports │         └──────────────────────┘
                                └──────────────────┘
```

### The single contract that holds it together

Everything between the GPU and the JSON response is **pure NumPy**:

```
TribeModel.predict()  →  np.ndarray (T, 20484)  →  aggregate()  →  dict[str, float]
                                                  →  normalize()   →  dict[str, float]   (0–100)
                                                  →  overall_score(goal)  →  float       (0–100)
                                                  →  BrainReport (pydantic)
```

That `np.ndarray (T, 20484)` is the contract. Whether it comes from `MockRunPodClient` (synthetic), `RunPodPodClient` (Pod HTTP), `RunPodClient` (serverless), or a future Cortyze-trained model swap-in — the rest of the pipeline doesn't care.

### Eight marketing regions

The `(T, 20484)` array is per-vertex on the **fsaverage5** cortical surface mesh. We aggregate it into 8 named regions using the **Desikan-Killiany atlas** (via a Destrieux→DK projection — see [scripts/build_atlas_labels.py](scripts/build_atlas_labels.py)):

| Region | Marketing question it answers |
|---|---|
| `visual_cortex` | Do visuals grab attention? |
| `fusiform_face` | Do faces create connection? |
| `amygdala` (insula proxy) | Is there emotional impact? |
| `prefrontal` | Is the viewer considering action? |
| `temporal_language` | Is the message processed? |
| `hippocampus` | Will they remember it? |
| `motor` | Do they want to act (swipe, click)? |
| `reward` | Does it feel rewarding? |

Region keys are stable identifiers. `core/atlas/regions.py` is the single source of truth — all other layers (scoring, schemas, frontend types) reference these strings.

### Goal weighting

Each of the 4 marketing goals weights the 8 regions differently. From [core/scoring/goals.py](core/scoring/goals.py):

```
overall_score = Σ(region_score × goal_weight[region])
```

Each goal column sums to 1.0 (verified by tests). Same content, different goal → different overall score; region scores are stable.

### Storage planes (split by who writes)

| Bucket / Table | Writer | Why |
|---|---|---|
| **R2 `uploads/`** | Browser (presigned PUT) | User videos. 7-day TTL. Browser-writable, so this bucket needs CORS. |
| **R2 `predictions/`** | API server | The `(T, 20484)` array, float16 NPZ, ~2 MB each. Indefinite TTL. **This is the Stage 5 training set in the making.** No CORS needed. |
| **Supabase `reports` table** | API server | One row per `/analyze` call: `request_id`, `goal`, `region_scores` (jsonb), `overall_score`, `model_version`, pointer to R2 prediction. Stage 4 joins this against `engagement_metrics` for the audience-brain profile. |

### Mock vs real inference

[api/clients/runpod.py](api/clients/runpod.py) `get_client()` reads `INFERENCE_MODE`:

| `INFERENCE_MODE` | Picks | Use case |
|---|---|---|
| `mock` (default) | `MockRunPodClient` | All local dev. Free, deterministic. Returns saved fixture if one exists, else synthetic noise with per-region biases. |
| `runpod` + `RUNPOD_POD_URL` | `RunPodPodClient` | Direct Pod HTTP. For GPU iteration during bring-up. |
| `runpod` + `RUNPOD_ENDPOINT_ID` + `RUNPOD_API_KEY` | `RunPodClient` | RunPod Serverless `/runsync`. Production. |

Tests don't set `INFERENCE_MODE` so they default to mock — flipping the toggle in `.env` doesn't break the test suite.

---

## File map

Where to start reading, by area:

| Path | What's in it | Entry point |
|---|---|---|
| [core/](core/) | Pure-Python library: atlas, scoring, schemas. **No I/O. No FastAPI. Importable from anywhere.** Stage 4's batch worker, Stage 5's training pipeline, Mirofish — all consume from here. | [core/schemas.py](core/schemas.py) for the data shapes |
| [api/](api/) | FastAPI app, routes, RunPod clients, prediction pipeline | [api/predict.py](api/predict.py) — the central function |
| [services/](services/) | Bounded contexts. Today: `storage/r2.py`, `persistence/reports.py`, `examples/library.py`, `visualization/brain_plot.py`. Stage 2 adds `suggestions/` here. | [services/persistence/migrations/001_reports.sql](services/persistence/migrations/001_reports.sql) for the DB schema |
| [gpu_worker/](gpu_worker/) | Code that runs on RunPod, NOT on your Mac. Loads TRIBE v2, exposes Pod-mode FastAPI + Serverless `def handler(event)`. | [gpu_worker/README.md](gpu_worker/README.md) for deploy steps |
| [docker/](docker/) | RunPod Dockerfile + build script. Cross-compiles linux/amd64 from Apple Silicon. | [docker/runpod.Dockerfile](docker/runpod.Dockerfile) |
| [scripts/](scripts/) | One-shot dev tools: build_fixture (TRIBE v2 on Mac CPU), build_atlas_labels (Destrieux→DK), register_reference_ad, calibrate_from_fixture, plot_fixture, benchmark_runpod | [scripts/build_fixture.py](scripts/build_fixture.py) for the CPU bridge |
| [tests/](tests/) | pytest suites. 59 tests, 0.2s run. Includes the golden-snapshot regression test. | [tests/test_e2e_golden.py](tests/test_e2e_golden.py) is the integration anchor |
| [tests/fixtures/](tests/fixtures/) | The real `(T, 20484)` array from a Mac CPU run of TRIBE v2 against sintel. ~4 MB committed. | [tests/fixtures/golden_pred_sintel_T53.npy](tests/fixtures/golden_pred_sintel_T53.npy) |
| [data/reference_ads/](data/reference_ads/) | Manifests for Stage 2's example library. JSON per ad with computed scores. | [data/reference_ads/sintel_trailer.json](data/reference_ads/sintel_trailer.json) |
| `../cortyze_frontend/` | Next.js demo (sibling repo) | [../cortyze_frontend/app/page.tsx](../cortyze_frontend/app/page.tsx) |
| `../tribev2/` | Sibling clone of Meta's tribev2 repo. Used by `build_fixture.py` to run inference on the Mac. | [../tribev2/run.py](../tribev2/run.py) is the CPU+bf16 reference |

---

## Local development (3 terminals)

```bash
# Terminal 1: object storage
cd cortyze_product
./scripts/dev_minio.sh start

# Terminal 2: API
uv sync         # first time only
uv run uvicorn api.main:app --reload

# Terminal 3: frontend
cd ../cortyze_frontend
npm install     # first time only
npm run dev
```

Open <http://localhost:3000>. Drag any video file into the dropzone, pick a goal, click Run BrainScore. With INFERENCE_MODE=mock (default) you'll see deterministic scores derived from the sintel fixture. Full quick-start with env-var details lives in [README.md](README.md).

---

## TODO

Status legend: ✅ done · 🚧 in progress · ⏳ blocked on user / external · ⏸ deferred (next stage)

### Stage 1 — Engine (~85%)

- ✅ Atlas mapper, scoring, goals, schemas — 59 tests
- ✅ FastAPI: `/analyze`, `/upload-url`, `/report/{id}`, `/health` + CORS
- ✅ Storage: R2 client + MinIO local override via `S3_ENDPOINT_URL`
- ✅ Persistence: `ReportsStore` (psycopg, raw SQL, no ORM)
- ✅ Mock + RunPodPod + RunPod (serverless) clients
- ✅ `INFERENCE_MODE` toggle
- ✅ Brain visualization (nilearn, base64 PNG inline in BrainReport)
- ✅ Frontend demo with drag-drop + brain image
- ✅ Reference library scaffold + sintel registered
- ✅ Single-clip calibration from sintel fixture
- ✅ E2E golden test (`tests/test_e2e_golden.py`)
- ✅ RunPod benchmark script ready
- ✅ Supabase provisioned + migration run
- ✅ Cloudflare R2 provisioned + CORS configured
- ✅ RunPod end-to-end pipeline validated on A40 — see [docs/runpod_benchmark.md](docs/runpod_benchmark.md)
- 🚧 **Tier 1 speedup work** (frontend cap, `torch.compile`, visual-only audio skip, resolution downscale) — ~3× combined warm-call speedup. See [SCALING.md](SCALING.md).
- 🚧 **V-JEPA feature caching** (Tier 2, ~1 day) — biggest single product win, ~7× effective speedup on multi-goal re-runs. See [SCALING.md](SCALING.md) §E.
- 🚧 **Async UX with progress polling** (Tier 2) — required given 5–10 min real latency. See [SCALING.md](SCALING.md) §F.
- 🚧 §6 concurrency probe + §8 failure-mode tests in [docs/runpod_benchmark.md](docs/runpod_benchmark.md) (~20 min, $0.15)
- ⏸ Cross-clip calibration with 30 reference clips
- ⏸ Phase 7 RunPod Serverless conversion + benchmark

### Stage 2 — Suggestion engine + Example library (5%)

- ✅ `services/examples/library.py` query API (`top_n_for_region`, `top_n_for_goal`)
- ⏸ `services/suggestions/` — threshold rules: when does a low region score warrant a suggestion?
- ⏸ Claude API integration (claude-sonnet-4-6) for suggestion text. Prompt-cache the rules block.
- ⏸ 3–5 more reference ad fixtures (build_fixture → register_reference_ad). Curated URLs in [data/reference_ads/README.md](data/reference_ads/README.md).
- ⏸ Frontend suggestion cards + matched-example renders (extend [../cortyze_frontend/app/page.tsx](../cortyze_frontend/app/page.tsx))
- ⏸ Per-region threshold tuning (probably needs ~50 real user analyses to calibrate)

### Stage 3 — Landing page (0%)

- ⏸ Marketing copy + hero (paste from spec doc)
- ⏸ Waitlist email signup (Resend or Postmark)
- ⏸ Hero brain image (use `docs/brain_demo.png` or render fresh)
- ⏸ Vercel deploy from `cortyze_frontend/` repo
- ⏸ Custom domain (cortyze.com or similar) on Cloudflare DNS
- ⏸ PostHog or Plausible analytics
- ⏸ Backend deploy (Railway / Fly) so the production frontend has a real API

### Stage 4 — Account linking + audience profiling (0%)

- ⏸ Supabase Auth (magic link)
- ⏸ Instagram Graph API OAuth (Business / Creator accounts only)
- ⏸ TikTok Login Kit OAuth
- ⏸ Background job queue (RQ on Redis, or Supabase Edge Functions)
- ⏸ Backfill: pull last 50–100 posts → run each through `/analyze`
- ⏸ Audience profile builder: Pearson correlations of `region_scores` vs engagement metrics
- ⏸ Personalized scoring (per-creator goal-weight overrides)

### Stage 5 — Commercial license / proprietary model (0%)

- ⏸ Train Cortyze brain encoder on `(content, raw_predictions, engagement)` triples — that data is already being persisted from day 1 (see §6.3 forward-compat)
- ⏸ OR negotiate commercial TRIBE v2 license with Meta FAIR
- ⏸ Model registry + `model_version` rollout policy

### Stage 6 — Full launch (0%)

- ⏸ Stripe billing (paid tiers per the strategy doc)
- ⏸ Rate limiting (Upstash Ratelimit)
- ⏸ SOC2-lite (Sentry, audit logs, PII handling)
- ⏸ Marketing automation
- ⏸ Higher-concurrency RunPod scaling

### Mirofish (long-term, separate codebase)

- ⏸ Expose `POST /predict_raw` returning unmodified `(T, 20484)`
- ⏸ Mirofish renders environments → MP4 → calls our `/analyze`
- ⏸ API-key authentication for B2B partners

### Cross-cutting infra / DX

- ⏸ GitHub Actions CI (pytest backend + `next build` frontend on PR)
- ⏸ Sentry (both repos)
- ⏸ Axiom or Logtail structured log sink
- ⏸ Rate limiting (slowapi for FastAPI)
- ⏸ `openapi-typescript` autogen for frontend types from `/openapi.json`
- ⏸ Docker Compose for `cortyze_product` so MinIO + API come up together
- ⏸ Domain purchase + Cloudflare DNS

---

## Forward-compat decisions that paid off

These were called out in [IMPLEMENTATION_PLAN.md §6](IMPLEMENTATION_PLAN.md). They cost ~zero to add at Stage 1 and unlock Stages 2–6 + Mirofish without rewrites:

1. **`Goal` is a typed enum** ([core/scoring/goals.py](core/scoring/goals.py)) — Stage 2's suggestion threshold rules and Stage 4's audience profile both consume the same values. Stringly-typed goals would have pattern-match-bugged in every later stage.
2. **`request_id` end-to-end** — UUID at API entry, in every log line, in Supabase, in the R2 prediction filename. Stage 4 joins engagement data on this; Stage 5 joins training data on this. Without it, neither is reconstructible.
3. **Schemas carry forward fields** — `BrainReport.user_id`, `model_version`, `raw_predictions_uri` are populated when relevant; otherwise None. Stage 4's user_id and Stage 5's model swap don't require schema changes.
4. **Inference is callable, not request-bound** — `predict_brain_report(req)` is a plain function in [api/predict.py](api/predict.py). Stage 4's batch backfill imports and loops it; doesn't entangle with FastAPI's request lifecycle.
5. **`core/` is pure (no I/O)** — atlas + scoring + goals + schemas import nothing from `api/` or `services/`. Mirofish or a Cortyze CLI could import `core` directly without dragging FastAPI along.

---

## Open decisions / gotchas

Things the next person might want to revisit:

1. **Single-clip calibration is a placeholder.** [core/scoring/calibration.json](core/scoring/calibration.json) was derived from one Mac-CPU run of sintel. The shared `mu` (clip-baseline) and per-region `sigma` (within-region std) give meaningful inter-region differentiation, but absolute scores are only meaningful relative to other content. **Replace with 30-clip cross-clip statistics post-RunPod.** Schema is stable; just rewrite the JSON.
2. **Amygdala uses insula as a cortical proxy.** TRIBE v2 outputs cortical surface vertices only; the amygdala is subcortical. Insula is the documented stand-in (see [core/atlas/regions.py](core/atlas/regions.py) + [IMPLEMENTATION_PLAN.md §8](IMPLEMENTATION_PLAN.md)). Re-evaluate when ground-truth ad data is available.
3. **`neuralset==0.0.2` install risk on RunPod.** Listed as a tribev2 transitive dep; needs to resolve from PyPI inside the Docker image. If it doesn't, the fallback is to vendor or use a wheel mirror — not yet hit, but flagged.
4. **Brain PNG rendering is ~1.85s per request.** Acceptable for Stage 1 but adds latency. Easy follow-ups: lower DPI from 80→60, render asynchronously and SSE-stream the result, or render on the frontend (compute-heavy lifting in WebGL).
5. **MinIO is not reachable from RunPod.** The drag-drop upload path posts to `localhost:9000`, which the cloud GPU worker can't fetch. For end-to-end pipeline tests against real GPU, you need real Cloudflare R2 (or test via direct public URLs like the sintel CDN). Migration is documented in [README.md](README.md) §3.2.
6. **`tribev2/run.py` has a hardcoded `HF_TOKEN`.** Per the workspace [CLAUDE.md](../CLAUDE.md), do not propagate that pattern. `scripts/build_fixture.py` correctly reads from env. Keep an eye on it if you ever pull updates from upstream tribev2.
7. **`caudalmiddlefrontal` not represented in our DK label vocabulary.** The Destrieux→DK projection in [scripts/build_atlas_labels.py](scripts/build_atlas_labels.py) doesn't subdivide `G_front_middle`, so `regions.py`'s "caudalmiddlefrontal" entry has zero vertices. Stage 2 can fix with proper FreeSurfer aparc.
8. **Inference is genuinely slow and that's structural, not a bug.** A 52 s clip takes 5–12 min on an A40 because V-JEPA2-Giant is 1B params and processes video in 64-frame chunks. The model wasn't built for interactive use. Treat this as a product constraint and design accordingly: **frontend should cap clips at 15 s and use async UX with progress polling**. Full speedup roadmap with realistic gains lives in [SCALING.md](SCALING.md).
9. **Warm latency has ±15% variance.** Across 4 sintel runs we saw 602–717 s range. The fastest run was an outlier; typical p50 is ~692 s. Quote ranges to users ("6–12 minutes for ~1 minute clips"), never point estimates. `torch.compile` should reduce this jitter — see [SCALING.md](SCALING.md) §B.
10. **The `S3_ENDPOINT_URL`-based MinIO/R2 switch is fragile.** Replaced with explicit `STORAGE_MODE=off|minio|r2` env var. If your `.env` predates this change, both the legacy auto-detect path and the new explicit one work, but new setups should use `STORAGE_MODE`.

---

## Documents to read in order

If you're ramping up on this project:

1. **This file** (HANDOFF.md) — bird's-eye view, you're here
2. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — Stage 1 blueprint with §6 forward-compat decisions and §11 frontend gaps
3. [README.md](README.md) — local quick-start + env-var → behavior table
4. [SCALING.md](SCALING.md) — speedup levers, capacity planning, when to invest in which optimization
5. [docs/runpod_benchmark.md](docs/runpod_benchmark.md) — measured A40 latency / cost / phase breakdown — the source data for SCALING.md
6. [RUNPOD_SESSION.md](RUNPOD_SESSION.md) — step-by-step RunPod deployment session with troubleshooting log
7. [gpu_worker/README.md](gpu_worker/README.md) — RunPod deploy + cost-optimization rules
8. [data/reference_ads/README.md](data/reference_ads/README.md) — Stage 2 reference library + curated video list
9. The strategy doc (in conversation history with the original Cortyze brief) — product vision, business case, competitive landscape

For a code tour:
1. [core/schemas.py](core/schemas.py) — the data shapes
2. [core/atlas/regions.py](core/atlas/regions.py) → [core/atlas/mapper.py](core/atlas/mapper.py) → [core/scoring/normalize.py](core/scoring/normalize.py) → [core/scoring/goals.py](core/scoring/goals.py) — the math, in pipeline order
3. [api/predict.py](api/predict.py) — the central function that wires it all together
4. [api/clients/runpod.py](api/clients/runpod.py) — mock/pod/serverless selection
5. [gpu_worker/inference.py](gpu_worker/inference.py) → [gpu_worker/handler.py](gpu_worker/handler.py) — what runs on the GPU
6. [tests/test_e2e_golden.py](tests/test_e2e_golden.py) — the integration anchor; run this and read it to understand the full pipeline in <100 lines
