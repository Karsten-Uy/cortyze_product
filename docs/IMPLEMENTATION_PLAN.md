# Stage 1 Implementation Plan — BrainScore Engine

**Exit criteria:** Upload a video / image / text + select a goal → API returns 8 brain-region scores (0–100) and one goal-weighted overall score (0–100). Engine works end-to-end.

This plan diverges from the strategy doc in one place: **GPU host is RunPod, not Modal.** Reason: the developer environment is an 8GB M3 Mac (cannot run TRIBE v2 locally — needs ~40GB VRAM), and we want pay-per-second GPU compute we can ssh into for debugging during the bring-up.

---

## 1. Architecture

```
┌──────────────────┐   POST /analyze   ┌────────────────────┐   HTTPS   ┌─────────────────────────┐
│  Client / curl   │ ────────────────▶ │  FastAPI (local /  │ ────────▶ │  RunPod GPU worker      │
│                  │ ◀──────────────── │  Railway later)    │ ◀──────── │  TRIBE v2 (A100 40GB)   │
└──────────────────┘   JSON scores     └────────────────────┘           │  → (T, 20484) np.array  │
                                              │                          └─────────────────────────┘
                                              ▼
                                     ┌──────────────────────┐
                                     │  Atlas Mapper        │   pure numpy, runs on the API host
                                     │  Scoring + Goal Mix  │   (no GPU needed for this step)
                                     └──────────────────────┘
```

**Why split this way:** the heavy work (model inference) is on RunPod where GPU is. Vertex→region aggregation and 0–100 normalization are ~kilobytes of math; running them on the API host keeps the GPU worker stateless and lets us iterate on scoring without redeploying images.

**Two-tier deploy decision for Stage 1:** start with a **RunPod Pod** (always-on while we're building) for fast iteration; switch to a **RunPod Serverless endpoint** at the end of Stage 1 so it scales to zero when idle.

---

## 2. Repo layout (`cortyze_product/`)

```
cortyze_product/
├── pyproject.toml              # uv-managed, Python 3.11+
├── README.md
├── IMPLEMENTATION_PLAN.md      # this file
├── .env.example                # HF_TOKEN, RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID
├── docker/
│   └── runpod.Dockerfile       # CUDA 12.1, torch, tribev2, weights pre-pulled
├── gpu_worker/                 # what runs on RunPod
│   ├── handler.py              # entrypoint (serverless) / FastAPI (pod)
│   ├── inference.py            # TribeModel wrapper, returns (T, 20484) np
│   └── requirements.txt
├── api/                        # what runs locally / on Railway later
│   ├── main.py                 # FastAPI app
│   ├── routes/
│   │   ├── analyze.py          # POST /analyze
│   │   └── health.py
│   └── clients/runpod.py       # HTTP client for the worker
├── core/                       # pure-python, no GPU; one-way dep target — api/, gpu_worker/, services/ may import this, never the reverse (§6.2)
│   ├── atlas/
│   │   ├── mapper.py           # (T, 20484) → dict[region → activation]
│   │   ├── regions.py          # 8-region grouping table (single source of truth)
│   │   └── data/
│   │       └── fsaverage5_dk_labels.npy   # precomputed (20484,) int label array
│   ├── scoring/
│   │   ├── normalize.py        # raw activation → 0–100
│   │   └── goals.py            # Goal enum + weight table + overall score
│   └── schemas.py              # pydantic models — extension fields baked in now: request_id, user_id?, model_version (§6.4)
├── services/                   # bounded contexts — empty in Stage 1, populated in Stage 2 (§6.7)
│   └── README.md               # placeholder: suggestions/ + examples/ land here
└── tests/
    ├── test_atlas.py
    ├── test_scoring.py
    ├── test_goals.py
    └── fixtures/
        └── golden_pred_T120.npy   # canned (120, 20484) array for offline tests
```

---

## 3. Phase 1.1 — Atlas + Scoring (local, no GPU) — ~3 days

Everything in `core/` is pure NumPy and runs on the Mac. Build and test this first; it de-risks the rest.

### 3.1 Atlas Mapper

- TRIBE v2 predicts on **fsaverage5**: 10,242 vertices per hemisphere, 20,484 total. Vertex order convention (used by `tribev2.plotting`): `[lh_0..lh_10241, rh_0..rh_10241]`. Verify by reading `tribev2/tribev2/plotting/` and `utils_fmri.py` before coding.
- Need a `(20484,)` int array assigning each vertex to one of the 68 Desikan-Killiany labels (or "unknown"). Generate once and ship as `core/atlas/data/fsaverage5_dk_labels.npy`:
  - Use `nilearn.datasets.fetch_atlas_surf_destrieux` (Destrieux) **or** download `lh.aparc.annot` / `rh.aparc.annot` from FreeSurfer's fsaverage subject and resample to fsaverage5. Prefer FreeSurfer `aparc` since the spec uses DK names like `pericalcarine`, `cuneus`, etc.
  - Build a tiny one-shot script `scripts/build_atlas_labels.py` that produces the `.npy` and a JSON mapping `{label_id: label_name}`.
- `core/atlas/regions.py` — single source of truth for the 8 regions, copied verbatim from the spec table:
  ```python
  REGIONS = {
      "visual_cortex": ["pericalcarine", "cuneus", "lateraloccipital", "lingual"],
      "fusiform_face": ["fusiform"],
      "amygdala":      ["..."],   # see open question (§8)
      "prefrontal":    ["superiorfrontal", "rostralmiddlefrontal", ...],
      "temporal_language": ["superiortemporal", "middletemporal", ...],
      "hippocampus":   ["parahippocampal", "entorhinal"],
      "motor":         ["precentral", "postcentral", "paracentral"],
      "reward":        ["rostralanteriorcingulate", "caudalanteriorcingulate", ...],
  }
  ```
- `mapper.py:aggregate(preds: np.ndarray) -> dict[str, float]`:
  1. Average across the time axis → `(20484,)`
  2. For each region, mean over vertex indices belonging to its DK labels → 8 floats
  3. Return `{region_name: mean_activation}`
- Unit test against `golden_pred_T120.npy` — assert all 8 keys present, values finite, shape invariants hold.

### 3.2 Scoring (raw → 0–100)

- TRIBE v2 outputs are **raw predicted BOLD signal** (continuous, roughly z-ish per-vertex per the training objective). They are not 0–100.
- **Stage 1 minimum:** sigmoid-then-scale calibration with hardcoded constants. We pick `(mu, sigma)` per region by running ~30 reference clips through the model (sintel + a handful of ad samples) and storing per-region mean/std. Then:
  ```python
  z = (raw - mu) / sigma
  score = 100 * 1 / (1 + exp(-z))   # bounded, monotonic, smooth
  ```
- Persist the calibration constants in `core/scoring/calibration.json` (committed). They get refined in Stage 2 once the reference ad library lands; the schema doesn't change.
- This is a deliberate quick-and-dirty: the absolute number is meaningless, but **rank order across content is preserved** and that's what diagnosis needs. Document this explicitly in the docstring.

### 3.3 Goal mixing

- `core/scoring/goals.py` — `Goal` enum (`CONVERSION | AWARENESS | ENGAGEMENT | BRAND_RECALL`) plus the four weight columns from the spec verbatim. Sanity check: each column sums to 1.00 (the table sums to 100%).
- `overall_score(region_scores: dict, goal: Goal) -> float`: weighted sum of the 8 region scores by the goal's column, returning 0–100. Stage 2's suggestion thresholds and Stage 4's audience profile both consume the same `Goal` enum — no stringly-typed goals downstream (§6.5).

### 3.4 Tests (pytest)

- `test_atlas.py`: known synthetic input (e.g. all-ones at visual_cortex vertices) → only `visual_cortex` activates.
- `test_scoring.py`: monotonicity (higher raw → higher score), bounded [0, 100], NaN/inf rejected.
- `test_goals.py`: each goal column sums to 1.0; goal score is bounded [0, 100]; goal weights match the spec table exactly.

Exit for Phase 1.1: `pytest` green on the Mac, no GPU touched.

---

## 4. Phase 1.2 — TRIBE v2 on RunPod — ~4 days

### 4.1 Build the GPU worker image

`docker/runpod.Dockerfile`:
- Base: `runpod/pytorch:2.5.1-py3.11-cuda12.1.0-devel-ubuntu22.04` (or current equivalent)
- Install `tribev2` from the sibling repo:
  - For Stage 1: `pip install -e git+https://github.com/facebookresearch/tribev2.git@main#egg=tribev2`
  - Once we fork or pin: switch to a tagged commit
- Pre-pull the gated `facebook/tribev2` weights at **build time** with `huggingface-cli download` so cold start doesn't pay for the ~10GB download. Requires `HF_TOKEN` as a build secret. Cache directory: `/opt/hf_cache`, exposed via env var `HF_HOME`.
- Also pre-pull V-JEPA2-Giant, Wav2Vec-BERT 2.0, LLaMA-3.2-3B (these are what TRIBE v2 loads internally — check `tribev2/main.py` and `demo_utils.py` for the exact HF repo IDs and bake them all in).
- Total image size will be ~25–30 GB. That's fine; RunPod pulls once per node.

### 4.2 GPU worker code

`gpu_worker/inference.py`:
```python
class BrainPredictor:
    def __init__(self):
        self.model = TribeModel.from_pretrained(
            "facebook/tribev2",
            cache_folder="/opt/hf_cache",
            device="cuda",
        )

    def predict(self, *, video_path=None, audio_path=None, text=None) -> np.ndarray:
        # build events df via model.get_events_dataframe(...)
        # NO balanced_df.head(1) hack — we have GPU now, run all events
        preds, segments = self.model.predict(events=df)
        return preds   # (T, 20484) float32
```

The model loads once at process start (~30s) and stays warm for the life of the worker.

`gpu_worker/handler.py` (two flavors, same code path):
- **Pod mode (during Stage 1 development):** wrap in FastAPI, expose `POST /predict` accepting `{video_url, audio_url, text}`. Run with `uvicorn` on port 8000. Tunnel via RunPod's public TCP proxy.
- **Serverless mode (end of Stage 1):** export a `def handler(event)` that RunPod's serverless runtime calls. Same `BrainPredictor.predict` underneath.

Build both from day one — they're a 30-line diff.

### 4.3 Deploy Pod

- A100 40GB, on-demand (~$2/hr at current RunPod prices). Mount a network volume at `/opt/hf_cache` so weights survive pod restarts even if we rebuild the image.
- Confirm: end-to-end a 30s sintel clip → returns `(T, 20484)` array in <60s wall clock. Smoke test from the Mac via curl.
- **Cost discipline:** stop the pod when not actively testing. Add a `make pod-up` / `make pod-down` to the repo so it's a one-liner.

### 4.4 Convert to Serverless endpoint

Once the pod path works:
- Push image to a registry (RunPod's, or GHCR)
- Create a Serverless endpoint pointing at the image, min-workers=0, max-workers=2, idle timeout=5min
- Switch the API client to call the serverless endpoint URL

Cold start will be ~30s (model load). For Stage 1 that's acceptable; we revisit warm-pool tactics in Stage 4 when concurrency matters.

---

## 5. Phase 1.3 — API + glue + storage — ~3 days

### 5.1 FastAPI server (`api/`)

Endpoints — expanded vs. an earlier draft to give the Stage 3 frontend (§11) a clean integration surface, so Stage 1 → Stage 3 doesn't require an API rewrite:
- `POST /analyze` — body: `AnalyzeRequest` → streams **Server-Sent Events** during processing (`extracting_audio`, `predicting`, `aggregating_regions`, `done`), terminating with the `BrainReport` payload. Without SSE the frontend stares at a 30-second blank spinner; with it, the UX feels alive.
- `POST /upload-url` — returns `{put_url, get_url, content_url}` — presigned R2/S3 PUT + GET URLs valid for 5 min. Browser uploads the file directly to object storage, then sends `content_url` to `/analyze`. Keeps the API server out of the upload data path.
- `GET /report/{request_id}` — fetches a persisted `BrainReport`. Required for shareable links and frontend page refreshes. Backed by the `reports` table from §5.2.
- `GET /health` — liveness + reports whether the RunPod endpoint responds.
- (Reserved, **not built yet**) `POST /predict_raw` returns the `(T, 20484)` array unmodified — Mirofish will want this; `gpu_worker/`'s contract supports it for free (§6.8).

**CORS middleware** allow-lists `http://localhost:3000` (Next.js dev) and the production frontend origin via env vars. Without it, no browser request reaches the API at all.

Schemas — see §6.3 / §6.4 for why the optional fields are baked in now:
```python
class AnalyzeRequest(BaseModel):
    content_url: str
    content_type: Literal["video", "image", "text"]
    goal: Goal
    user_id: Optional[str] = None        # populated in Stage 4
    request_id: str = Field(default_factory=lambda: str(uuid4()))

class BrainReport(BaseModel):
    request_id: str                       # join key for Stage 4 engagement data + Stage 5 training set
    region_scores: dict[str, float]       # 8 keys, 0-100
    overall_score: float                  # 0-100
    goal: Goal
    user_id: Optional[str] = None
    model_version: str                    # e.g. "tribev2-2026-03"; required for Stage 5 model swap
    raw_predictions_uri: Optional[str] = None   # R2/S3 URI for the (T, 20484) array
    elapsed_ms: int
```

`api/clients/runpod.py` calls the RunPod endpoint, deserializes the `(T, 20484)` array (msgpack or base64-encoded npz — msgpack is smaller), passes it to `core.atlas.aggregate` and `core.scoring`.

### 5.2 Storage + persistence (slim Stage 2 pull-forward)

To support `GET /report/{request_id}`, presigned uploads, and the §6.3 prediction-persistence requirement, Stage 1 needs the smallest possible storage footprint:

- **Object storage — Cloudflare R2** (recommended for $0 egress; S3 fallback). Two buckets:
  - `uploads/` — user content, 7-day TTL via lifecycle rule. Bucket-level CORS allows browser PUTs.
  - `predictions/` — the `(T, 20484)` arrays (§6.3), float16 NPZ, indefinite TTL. Stage 5 training data lives here.
  - Wire credentials via `R2_ACCESS_KEY`, `R2_SECRET_KEY`, `R2_BUCKET_UPLOADS`, `R2_BUCKET_PREDICTIONS` env vars.
- **Postgres — Supabase** (decided in §6.7). One table is enough for Stage 1:
  ```sql
  create table reports (
    request_id uuid primary key,
    user_id text null,
    goal text not null,
    region_scores jsonb not null,
    overall_score real not null,
    model_version text not null,
    raw_predictions_uri text null,
    created_at timestamptz not null default now()
  );
  ```
- This is **the** schema Stage 2 extends, not replaces. No ORM yet — `psycopg` + raw SQL inside `services/persistence/` is fine for a single table.
- Persistence wiring: after `core/scoring` produces a `BrainReport`, write the `(T, 20484)` array to R2 (returning the URI), then INSERT one row into `reports`. Whole thing is ~30 LOC.

### 5.3 Run locally

- `uvicorn api.main:app --reload` on the Mac
- Hits the deployed RunPod endpoint via HTTPS — no GPU required locally
- Once the strategy doc's Phase 3 lands, this same FastAPI deploys to Railway / Fly unchanged.

### 5.4 End-to-end golden test

- Pin one sample (e.g. the sintel trailer used in `tribev2/run.py`) as a fixture URL
- Run through the full pipeline and snapshot the resulting `BrainReport`
- CI (manual / nightly, since it costs $0.10ish per run) re-runs and diff-checks against the snapshot. Tolerances: scores within ±2 points of snapshot.

Exit for Stage 1: this golden test passes from a fresh `git clone` + `uv sync` + RunPod credentials.

---

## 6. Forward-compatible architecture (discipline, not Stage 1 work)

Stage 1 doesn't build any of what's below — but the repo skeleton, schemas, and module boundaries above are deliberately shaped so Stages 2–6 + Mirofish slot in without rewrites.

### 6.1 Inference engine is the asset; everything else consumes it

- `gpu_worker/inference.py` exposes `predict(...) -> np.ndarray of shape (T, 20484)`. That contract is stable across Stages 1–6. Stage 5 may swap TRIBE v2 for a commercial license or a Cortyze-trained model — same signature, different implementation.
- Don't leak `TribeModel`-specific types (event dataframes, segment objects) above the worker boundary. The API and `core/` only see `np.ndarray`.
- Every response carries `model_version` so when the swap happens, downstream data is forensically traceable.

### 6.2 `core/` is a pure, one-way dependency

- `api/`, `gpu_worker/`, and future `services/` import from `core/`. `core/` imports nothing from them.
- No I/O inside `core/` — no `requests`, no `os.environ`, no DB calls. Pure functions over arrays and pydantic models. This is what makes it trivially callable from a batch worker (Stage 4) or a Mirofish backend (long-term) without dragging FastAPI along.

### 6.3 `request_id` end-to-end, from day 1

- UUID generated at API entry, attached to every log line, every persisted artifact, and the response.
- Stage 4's audience profile builder joins `request_id` ↔ engagement metrics. Stage 5's training pipeline joins `request_id` ↔ raw `(T, 20484)` predictions. Without a stable join key starting now, that data is unreconstructable.
- Persist every prediction's full `(T, 20484)` array to object storage (Cloudflare R2 or S3). At float16 it's ~40 KB/s of content — pennies/month at Stage 1 volume; gold for Stage 5 training data.

### 6.4 Schemas carry forward fields Stage 1 doesn't populate

- `AnalyzeRequest.user_id: Optional[str]` — None in Stage 1, required in Stage 4.
- `BrainReport.model_version: str` — populated from a constant in Stage 1, from a registry in Stage 5.
- `BrainReport.raw_predictions_uri: Optional[str]` — points to the persisted full array.
- Adding optional fields later is free; threading them through every layer in Stage 4 is not.

### 6.5 `Goal` is a typed enum, not a string

- Lives in `core/scoring/goals.py`. Stage 1 uses it for goal mixing. Stage 2's suggestion threshold rules consume the same enum. Stage 4's audience profile weighting consumes it again. Stringly-typed goals will pattern-match-bug their way through every later stage.

### 6.6 Inference is callable, not request-bound

- Wrap RunPod inference behind a plain function `predict_brain_report(req: AnalyzeRequest) -> BrainReport` that the FastAPI handler calls. Stage 4's batch backfill (pull 50–100 past Instagram posts per user) imports the same function and loops. Don't entangle inference with FastAPI's request lifecycle.

### 6.7 Reserved space for Stage 2's bounded contexts

- `services/` exists in the Stage 1 tree as an empty directory + README. Stage 2 adds `services/suggestions/` (Claude calls + threshold rules) and `services/examples/` (reference ad retrieval). Picking the location now means `api/routes/analyze.py` knows where to delegate later — no refactor.
- DB choice is decided but not installed. Recommend **Supabase** (managed Postgres + auth + storage) over raw Railway Postgres: it covers Stage 4 auth and Stage 6 file storage in one provider. Pin the decision before Stage 2 starts; don't add the dependency yet.

### 6.8 Mirofish is a downstream caller, not a peer

- Mirofish renders billboard / sponsorship / retail scenes → MP4 → calls `POST /analyze` with the rendered video. Already supported by Stage 1's API.
- Forward consideration: expose `POST /predict_raw` (returns `(T, 20484)` unmodified, no atlas / scoring / goal mixing) when Mirofish needs spatial analysis the 8-region grouping doesn't capture. Don't build it now — but keep the worker contract clean enough that it's a 30-line addition later.
- Integration surface is HTTPS + API keys. Don't try to merge the codebases — Mirofish has its own Flask backend (per workspace CLAUDE.md).

### 6.9 Frontend lives in a separate repo

- The Next.js app (Stage 3 of the strategy doc) does not nest inside `cortyze_product/`. Vercel deploys cleanest from a repo root; co-mingling Python and a Node toolchain creates double-toolchain pain in CI.
- FastAPI's auto-generated OpenAPI schema is the contract. The frontend imports types via `openapi-typescript` rather than hand-rolling shapes.

---

## 7. Cost estimate (Stage 1 dev + first 100 analyses)

| Item | Estimate |
|---|---|
| Pod (A100 40GB) for ~10 hrs of dev/debug | ~$20 |
| Image storage (RunPod registry) | ~$0 |
| Network volume (50GB) | ~$3/mo |
| 100 analyses on Serverless A100 (avg 15s each) | ~$8 |
| **Stage 1 total** | **~$30–50** |

Spec doc claims $0.03–0.08/analysis. Reality on A100 at ~15s/analysis is closer to **$0.10–0.20**. Flagging — does not change Stage 1, but worth re-quoting in the public landing-page copy before Stage 3.

---

## 8. Open questions / risks

1. **Amygdala has no cortical surface representation.** TRIBE v2 outputs cortical-only fsaverage5 vertices. The spec hand-waves "cortical proxy." Options: (a) drop amygdala from Stage 1 and ship 7 regions, (b) use insula or orbitofrontal as a documented stand-in, (c) ask the team. Recommend (b) with a docstring note; revisit when we have ground-truth ad data.
2. **`neuralset==0.0.2` provenance.** Listed in tribev2 deps; needs to resolve on PyPI from the RunPod image. If it's a private Meta package, we'll need to vendor it or use a wheel mirror. Verify in the first hour of Phase 1.2.
3. **Calibration constants are placeholder-quality.** The 0–100 numbers will look reasonable but aren't grounded until Stage 2's reference ad library exists. Land a `# TODO(stage 2): recalibrate` marker, don't try to perfect this now.
4. **HF gated model token leak.** `tribev2/run.py` has a hardcoded `HF_TOKEN` (per CLAUDE.md). Do **not** copy that pattern into `cortyze_product/`. Use env var + RunPod build secret only.
5. **Mac dev loop for the GPU worker.** We can't run `gpu_worker/inference.py` on the Mac. Mitigation: a `--mock` flag on the API server that returns a pre-saved `(T, 20484)` fixture instead of calling RunPod, so frontend / atlas / scoring work continues during pod downtime.

---

## 9. Out of scope for Stage 1 (do not build now)

- Suggestion engine + Claude integration (Stage 2)
- Reference ad library schema (Stage 2) — note: a minimal `reports` table **is** in Stage 1 (§5.2); the wider data model is not
- Frontend codebase (`cortyze-web/`, separate repo) — Stage 3; backend prerequisites for it land in Stage 1 (§5, §11)
- Auth + account linking (Stage 4) — but anonymous session tracking is in Stage 1's optional `user_id` field (§6.4)
- Audience brain profiling (Stage 4)
- Async job queue (synchronous request + SSE-streamed 30-second response is fine)
- 3D brain mesh visualization (Stage 6) — Stage 3 ships with backend-rendered PNG (§11.2)
- Sentry / Axiom / PostHog / Resend / Ratelimit / CI — all Stage 3+ (§11.3)

---

## 10. Suggested execution order

1. `pyproject.toml` + repo skeleton + `.env.example` (1 hr)
2. `scripts/build_atlas_labels.py` → produces `fsaverage5_dk_labels.npy` (½ day, FreeSurfer download is the slow part)
3. `core/atlas/` + `core/scoring/` + `core/goals/` + tests (2 days, all on Mac)
4. `gpu_worker/` + Dockerfile + first pod boot + sintel smoke test (2 days)
5. `api/` skeleton + RunPod client + `POST /analyze` (sync first) (1 day)
6. CORS + `POST /upload-url` (R2 presigned) + `GET /report/{id}` + `reports` table on Supabase (1 day)
7. SSE-stream the `/analyze` response + persist `(T, 20484)` to R2 (½ day)
8. Convert pod → serverless endpoint, document `make` targets (½ day)
9. Golden end-to-end test (½ day)

**Total: ~8 working days for Stage 1 exit criteria.** (+1 day vs. an engine-only build — buys Stage 3 a clean integration surface; no Stage 1 → Stage 3 refactor.)

---

## 11. Full-stack gaps — what Stage 1 leaves for Stages 3–4

Stage 1 ships a backend engine with the integration hooks Stage 3 needs (CORS, presigned uploads, `GET /report/{id}`, SSE). Below is the punch list of what's still missing to be a deployable web app, grouped by where it lives.

### 11.1 Frontend codebase (`cortyze-web/`, sibling repo to `cortyze_product/`)

Separate Next.js 15 repo. Recommended stack: Tailwind + React Query + types auto-generated from FastAPI's OpenAPI schema via `openapi-typescript`. Hosted on Vercel.

```
cortyze-web/
├── package.json
├── next.config.ts
├── tailwind.config.ts
├── app/
│   ├── page.tsx                   # landing (Stage 3) — waitlist signup
│   ├── scan/page.tsx              # the BrainScore tool — upload → goal → results
│   └── report/[id]/page.tsx       # saved report view (shareable URL)
├── components/
│   ├── upload/                    # file picker + drag-drop, posts to presigned URL from §5.1
│   ├── goal-selector/             # 4-button picker driven by Goal enum
│   ├── brain-map/                 # see §11.2
│   ├── region-card/               # 8 cards, one per region score
│   └── suggestion-card/           # populated in Stage 2
├── lib/
│   ├── api-client.ts              # fetch wrapper with SSE handling
│   ├── api-types.ts               # GENERATED from FastAPI /openapi.json — never hand-edit
│   └── presigned-upload.ts
└── public/
```

**Bootstrap:**

```bash
cd /Users/kirby/Documents/cortyze
npx create-next-app@latest cortyze-web --typescript --tailwind --app --eslint
cd cortyze-web
npm install @tanstack/react-query zod
npm install -D openapi-typescript

# After Stage 1 backend is running locally:
npx openapi-typescript http://localhost:8000/openapi.json -o lib/api-types.ts
```

The frontend imports types from `lib/api-types.ts` so any API breaking change becomes a TypeScript error at build time — keeps the contract honest.

### 11.2 Brain visualization — pragmatic path

Three options, increasing effort:

1. **Backend-rendered PNG** (1 day, ugliest) — `tribev2/plotting/PlotBrain` already exists; have `/analyze` return a base64 PNG alongside the JSON `BrainReport`. Frontend `<img src={`data:image/png;base64,...`} />`. Recommended for Stage 3 launch.
2. **2D SVG heatmap** (3 days, OK) — pre-export a flat brain SVG once (lh + rh outlines with region polygons); D3 fills polygons by score on the client. Tailwind-friendly, instant interactivity. Stage 4 polish.
3. **Three.js + fsaverage5 mesh** (1–2 weeks, gorgeous) — proper 3D rotatable brain, vertex-level coloring. Stage 6 territory.

Ship (1) for the Stage 3 landing demo, iterate to (2) once you have user feedback on what actually matters in the visual.

### 11.3 Cross-cutting product infra (Stage 3+ prerequisites)

| Item | Stage | Notes |
|---|---|---|
| Sentry (errors) | 3 | Both repos |
| Axiom or Logtail (structured logs) | 3 | Sink for the §6.3 `request_id` log lines |
| Upstash Ratelimit | 3 | Free-tier abuse will eat RunPod budget fast |
| PostHog or Plausible | 3 | Landing → scan → result conversion funnel |
| Resend or Postmark | 3 | Waitlist confirmations, report-link emails |
| Anonymous session tracking | 3 | Cookie or device ID feeds into `user_id` (§6.4) |
| Domain + Cloudflare DNS | 3 | `cortyze.com` or similar |
| Vercel deploy (frontend) | 3 | Auto from GitHub `main` |
| Railway / Fly deploy (API) | 3 | Auto from GitHub `main` |
| Supabase prod project | 3 | Separate from dev |
| RunPod prod endpoint | 3 | Separate API key from dev |
| GitHub Actions CI | 3 | `pytest` on backend, `next build` on frontend, both gates on PR |
| Instagram OAuth | 4 | Strategy doc Phase 4 |
| TikTok OAuth | 4 | Strategy doc Phase 4 |
| Background job queue | 4 | Backfill 50–100 past posts per linked user |

### 11.4 Deliberate Stage 1 gaps the frontend will work around

These are **not** in Stage 1, but the Stage 3 frontend ships without them:

- **Account creation / login** — Stage 4. The free preview is anonymous; `user_id` stays null.
- **Saved-report listing per user** — Stage 4. Stage 1's `GET /report/{id}` works by ID only; Stage 3 frontend keeps recent IDs in `localStorage` and shows them as "Your recent scans" until accounts land.
- **Suggestion text** — Stage 2. Frontend renders `BrainReport` without a "suggestions" section until then.
- **Reference ad examples** — Stage 2.

Stage 4's additions are purely additive endpoints (no breaking changes to Stage 1's surface), so the frontend evolves without rewrites.