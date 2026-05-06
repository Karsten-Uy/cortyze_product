# External services setup

Concrete walkthrough for the accounts and API keys you need to wire Cortyze end-to-end. Every service here is **opt-in via env var** — the backend already auto-degrades to mock/synthetic mode when a key is missing, so don't feel obligated to set them all up at once. Work top-to-bottom, the order is roughly cheapest/fastest to most operational.

> **Reference architecture:** see `docs/architecture/architecture_v2.md`. Phases 5 (generative editing) and 6 (PAYG billing) are out of scope for this build, so anything Runway / ElevenLabs / Suno / Stripe in older docs is **skip**.

---

## Quick checklist

- [ ] **Supabase** — auth + Postgres state DB *(required to leave anonymous mode)*
- [ ] **HuggingFace** — TRIBE v2 model weights *(required for real neural inference)*
- [ ] **RunPod** — GPU compute for TRIBE v2 *(required for real neural inference)*
- [ ] **Anthropic** — Claude for suggestion synthesis *(required for real suggestions)*
- [ ] **Cloudflare R2** — media uploads + prediction caching *(required to persist beyond one run)*
- [ ] **Zep Cloud** *(optional, Phase 4)* — agent memory graph for MiroFish
- [ ] **OpenAI / Groq / Cerebras** *(optional, Phase 4)* — fast LLM for the agent swarm
- [ ] **Neo4j AuraDB** *(optional, Phase 2)* — knowledge graph for trend context
- [ ] **Trend data source** *(optional, Phase 2)* — pick one: NewsAPI / Reddit / X / scraping

A v1 release can ship with **only the four required services**: Supabase, HuggingFace, RunPod, Anthropic. Phase 2 (trend context) and Phase 4 (validation swarm) can come later — see open questions in the architecture doc.

---

## 1. Supabase — auth + state DB

**What it does.** Hosts both the user authentication system the frontend already uses (`/login`, `/signup`, `/reset-password`) and the Postgres database that backs `runs`, `region_scores`, `composites`, `suggestions`.

**Required.** Without Supabase, anyone can hit any URL (the auth proxy falls through), and there's no persistence — every run is forgotten on page reload.

**Steps.**
1. Sign up at <https://supabase.com>. Free tier covers ~500 MB Postgres + 50k monthly active users — fine for this build.
2. **New project** → set a strong DB password → wait ~2 min for provisioning.
3. **Project Settings → API** → copy `Project URL` and `anon public` key.
4. **Project Settings → Database → Connection string → URI** → copy the full `postgresql://...` URL.
5. **SQL Editor → New query** → run the schema from `cortyze_product/services/persistence/migrations/` (currently `001_reports.sql`; you'll add a `002_runs_v2.sql` once Phase 1 of the new schema is implemented — see architecture doc §4.1).
6. Add to `cortyze_product/.env`:
   ```
   SUPABASE_URL=https://<ref>.supabase.co
   SUPABASE_ANON_KEY=ey...
   DATABASE_URL=postgresql://postgres:<password>@db.<ref>.supabase.co:5432/postgres
   ```
7. Add to `cortyze_frontend/.env.local`:
   ```
   NEXT_PUBLIC_SUPABASE_URL=https://<ref>.supabase.co
   NEXT_PUBLIC_SUPABASE_ANON_KEY=ey...
   ```

**Cost.** $0 on the free tier until you have real users.

---

## 2. HuggingFace — TRIBE v2 model weights

**What it does.** Hosts `facebook/tribev2` — the gated model checkpoint that does the actual neural encoding in Phase 1.

**Required.** Without it, you can't run real neural inference (mock mode emits synthetic predictions, fine for UI dev but not for an actual product).

**Steps.**
1. Sign up at <https://huggingface.co>.
2. Visit <https://huggingface.co/facebook/tribev2> → click **Agree and access repository** → fill in the gating form. Approval is usually instant.
3. **Settings → Access Tokens → Create new token** → name it `cortyze-prod` → scope **Read** → copy the `hf_...` value.
4. Add to `cortyze_product/.env`:
   ```
   HF_TOKEN=hf_...
   ```

**Cost.** $0. The model weights themselves are free; you're just authenticating to download them.

---

## 3. RunPod — GPU compute for TRIBE v2

**What it does.** Provides the 30+ GB-VRAM GPU TRIBE v2 needs. Cortyze supports two RunPod modes — a long-lived Pod (faster iteration during dev) or a Serverless endpoint (auto-scales, billed per second of GPU time, production default).

**Required for real inference.** Mock mode is `INFERENCE_MODE=mock` (the default) and skips this entirely.

**Steps.**
1. Sign up at <https://www.runpod.io>.
2. Add a payment method. RunPod has no free tier.
3. **Storage → Create Network Volume** in your preferred region (50–100 GB; this caches the TRIBE weights so cold starts don't redownload them).
4. Build the worker image via `cortyze_product/docker/build.sh` and push it to a registry RunPod can pull from (Docker Hub, GHCR).
5. Pick one of the two modes:
   * **Pod (dev):** **Pods → Deploy** → pick an A100 / H100 / L40S → attach your network volume → set `HF_TOKEN` as an env var → start. Note the public HTTP endpoint URL.
     ```
     INFERENCE_MODE=runpod
     RUNPOD_POD_URL=https://<pod-id>-8000.proxy.runpod.net
     ```
   * **Serverless (prod):** **Serverless → New Endpoint** → point it at your image → attach the volume → save. Note the endpoint id and grab an API key from **Settings → API Keys**.
     ```
     INFERENCE_MODE=runpod
     RUNPOD_ENDPOINT_ID=...
     RUNPOD_API_KEY=...
     ```
6. Detailed walkthrough already exists in `cortyze_product/docs/RUNPOD_SESSION.md`. Follow that for the build/push/deploy commands.

**Cost.** A100 80GB ≈ $1.89/hr on-demand, ≈ $0.39/sec on Serverless. Per the benchmark in `docs/runpod_benchmark.md`, a single 30-second clip costs **roughly $0.30–$1.00 in GPU time** depending on configuration. Plan accordingly.

---

## 4. Anthropic — Claude for suggestion synthesis

**What it does.** Generates the structured Suggestion Plan JSON that drives the Results view. This is Phase 3.

**Required for real suggestions.** A mock mode in `services/suggestions/` returns templated suggestions for free.

**Steps.**
1. Sign up at <https://console.anthropic.com>.
2. Add a payment method.
3. **API Keys → Create Key** → name it `cortyze-prod` → copy `sk-ant-...`.
4. Add to `cortyze_product/.env`:
   ```
   ENABLE_SUGGESTIONS=true
   SUGGESTION_LLM_MODE=anthropic
   ANTHROPIC_API_KEY=sk-ant-...
   SUGGESTION_MODEL=claude-sonnet-4-6
   ```
5. **Enable prompt caching** in the suggestion synthesis path. The system prompt + JSON schema together are ~3 KB and the same for every call — caching them puts the per-call input cost near zero. (See `docs/IMPLEMENTATION_PLAN.md` for the existing cache strategy.)

**Cost.** Sonnet 4.6 is roughly $3 / million input tokens and $15 / million output tokens. With prompt caching on the static portion, a typical Suggestion Plan call is **~$0.005–$0.02** depending on how much trend context you ship in. If cost is a concern, route easy cases to Haiku 4.5 (`claude-haiku-4-5-20251001`) and only escalate hard cases to Sonnet/Opus.

---

## 5. Cloudflare R2 — media + prediction storage

**What it does.** Holds the user-uploaded video/image (so the GPU worker can stream it) and caches prediction artifacts so re-rendering Results doesn't require re-running TRIBE.

**Required to ship.** `STORAGE_MODE=off` (default) skips persistence — the run still works, but `/upload-url` returns 503 and predictions vanish at end of request.

**Steps.**
1. Sign up at <https://dash.cloudflare.com>. Add a payment method (R2 has a 10 GB free tier but requires a card).
2. **R2 → Create bucket** twice: `cortyze-uploads` and `cortyze-predictions`.
3. On `cortyze-uploads` set:
   * **CORS rule** allowing `PUT` from your frontend origin (so browser direct-uploads work).
   * **Lifecycle rule** deleting objects after 7 days (you don't need to keep raw uploads forever — predictions are derived).
4. **R2 → Manage R2 API Tokens → Create API token** → scope **Object Read & Write** → copy the **Account ID**, **Access Key**, and **Secret Key**.
5. Add to `cortyze_product/.env`:
   ```
   STORAGE_MODE=r2
   R2_ACCOUNT_ID=...
   R2_ACCESS_KEY=...
   R2_SECRET_KEY=...
   R2_BUCKET_UPLOADS=cortyze-uploads
   R2_BUCKET_PREDICTIONS=cortyze-predictions
   ```

**Cost.** $0 on the 10 GB free tier. Egress is free on R2 (the whole point) — that's a meaningful win over S3 for a media product.

---

## 6. Zep Cloud — MiroFish memory graph (optional)

**What it does.** MiroFish (Phase 4) uses Zep to persist agent memory across simulation rounds. Without it, the validation swarm runs but agents are amnesiac.

**Optional.** Skip this entirely if you're shipping v1 without Phase 4.

**Steps.**
1. Sign up at <https://www.getzep.com>. Free tier exists.
2. **Project → API Keys → Create Key**.
3. Add to `cortyze_product/.env` (or wherever MiroFish is wired in):
   ```
   ZEP_API_KEY=z_...
   ```
4. The MiroFish backend at `cortyze/MiroFish/backend/app/config.py` already validates this on startup — see `cortyze/CLAUDE.md` for that project's setup detail.

**Cost.** $0 on the free tier; ~$0.20 / 1k message-events on paid plans.

---

## 7. Fast LLM for the agent swarm (optional)

**What it does.** Each MiroFish agent runs ~10–50 LLM calls per simulation. You want this fast and cheap — hence Groq, Cerebras, or a local quantized model.

**Optional.** Same as above: skip if you're not running Phase 4 yet.

**Best options.**
* **Groq** (<https://console.groq.com>) — Llama 3.3 70B at ~700 tok/s, cheap. Good default.
* **Cerebras** (<https://cloud.cerebras.ai>) — even faster (~2000 tok/s) on Llama 3.3 70B, similar price. Either works.
* **OpenAI** — fine if you already have an account; gpt-4o-mini is a reasonable budget pick.

**Steps.**
1. Pick one provider, sign up, grab an API key.
2. MiroFish reads `LLM_API_KEY` and `LLM_BASE_URL` (OpenAI-compatible). Set both:
   ```
   LLM_API_KEY=gsk_...
   LLM_BASE_URL=https://api.groq.com/openai/v1
   LLM_MODEL=llama-3.3-70b-versatile
   ```
3. Per `cortyze/CLAUDE.md`, optional `LLM_BOOST_*` vars route hard cases to a stronger model — **omit them entirely** if unused. Leaving placeholder values breaks startup.

**Cost.** A typical 100-agent simulation lands around $0.20–$0.50 on Groq. Order-of-magnitude cheaper than running it on Sonnet.

---

## 8. Neo4j AuraDB — GraphRAG knowledge graph (optional)

**What it does.** Stores the entity / relationship / sentiment graph that Phase 2 queries against. Without it, suggestions are produced from neural scores alone — no cultural grounding, no reference-campaign cards in the Results view.

**Optional.** Skip for v1.

**Steps.**
1. Sign up at <https://neo4j.com/cloud/aura> → AuraDB Free.
2. Create a new instance → save the password it shows you (it's only shown once).
3. Note the connection URI (`neo4j+s://...`).
4. Add to `cortyze_product/.env`:
   ```
   NEO4J_URI=neo4j+s://...
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=...
   ```

**Cost.** Free tier: 200k nodes / 400k relationships, paused after 3 days of inactivity. Fine for dev. Production wants the $65/mo Pro tier.

---

## 9. Trend data source (optional)

**What it does.** Feeds the GraphRAG firehose with current entities, references, and sentiment. Picks one — they're all defensible:

| Source | Strengths | Weaknesses |
|---|---|---|
| **NewsAPI** (<https://newsapi.org>) | Easiest. ~$0/mo dev, $449/mo prod. | Sparse on subcultures; news-shaped only. |
| **Reddit API** | Free up to 100 req/min. Good signal on creator-economy / consumer trends. | Skewed audience; auth dance. |
| **X/Twitter API** | Best raw signal for ad-relevant memes. | $200/mo Basic, $5k/mo for serious volume. |
| **Custom scraping** | Cheapest, most flexible. | Operational cost (proxies, IP bans, parsing). |

For a v1 GraphRAG, **NewsAPI + Reddit** covers most of the surface area at near-zero cost. Add X if/when you have evidence the cultural-grounding signal is mostly on X.

**Steps.** Whichever source you pick, the env-var convention is:
```
TREND_SOURCE=news|reddit|x|scrape
NEWS_API_KEY=...
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
X_BEARER_TOKEN=...
```

**Cost.** Highly variable — see table.

---

## After you have keys

1.  **Backend dev loop:** in `cortyze_product/`:
    ```bash
    cp .env.example .env       # then paste your keys in
    uv sync
    uv run pytest              # 48 tests should pass
    uv run uvicorn api.main:app --reload
    ```
2.  **Frontend dev loop:** in `cortyze_frontend/`:
    ```bash
    cp .env.local.example .env.local   # then paste NEXT_PUBLIC_SUPABASE_*
    npm install
    npm run dev
    ```
3.  Smoke test: hit `http://localhost:8000/health`, then `http://localhost:3000` — sign in via Supabase, click **Run analysis**, watch the result.

If something fails, the backend's `INFERENCE_MODE=mock` / `STORAGE_MODE=off` / `ENABLE_SUGGESTIONS=false` toggles let you peel back layers one at a time to isolate which service is the problem.
