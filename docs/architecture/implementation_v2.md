# Neuro-Social Content Engine — Implementation Plan (Stages 2–4)

**Repo:** `cortyze_product`
**Baseline:** Stage 1 is complete — TRIBE v2 inference via RunPod, `core/` atlas/scoring/goals, FastAPI (`/analyze`, `/upload-url`, `/report/{id}`), Supabase `reports` table, Cloudflare R2 storage.

**Goal of this document:** Map the three remaining architecture phases (GraphRAG, Claude synthesis, MiroFish simulation) onto the existing codebase with zero rewrites to Stage 1.

---

## How the New Phases Plug Into Stage 1

Stage 1 already does the heavy lifting — it produces a `BrainReport` with `request_id`, `region_scores`, `overall_score`, and persists the raw `(T, 20484)` array to R2 + a row to Supabase. The new phases are **purely additive**:

- Phase 2 (GraphRAG) runs in parallel with TRIBE inference and writes `trend_context` to the existing `reports` table (new JSONB column).
- Phase 3 (Claude synthesis) is already stubbed behind `ENABLE_SUGGESTIONS=true`; it reads both outputs from the DB and populates `services/suggestions/`.
- Phase 4 (MiroFish) runs after Phase 3 and reads from the same DB row.

The state handoff DB called for in the architecture doc is the **Supabase `reports` table you already have.** No new database needed.

```
┌─────────────────────────────────────────────────────────────┐
│  POST /analyze (existing)                                    │
│    │                                                         │
│    ├──► RunPod TRIBE (Phase 1, existing) ──► neuro_metrics  │
│    │                                              │          │
│    └──► GraphRAG Worker (Phase 2, NEW) ──► trend_context    │
│                                              │    │          │
│                     Supabase reports table ◄─────┘          │
│                              │                               │
│                         Claude API (Phase 3, stub→real)     │
│                              │                               │
│                         MiroFish (Phase 4, NEW)             │
│                              │                               │
│                    GET /report/{id} (existing)               │
└─────────────────────────────────────────────────────────────┘
```

---

## Stage 2 — GraphRAG Social Context Pipeline

**Effort:** ~5 days
**Exit criteria:** Every `/analyze` call produces a `trend_context` JSON blob alongside `neuro_metrics`, both stored in Supabase. Claude synthesis reads both.

### 2.1 New files

```
services/
└── social_context/
    ├── __init__.py
    ├── scraper.py          # firehose ingestion (Reddit, Twitter/X, Google Trends)
    ├── graph_builder.py    # NetworkX → Neo4j sync
    ├── query.py            # extract trending entities/sentiment for a given video
    └── schemas.py          # TrendContext pydantic model
```

### 2.2 Schema additions

Add two columns to the existing `reports` table:

```sql
-- Migration: 002_trend_context.sql
ALTER TABLE reports
  ADD COLUMN trend_context  jsonb   NULL,
  ADD COLUMN suggestions    jsonb   NULL,
  ADD COLUMN simulation     jsonb   NULL,
  ADD COLUMN pipeline_stage text    NOT NULL DEFAULT 'neuro_only';
  -- stages: neuro_only | neuro+trend | neuro+trend+suggestions | complete
```

The `pipeline_stage` column lets `GET /report/{id}` tell the frontend exactly how far along a job is — no polling ambiguity.

### 2.3 The knowledge graph

**Stack choice — start with NetworkX, graduate to Neo4j:**

For dev, use an in-process `networkx.DiGraph` stored as a pickle/JSON on disk (or in Supabase's JSONB column). Promote to a hosted Neo4j AuraDB free tier once entity count exceeds ~50k nodes. The `graph_builder.py` interface is the same either way — this is a one-line swap in `__init__.py`.

**Graph structure:**

```
(Entity: "Nike") --[SENTIMENT: 0.72]--> (Topic: "running shoes")
(Entity: "Nike") --[TRENDING_ON: 2026-05-03]--> (Platform: "TikTok")
(Topic: "running shoes") --[CO-OCCURS_WITH]--> (Topic: "marathon")
```

**Ingestion sources (rolling 48h window):**

| Source | Library | What you get |
|--------|---------|-------------|
| Reddit | `praw` | Post titles, scores, subreddit sentiment |
| Google Trends | `pytrends` | Rising queries, related topics |
| Twitter/X | `tweepy` (Basic tier) | Trending hashtags, sentiment via VADER |
| News | `newspaper3k` or `newsapi.org` | Entity extraction via `spaCy` |

`scraper.py` runs as a background APScheduler job (already a natural fit given the FastAPI server), refreshing every 30 minutes. Entity extraction uses `spaCy en_core_web_sm` — no GPU.

**`query.py:get_trend_context(video_metadata: dict) -> TrendContext`:**

```python
# Given: transcript, brand mentions, content_type, goal
# 1. Extract named entities from transcript via spaCy
# 2. Query graph: for each entity, pull connected sentiment nodes + trending edges
# 3. Score each entity: trend_velocity, sentiment_polarity, risk_flag
# Return: TrendContext(entities=[...], dominant_sentiment=..., risk_score=...)
```

**`TrendContext` schema:**

```python
class EntityContext(BaseModel):
    name: str
    trend_velocity: float      # 0–1, relative to 48h baseline
    sentiment_polarity: float  # -1 to 1
    sarcasm_flag: bool         # True if dominant framing is ironic
    platform_peaks: list[str]  # ["TikTok", "Reddit"]

class TrendContext(BaseModel):
    entities: list[EntityContext]
    dominant_topic: str
    brand_risk_score: float    # 0–1
    cultural_moment: str | None  # e.g. "post-Super Bowl ad cycle"
    snapshot_timestamp: datetime
```

### 2.4 Wiring into `/analyze`

GraphRAG runs **concurrently with RunPod inference** using `asyncio.gather`. TRIBE takes 3–12 minutes; GraphRAG takes ~2 seconds. There is no latency cost:

```python
# api/routes/analyze.py  (additions to existing handler)
async def analyze(req: AnalyzeRequest):
    tribe_task = asyncio.create_task(runpod_client.predict(req))
    trend_task = asyncio.create_task(get_trend_context(req))  # NEW

    brain_preds, trend_ctx = await asyncio.gather(tribe_task, trend_task)

    brain_report = build_report(brain_preds, req)
    await db.update_report(req.request_id, trend_context=trend_ctx.dict())
    # ... existing persistence
```

The SSE stream gains a new event: `{"event": "trend_context_ready", "data": {...}}` — the frontend can render the social context panel immediately while TRIBE is still running.

### 2.5 Tests

- `tests/test_social_context.py`: mock graph with 3 entities → assert `TrendContext` keys present, velocity bounded 0–1, sarcasm flag propagates.
- `tests/test_analyze_parallel.py`: mock both RunPod and GraphRAG → assert both complete, DB row has non-null `trend_context`.

---

## Stage 3 — Claude Synthesis Engine

**Effort:** ~3 days (the stub is already wired; this is filling it in)
**Exit criteria:** With `ENABLE_SUGGESTIONS=true` + `SUGGESTION_LLM_MODE=anthropic`, `/analyze` returns a `suggestions` array with ≥3 specific edits grounded in both neuro and trend data.

### 3.1 Activate the existing stub

The existing `services/suggestions/` directory (currently empty) gets:

```
services/
└── suggestions/
    ├── __init__.py
    ├── synthesizer.py      # Claude API call + prompt assembly
    ├── prompt_templates.py # structured prompt builder
    └── schemas.py          # Suggestion, SuggestionSet pydantic models
```

### 3.2 Prompt architecture

The prompt assembler in `prompt_templates.py` builds a structured prompt from three inputs: the user's campaign context, the `BrainReport`, and the `TrendContext`. It is deliberately template-driven so prompt variants can be A/B tested without touching code:

```
SYSTEM: You are a content strategist for video advertising. You analyze neurological
engagement data and social trend data to produce specific, actionable editing suggestions.
Always output valid JSON matching the SuggestionSet schema below.

[SCHEMA]
{
  "suggestions": [
    {
      "id": "s1",
      "timestamp_range": [5.0, 8.5],   // seconds in the original video
      "region_implicated": "prefrontal",
      "type": "pacing" | "script" | "visual" | "audio" | "cta",
      "finding": "Attention drops sharply here (prefrontal score: 23/100).",
      "action": "Cut 2 seconds from the product reveal; lead with the problem statement.",
      "trend_rationale": "Topic is trending on TikTok but in a sarcastic register — reframe tone.",
      "confidence": 0.87
    }
  ],
  "overall_strategy": "string",
  "risk_flags": ["string"]
}

USER:
Campaign goal: {goal}
Brand context: {brand_context}

TRIBE neuro-metrics:
{neuro_metrics_json}    ← brain_report.region_scores, timestamped activation dips

Social trend context:
{trend_context_json}    ← TrendContext.entities, sentiment, risk_score
```

**Key prompt design decisions:**
- Timestamped activation dips from the raw `(T, 20484)` array (already in R2) are pre-computed into a `[{t: seconds, score: float}]` series and injected — this gives Claude specific temporal anchors, not just averages.
- `sarcasm_flag=True` in any entity triggers a mandatory "tone alignment" suggestion.
- `brand_risk_score > 0.7` forces a `risk_flags` entry.

### 3.3 `synthesizer.py`

```python
async def generate_suggestions(
    brain_report: BrainReport,
    trend_ctx: TrendContext,
    campaign_context: str,
) -> SuggestionSet:
    prompt = build_prompt(brain_report, trend_ctx, campaign_context)
    response = await anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )
    return SuggestionSet.model_validate_json(response.content[0].text)
```

This runs after `asyncio.gather` completes in the `/analyze` handler — it depends on both `brain_report` and `trend_ctx` being ready. The DB orchestration check from the architecture doc is naturally enforced by Python's `await` chain; no separate polling loop is needed at Stage 3 volume.

### 3.4 `SUGGESTION_LLM_MODE` behavior (extending the existing env var)

| Mode | Behavior |
|------|---------|
| `mock` (default, existing) | Returns templated suggestions; $0, deterministic |
| `anthropic` | Real Claude API via `anthropic` SDK |
| `openai_compatible` | Any OpenAI-format endpoint (Groq for cheaper iteration) |

### 3.5 DB and API changes

- `reports.suggestions` JSONB column (added in migration `002` above) is populated.
- `pipeline_stage` advances to `neuro+trend+suggestions`.
- `GET /report/{id}` response now includes `suggestions` array.
- `BrainReport` schema gains `suggestions: list[Suggestion] | None` (optional — backwards compatible with Stage 1 clients that don't consume it).
- New SSE event: `{"event": "suggestions_ready", "data": {...}}`.

### 3.6 Tests

- `tests/test_suggestions.py`: fixture `BrainReport` + fixture `TrendContext` → mock Claude API → assert `SuggestionSet` schema valid, all suggestions have `timestamp_range`, sarcasm entity triggers tone suggestion.
- Prompt round-trip test: raw prompt string contains the neuro-score values and entity names from fixtures (regression against prompt template drift).

---

## Stage 4 — MiroFish Validation Swarm

**Effort:** ~6 days
**Exit criteria:** After suggestions are generated, a multi-agent simulation runs and produces a `SimulationReport` comparing baseline vs. suggested-edit performance on projected virality, sentiment, and brand risk.

### 4.1 New files

```
services/
└── simulation/
    ├── __init__.py
    ├── agent_factory.py    # builds agent personas from TrendContext + BrainReport
    ├── swarm.py            # orchestrates N agent calls, aggregates results
    ├── ab_test.py          # baseline vs. suggestions comparison logic
    └── schemas.py          # SimulationReport, AgentPersona, SimResult
```

```
api/routes/
└── simulate.py             # POST /simulate (new optional endpoint)
```

### 4.2 Agent persona generation

The key insight from the architecture doc is that agent biases are **derived from your existing data**, not randomly assigned. `agent_factory.py` builds `AgentPersona` objects by combining `TrendContext` and `BrainReport`:

```python
class AgentPersona(BaseModel):
    persona_id: str
    platform_affinity: list[str]  # ["TikTok", "Instagram"] — from TrendContext.platform_peaks
    sentiment_prior: float        # from TrendContext.dominant_sentiment
    attention_threshold: float    # from BrainReport.region_scores["prefrontal"] — low = scrolls faster
    sarcasm_sensitivity: float    # agents more likely to mock content with sarcasm_flag
    goal_alignment: Goal          # same Goal enum from core/scoring/goals.py
```

`agent_factory.build_swarm(n=100, brain_report, trend_ctx) -> list[AgentPersona]` generates a population whose distribution mirrors the trend data — e.g. if `brand_risk_score` is high, 20% of agents are seeded with adversarial priors.

### 4.3 The simulation loop

Each agent makes a single LLM call (fast/cheap model — Groq `llama-3.1-8b-instant` at ~$0.0001/call, so 100 agents ≈ $0.01). The call presents the agent with a content summary and asks for a structured reaction:

```python
# swarm.py
async def run_agent(persona: AgentPersona, content_summary: str) -> SimResult:
    prompt = f"""
    You are a social media user. Persona: {persona.model_dump_json()}
    You just saw this content: {content_summary}
    Respond ONLY as JSON: {{"action": "share"|"like"|"scroll"|"comment",
                            "sentiment": -1..1, "comment_text": str|null}}
    """
    # ... LLM call, parse response
    return SimResult(persona_id=persona.persona_id, action=..., sentiment=...)

async def run_swarm(personas, content_summary) -> list[SimResult]:
    return await asyncio.gather(*[run_agent(p, s) for p, s in zip(personas, [content_summary]*len(personas))])
```

### 4.4 A/B test structure

`ab_test.py` runs the swarm **twice** — once for the original content, once for the suggested-edit version — and diffs:

```python
async def ab_test(
    brain_report: BrainReport,
    trend_ctx: TrendContext,
    suggestions: SuggestionSet,
    n_agents: int = 100,
) -> SimulationReport:
    personas = agent_factory.build_swarm(n_agents, brain_report, trend_ctx)

    baseline_summary = build_content_summary(brain_report, suggestions=None)
    improved_summary = build_content_summary(brain_report, suggestions=suggestions)

    baseline_results, improved_results = await asyncio.gather(
        run_swarm(personas, baseline_summary),
        run_swarm(personas, improved_summary),
    )

    return SimulationReport(
        baseline=aggregate(baseline_results),
        improved=aggregate(improved_results),
        delta=compute_delta(baseline_results, improved_results),
    )
```

`build_content_summary` converts the `BrainReport` + optional `SuggestionSet` into a plain-language description of the content as it would appear — this is what the agents react to, not raw JSON scores.

### 4.5 `SimulationReport` schema

```python
class SwarmMetrics(BaseModel):
    share_rate: float         # 0–1, fraction of agents who shared
    like_rate: float
    scroll_rate: float        # "failed to engage"
    mean_sentiment: float     # -1 to 1
    virality_score: float     # composite: share_rate * (1 - scroll_rate) * (1 + mean_sentiment)
    brand_risk_flag: bool     # True if >15% of comments are negative

class SimulationReport(BaseModel):
    request_id: str
    n_agents: int
    baseline: SwarmMetrics
    improved: SwarmMetrics
    delta: dict[str, float]   # e.g. {"virality_score": +0.23, "brand_risk_flag": False}
    significant: bool         # True if virality delta > 0.1 (worth showing to client)
    agent_comments_sample: list[str]  # 5 representative comments from each condition
```

### 4.6 Wiring into the pipeline

MiroFish runs **after suggestions** — it is not on the critical path of `/analyze`. Two options:

**Option A — Inline (simple, acceptable for Stage 4):** extend the `/analyze` handler with `await ab_test(...)` after `generate_suggestions`. At 100 agents × Groq latency (~300ms each but parallelized), the simulation adds ~3–5 seconds wall time.

**Option B — Background job (better UX):** after writing `suggestions` to the DB, enqueue a background task (FastAPI's `BackgroundTasks` or APScheduler). The SSE stream sends a `{"event": "simulation_queued"}` immediately; the client polls `GET /report/{id}` for `pipeline_stage == "complete"`.

Recommend **Option B** for production — simulation is a bonus insight, not a blocker for the dashboard to load.

### 4.7 New API surface

```
POST /simulate
  body: { request_id: str, n_agents?: int }  # re-run simulation on an existing report
  → 202 Accepted + SSE stream → SimulationReport

GET /report/{id}  (existing, extended)
  → BrainReport now includes simulation?: SimulationReport | null
```

### 4.8 Tests

- `tests/test_simulation.py`: fixture `BrainReport` + `TrendContext` + `SuggestionSet` → mock Groq → assert `SimulationReport` keys present, `delta` keys match `SwarmMetrics` fields, `significant` flag logic correct.
- `tests/test_ab_test.py`: baseline and improved summaries are distinct strings, `improved` virality ≥ baseline (with mocked LLM favouring suggestions).

---

## Full Repo Tree After All Stages

```
cortyze_product/
├── core/                   (Stage 1, unchanged)
│   ├── atlas/
│   ├── scoring/
│   └── schemas.py          ← BrainReport gains suggestions?, simulation?
├── api/
│   ├── main.py
│   └── routes/
│       ├── analyze.py      ← extended with trend+suggestions+simulation hooks
│       ├── simulate.py     ← NEW (Stage 4)
│       ├── health.py
│       └── upload.py
├── gpu_worker/             (Stage 1, unchanged)
├── services/
│   ├── social_context/     ← NEW (Stage 2)
│   │   ├── scraper.py
│   │   ├── graph_builder.py
│   │   ├── query.py
│   │   └── schemas.py
│   ├── suggestions/        ← STUB→REAL (Stage 3)
│   │   ├── synthesizer.py
│   │   ├── prompt_templates.py
│   │   └── schemas.py
│   ├── simulation/         ← NEW (Stage 4)
│   │   ├── agent_factory.py
│   │   ├── swarm.py
│   │   ├── ab_test.py
│   │   └── schemas.py
│   ├── storage/r2.py       (Stage 1)
│   └── persistence/
│       ├── reports.py      (Stage 1)
│       └── migrations/
│           ├── 001_reports.sql
│           └── 002_trend_context.sql   ← NEW
├── scripts/
│   └── build_atlas_labels.py
└── tests/
    ├── test_atlas.py
    ├── test_scoring.py
    ├── test_goals.py
    ├── test_api.py
    ├── test_social_context.py   ← NEW
    ├── test_suggestions.py      ← NEW
    └── test_simulation.py       ← NEW
```

---

## New Environment Variables

```bash
# Stage 2 — GraphRAG
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
TWITTER_BEARER_TOKEN=...
NEWSAPI_KEY=...               # optional
NEO4J_URI=...                 # optional; falls back to networkx in-process
NEO4J_PASSWORD=...

# Stage 3 — Suggestions (existing stub, now populated)
ENABLE_SUGGESTIONS=true
SUGGESTION_LLM_MODE=anthropic    # mock | anthropic | openai_compatible
ANTHROPIC_API_KEY=...

# Stage 4 — Simulation
ENABLE_SIMULATION=true
SIMULATION_LLM_MODE=groq         # groq | openai_compatible | anthropic
GROQ_API_KEY=...
SIMULATION_N_AGENTS=100
```

---

## New Python Dependencies

```toml
# pyproject.toml additions
[project.dependencies]
# Stage 2
praw = ">=7.8"
pytrends = ">=4.9"
tweepy = ">=4.14"
spacy = ">=3.8"
networkx = ">=3.3"
apscheduler = ">=3.10"
vaderSentiment = ">=3.3"
neo4j = ">=5.20"          # optional, only if NEO4J_URI is set

# Stage 3
anthropic = ">=0.30"      # already likely present as test dep

# Stage 4
groq = ">=0.9"            # Groq SDK for cheap agent calls
```

After adding spaCy: `python -m spacy download en_core_web_sm` (add to Dockerfile and dev setup docs).

---

## Cost Estimate

| Component | Per analysis | Notes |
|-----------|-------------|-------|
| TRIBE v2 (RunPod A100) | $0.10–0.20 | Stage 1, existing |
| GraphRAG query | ~$0 | In-process; scraping is shared background cost |
| Claude synthesis | ~$0.003 | ~1500 tokens @ Sonnet pricing |
| MiroFish (100 agents, Groq) | ~$0.01 | 100 × `llama-3.1-8b-instant` calls |
| **Total** | **~$0.15–0.22** | Per analysis end-to-end |

Background scraping (APScheduler, every 30min) costs ~$0/mo in compute — just API rate limits to manage.

---

## Execution Order

1. **DB migration** — `002_trend_context.sql` on Supabase dev project (1hr)
2. **`services/social_context/`** — scraper stubs + networkx graph + `get_trend_context()` + tests (2 days)
3. **Wire GraphRAG into `/analyze`** — parallel `asyncio.gather`, SSE event, DB write (½ day)
4. **`services/suggestions/`** — prompt templates + synthesizer + `anthropic` mode + tests (1.5 days)
5. **Wire suggestions** — post-gather handler, DB write, `BrainReport` schema extension (½ day)
6. **`services/simulation/`** — agent factory + swarm + A/B test + tests (2.5 days)
7. **Wire simulation** — background task, `POST /simulate`, `pipeline_stage` tracking (1 day)
8. **End-to-end integration test** — one real video through all four phases (½ day)

**Total: ~9 working days** to full Stage 4 completion.