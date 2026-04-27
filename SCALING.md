# Cortyze Scaling & Inference Speedup

How to make Cortyze faster, cheaper, and serve more users. Grounded in the real RunPod benchmark numbers from [docs/runpod_benchmark.md](docs/runpod_benchmark.md), not estimates.

---

## TL;DR

- **Today's reality**: a warm 52 s sintel inference takes **~660 s ± 60 s** on A40 ($0.085/call). A 10 s clip takes **~170 s** (~$0.020/call). 87% of the time is V-JEPA2-Giant video encoding.
- **The cheap, free wins** (frontend cap + visual-only audio skip + `torch.compile`) cut warm latency by ~30%.
- **The real product win** is **V-JEPA feature caching** — splitting inference into "extract once, score N times" makes multi-goal re-runs ~7× faster and is a one-day implementation.
- **Hardware upgrades don't actually save money** — H100 is 2.5× faster but 6× more expensive per hour, so cost-per-call goes up. Only switch GPUs if latency UX matters more than $/call.
- **Scaling out is straightforward**: RunPod Serverless with `min_workers=0`, `max_workers=2-3` handles the projected 100-creators/day load on $3-10/day.

---

## What you should measure before optimizing

If your numbers don't look like the ones in this doc, the optimization advice probably doesn't apply. Before treating this as gospel:

1. **Re-run [docs/runpod_benchmark.md](docs/runpod_benchmark.md)** §4 (warm variance), §5 (duration scaling), §6 (concurrency). 30 minutes, ~$0.25.
2. **Confirm the bottleneck is V-JEPA**. SSH into the pod during inference: `nvidia-smi` should show GPU-Util at 80-100% and the process at ~13 GB VRAM. If it's something else (CPU-bound preprocessing, IO-bound model loading), the levers below are misaligned.
3. **Decide what you're optimizing for.** Latency, throughput, and cost-per-call don't all move together. The rest of this doc is structured by what you're trying to win.

---

## Where the time goes (warm sintel call)

From [docs/runpod_benchmark.md](docs/runpod_benchmark.md) §2 phase breakdown:

| Phase | Time | % of warm | Optimizable? |
|---|---|---|---|
| Audio extract (moviepy) | <1 s | <1 % | No |
| WhisperX transcription | 38 s | 6 % | Yes — skip for visual goals |
| Prepare extractors | 15 s | 2 % | Marginal |
| **V-JEPA2-Giant video encode** | **524 s** | **87 %** | **Yes — biggest lever** |
| Build dataloader + TRIBE forward pass | ~1 s | <1 % | No |
| Other (download, IO) | ~25 s | 4 % | Marginal |

V-JEPA dominates. Anything not targeting V-JEPA is rounding error.

### Cost model

- **Fixed overhead per call**: ~150–180 s (WhisperX, extractor prep, GPU warm-up, IO). Doesn't go down with shorter clips.
- **Per-second-of-content**: ~9–10 s on A40 (V-JEPA scaling).
- **Resolution sensitivity**: 360p clips run ~40% faster than 480p+ clips at the same duration.

So a 5 s clip costs nearly the same as a 10 s clip (~220 s); resolution downscaling is a free 40% speedup; capping clips at 15 s is product-realistic and keeps warm latency under 4 minutes.

---

## Three scaling axes

Don't conflate these. A single change usually only helps one.

### 1. Per-call latency (how long the user waits)

The user clicks "Run BrainScore" and stares at a spinner. Goal: get to result faster.

**Levers**: V-JEPA optimization, faster GPU, shorter clip cap, async-with-progress UX.

**Doesn't help**: more concurrent workers (each call still takes the same time), bigger network volume, more system RAM.

### 2. Throughput (calls served per hour)

Goal: feed more users with the same fleet. Doesn't help any individual user's latency.

**Levers**: more concurrent workers, larger pod fleet, request queuing with SLA promises.

**Doesn't help**: faster GPUs (you'd just pay more for the same throughput), feature caching (helps latency on cache hits, not throughput on cold calls).

### 3. Cost per call

Goal: serve more users on a fixed budget, or improve unit economics for paid tiers.

**Levers**: shorter clips, feature caching, sub-linear pricing tiers (Serverless idle-zero), V-JEPA feature reuse.

**Doesn't help**: H100 (faster but more expensive per call), V-JEPA-Large (requires retraining, big upfront cost).

---

## Speedup options, tiered by effort

Each entry includes: estimated effort, projected speedup vs current warm sintel (660 s), cost change, and which axis it helps.

### Tier 1 — Free wins (do these first)

Combined: warm sintel ~660 s → ~215 s for a 15 s clip in steady state. ~3× effective speedup. Zero infrastructure cost.

#### A. Frontend duration cap at 15 s

- **Effort**: 30 min
- **Effect on latency**: 660 s (52 s clip) → ~227 s (15 s clip predicted)
- **Cost change**: ~$0.085 → ~$0.028/call (3× cheaper)
- **Axis**: latency + cost

Most ads and short-form content are <15 s anyway. Add a hard reject in `cortyze_frontend/lib/limits.ts` with a clear message: "Best for clips under 15 s. Longer clips coming soon." Removes the worst-case 11+ minute scenarios from the product entirely.

#### B. `torch.compile` the V-JEPA encoder

- **Effort**: 1 hour (one-line change + re-deploy)
- **Effect on latency**: ~25-35% reduction on warm calls after first compile
- **Cost change**: same per-second cost; more calls per hour
- **Axis**: latency + throughput

In `gpu_worker/inference.py:39`:
```python
self.model = TribeModel.from_pretrained(...)
# Compile the V-JEPA backbone — first call after deploy is slower (~5 min for compilation),
# every call after gains 25-35% on the encoder pass.
self.model.model = torch.compile(self.model.model, mode="reduce-overhead")
```

Side benefit: stabilizes the variance we saw in §4 (602–717 s spread). Compiled kernels are deterministic; the eager-mode jitter from CUDA kernel selection goes away.

#### C. Skip audio/text path for visual-only goals

- **Effort**: 2 hours
- **Effect on latency**: 53 s saved (38 s WhisperX + 15 s extractor prep)
- **Cost change**: 660 s → ~607 s, ~8% cheaper for visual goals
- **Axis**: latency + cost

Conversion and Awareness goals lean visual; Engagement and Brand Recall lean audio/temporal. Add a conditional in `gpu_worker/inference.py:predict()`:

```python
if goal in (Goal.CONVERSION, Goal.AWARENESS):
    events_df = self.model.get_video_only_events_dataframe(video_path=local_path)
else:
    events_df = self.model.get_events_dataframe(video_path=local_path)
```

Note: this changes scoring behavior. Should be A/B tested against the audio path to confirm no quality regression for visual goals.

#### D. Pre-encode resolution downscale

- **Effort**: 2 hours (ffmpeg subprocess in inference.py)
- **Effect on latency**: ~40% on encoder time (we saw 360p take 124 s vs 214 s for the same-duration 720p clip)
- **Cost change**: matches the speedup
- **Axis**: latency + cost

V-JEPA processes at 256×256 internally anyway; feeding it 1080p doesn't help. Downscale to 360p before passing to TRIBE:

```python
def _downscale_video(src: Path, dst: Path) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-i", str(src),
        "-vf", "scale=-2:360",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-an",  # drop audio if visual-only path
        str(dst),
    ], check=True)
```

Cost of the downscale itself: <1 s on a 30 s clip with `ultrafast` preset. Pure win.

### Tier 2 — One-day investments

Combined with Tier 1: warm latency drops from 660 s → ~85 s effective on cache hits. ~7× effective speedup for the typical multi-goal user flow.

#### E. V-JEPA feature caching

- **Effort**: 1 day
- **Effect on latency**: 660 s → ~80 s on cache hit (skip the entire V-JEPA encode)
- **Cost change**: cache hits cost ~$0.01 instead of $0.085 (~10× cheaper)
- **Axis**: latency + cost

**The biggest single win in this doc.** Most users running BrainScore re-analyze the same content multiple times — different goals, different versions — without ever changing the underlying video. V-JEPA features are independent of goal selection, so they can be cached once per video.

Implementation:

1. Split `gpu_worker/inference.py:predict()` into two functions:
   ```python
   def extract_features(local_path: Path) -> tuple[np.ndarray, "events_df"]:
       """The slow V-JEPA + WhisperX bit. ~600 s on a 52 s clip."""
       ...
   
   def score_features(features: np.ndarray, events_df, goal: Goal) -> np.ndarray:
       """The fast TRIBE head. ~1 s."""
       ...
   ```

2. Add a content-hash cache (SHA-256 of bytes, store features in R2):
   ```python
   def predict(content_url, goal):
       local_path = download(content_url)
       features_key = sha256_file(local_path)
       features = r2.get(f"features/{features_key}.npz") or extract_features(local_path)
       if newly_extracted:
           r2.put(f"features/{features_key}.npz", features)
       return score_features(features, events, goal)
   ```

3. Invalidation is simple: cache is keyed on bytes, so any change to the video changes the key. No staleness possible.

Cache hit rate matters. If a user runs 1 video × 4 goals, that's 1 cold + 3 hot = 4× speedup on the *aggregate* user flow. Even if your real-world hit rate is 50%, you halve your average latency and cost.

Storage cost: each feature blob is ~10-20 MB (T × hidden_dim float16). 1000 videos cached = ~15 GB at $0.015/GB-mo on R2 = $0.22/mo. Negligible.

#### F. Async UX with progress polling

- **Effort**: 1 day (frontend + backend job queue)
- **Effect on latency**: doesn't change actual latency, transforms it from "user staring at spinner" to "user gets notification"
- **Cost change**: $0
- **Axis**: perceived latency

You have 5–10 minute inference times. Stop pretending it's real-time. Pattern:
1. `POST /analyze` returns immediately with a `request_id` and "estimated 6-8 minutes".
2. Frontend polls `/analyze/{id}/status` every 5 seconds (we already verified `/health` stays responsive under load — see [docs/runpod_benchmark.md](docs/runpod_benchmark.md) §7).
3. When done, frontend renders the report.

Bonus: send an email/notification on completion so the user can close the tab and come back. This is what Replicate, Suno, ElevenLabs, etc. all do for long ML jobs. Removes the Cloudflare 100 s proxy timeout headache entirely (the frontend never waits on a single long HTTP call).

### Tier 3 — Hardware swaps

Only worth it once you have paying users complaining about latency.

#### G. H100 PCIe ($2.69/hr, ~2.5× faster on V-JEPA)

- **Effort**: 0 (just change Pod GPU type)
- **Effect on latency**: 660 s → ~265 s warm sintel
- **Cost change**: **2.4× more expensive per call** ($0.085 → $0.20)
- **Axis**: latency only — costs more

H100 is faster, but the speedup is sub-linear in price. Don't switch unless your customers are vocally complaining about latency *and* you're not bottlenecked on the items above.

#### H. H100 SXM ($3.99/hr, ~3× faster)

- **Effort**: 0
- **Effect on latency**: 660 s → ~220 s
- **Cost change**: 3× more expensive per call ($0.085 → $0.27)
- **Axis**: latency only — costs even more

Worse $/call than PCIe. Only use if you need both H100-level speed and SXM's ~50% higher memory bandwidth (we don't — VRAM was 28% utilized).

### Tier 4 — Architectural changes (Stage 5+ territory)

These are research projects, not optimizations. Only feasible once you have user data.

#### I. V-JEPA2-Large variant (300M params instead of 1B)

- **Effort**: 1+ weeks (retrain TRIBE head on Large features, re-validate quality)
- **Effect on latency**: ~3-5× faster encoder (660 s → ~200 s)
- **Cost change**: zero ongoing; substantial up-front retrain cost (~$50-200 in GPU time)
- **Axis**: latency + cost

Meta released V-JEPA2 in both Large (300M) and Giant (1B) variants. TRIBE was trained on Giant features. If a Large-trained TRIBE variant exists or you're willing to retrain it, this is one of the biggest single speedups available. Quality risk: untested without your own validation data.

#### J. Distillation (Stage 5)

- **Effort**: months + ~5K real user runs as training data
- **Effect on latency**: 660 s → 5-15 s (~50-100× speedup)
- **Cost change**: zero ongoing inference; large data-collection prerequisite
- **Axis**: latency + cost (eventually)

Per the [strategy doc](HANDOFF.md#stage-roadmap), Stage 5 is "Cortyze model" — train a smaller student model on the `(content, raw_predictions, engagement)` triples being persisted from day 1. This becomes feasible once you have ~5K paired examples in R2. Don't plan around this for v1; plan for it as the path out of paying NVIDIA every month.

---

## Capacity planning

How many pods/workers to provision based on real numbers from [docs/runpod_benchmark.md](docs/runpod_benchmark.md).

### Per-pod throughput

| Clip length | Warm latency | Pod-hours per call | Calls per pod-hour |
|---|---|---|---|
| 5 s | 222 s | 0.062 | **16** |
| 10 s | 169 s avg | 0.047 | **21** |
| 15 s (predicted) | 227 s | 0.063 | **16** |
| 30 s (predicted) | 401 s | 0.111 | **9** |
| 52 s | 658 s avg | 0.183 | **5.5** |

After Tier 1 optimizations (cap at 15 s + visual-only path + downscale + compile), realistic per-pod throughput on 15 s clips: **~25-30 calls/hour**.

### Load scenarios

| Scenario | Calls/day | Peak calls/hr (8 hr active window) | Pods needed (post-Tier 1) |
|---|---|---|---|
| Demo / dev | 10 | ~2 | 1 |
| Beta (50 creators × 3 analyses) | 150 | ~19 | 1 |
| Stage 3 launch (200 creators × 3) | 600 | ~75 | 3 |
| Stage 4 (1000 creators × 5 backfill avg) | 5000 | ~625 | ~25 |

Stage 4 is when capacity becomes a real cost item. Before that, 1-3 A40 pods cover it for under $30/day.

### Cost model at different volumes

Assuming Tier 1 optimizations + 50% feature cache hit rate (Tier 2 #E):

| Volume | Cost/call (avg) | Daily cost | Monthly cost |
|---|---|---|---|
| 10 calls/day | $0.014 | $0.14 | $4.20 |
| 150 calls/day | $0.014 | $2.10 | $63 |
| 600 calls/day | $0.014 | $8.40 | $252 |
| 5000 calls/day | $0.014 | $70 | $2,100 |

**Note: per-call cost is roughly flat across volumes.** GPU pricing is hourly, not per-request, so as long as your pods are saturated, $/call doesn't drop with scale. The lever is throughput, not pricing tier.

---

## Recommended Phase 7 Serverless config

For RunPod Serverless deployment based on [docs/runpod_benchmark.md](docs/runpod_benchmark.md) §10:

| Setting | Value | Reasoning |
|---|---|---|
| GPU | A40 | Only 13 GB / 46 GB VRAM used. Larger GPUs are more expensive per inference. |
| `min_workers` | 0 | Cold start ~1 min image-pull + ~3 min WhisperX cache warm = acceptable for async UX. Pay $0 when idle. |
| `max_workers` | 2 | Concurrent bursts of 2 finish in ~12 min instead of ~24. Bumping to 3+ helps only if traffic is truly bursty. |
| Idle timeout | 300 s | Standard. Long enough to keep workers warm during a flurry, short enough to stop bleeding money. |
| Execution timeout | 1200 s (20 min) | Real-world warm sintel hit 717 s. 1200 s gives a 65% safety margin for jitter spikes. |

Bump `min_workers` to 1 only after you have enough sustained traffic that cold-start UX hurts retention.

---

## Decision tree: which lever for which problem

```
"My users wait too long" → A (frontend cap), F (async UX with progress), B (torch.compile), E (feature cache)
"Inference is too expensive" → A (cap), C (visual-only), D (downscale), E (cache)
"GPU bill is climbing with growth" → E (cache hit rate), A (cap), Phase 7 Serverless `min_workers=0`
"Need to serve more concurrent users" → bump `max_workers`, possibly multiple pod regions
"Need <60 s latency for live demos" → G (H100 PCIe) + B (compile) + C (skip audio) + D (downscale) — even then ~150 s is the realistic floor on Giant
"Need <10 s latency for true real-time" → I (V-JEPA-Large retrain) or J (distillation) — months-scale projects
```

Most product needs are solved by Tier 1 + Tier 2. Tier 3 and Tier 4 are escape valves for specific constraints, not default investments.

---

## Out of scope here

These adjacent topics live in other docs:

- **RunPod deployment mechanics**: [RUNPOD_SESSION.md](RUNPOD_SESSION.md) — image build, network volume setup, Pod vs Serverless conversion.
- **Per-test results**: [docs/runpod_benchmark.md](docs/runpod_benchmark.md) — raw numbers, phase breakdown, scaling fit.
- **Stage 5 distillation roadmap**: [HANDOFF.md](HANDOFF.md) §"Stage roadmap" — when to invest in own-model training.
- **Frontend latency UX patterns**: not yet documented; needed before async-UX lever (#F) ships.

---

## Anti-patterns: don't do these

Things that look like optimizations but aren't:

- **Bigger network volume**. We're not bottlenecked on disk; the volume is for weight caching, not runtime IO.
- **More system RAM**. The pod has 503 GB and uses 65 GB. RAM is not the constraint.
- **Bigger VRAM** (A100 80GB, B200). Current usage is 13 GB / 46 GB. More VRAM = more expensive idle capacity.
- **Higher batch sizes**. TRIBE doesn't batch single requests. The encoder is sequential per video.
- **Pre-baking model weights into the Docker image**. Image size balloons to 40 GB, pulls become slow, defeats the whole point of the network volume.
- **Caching the full BrainReport per content+goal**. Looks tempting, but forecloses on per-call goal customization, A/B testing, and Stage 4's per-creator weight overrides. Cache features (Tier 2 #E), not reports.
