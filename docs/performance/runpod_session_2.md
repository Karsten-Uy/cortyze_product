# Inference Runtime Analysis — `run_20260502_012937`

**Run date:** 2026-05-02 · **GPU:** NVIDIA A40 48 GB · **Model:** TRIBE v2 (facebook/tribev2)
**Mode:** `goal=engagement · compress=360p · max_duration=60s · caption=on`

---

## Summary

| Metric | Value |
|---|---|
| Total reels in job | 174 |
| Reels seen in log (log truncated at reel 102) | 102 |
| Completed (inference ran + `[done]`) | **77** |
| Skipped (exceeded 60 s max_duration cap) | 12 |
| In-flight / log cut off | 13 |
| Model cold-start (load to `Model ready`) | **4 s** |

> **Note:** The log ends mid-run at reel [102/174]. All statistics below reflect the 77 reels that completed inference.

---

## Inference Latency (elapsed wall time, all 77 completed reels)

| Stat | Value |
|---|---|
| Min | 77.4 s |
| P50 (median) | 276.2 s |
| Mean | **298.1 s** |
| P90 | 526.2 s |
| P95 | 570.9 s |
| Max | 626.1 s |

> **Mean is ~8% higher than P50**, indicating the distribution has a moderate right tail driven by longer clips. At 60 s (the cap), elapsed reaches 600–626 s, confirming latency is nearly linear with clip length.

---

## Latency by Video Duration Bucket

| Clip length | n | Avg elapsed | Min elapsed | Max elapsed | Avg ratio |
|---|---|---|---|---|---|
| < 10 s | 14 | 121.5 s | 77.4 s | 140.8 s | ~15× real-time |
| 10 – 20 s | 17 | 185.2 s | 137.5 s | 231.2 s | ~13× real-time |
| 20 – 30 s | 17 | 283.8 s | 238.1 s | 339.1 s | ~12× real-time |
| 30 – 45 s | 17 | 397.8 s | 325.2 s | 468.0 s | ~11× real-time |
| 45 – 60 s | 12 | 543.1 s | 482.2 s | 626.1 s | ~11× real-time |

**Key observation:** There is a fixed overhead of approximately **70–80 s** per reel regardless of clip length (evident from the 4.1 s clip completing in 77.4 s). This baseline covers audio extraction, whisperx transcription, caption embedding, and model setup per request. The variable component scales at roughly **10–11× real-time** for clips ≥ 20 s, flattening from the higher ratio seen on very short clips (where fixed overhead dominates).

---

## Skipped Reels (exceeded 60 s cap)

12 reels were downloaded but not processed because they exceeded the `max_duration=60s` limit:

| Reel # | Duration |
|---|---|
| 19 | 83.8 s |
| 23 | 132.2 s |
| 28 | 168.7 s |
| 31 | 80.8 s |
| 33 | 70.9 s |
| 66 | 72.7 s |
| 80 | 102.1 s |
| 82 | 139.4 s |
| 86 | 101.4 s |
| 95 | 124.1 s |
| 99 | 60.1 s* |
| 102 | 10.5 s** |

> *Reel 99 (60.1 s) barely exceeded the cap — worth revisiting if the cap is raised slightly.
> **Reel 102 (10.5 s) was likely in-flight when the log was cut off; it would not normally be skipped.

---

## Overall BrainScore Distribution

Scores range from **36.8 to 73.6** with a mean of **56.7**, tightly clustered in the mid-band:

| Score band | Count | Share |
|---|---|---|
| < 45 | 5 | 6% |
| 45 – 55 | 29 | 38% |
| 55 – 65 | 30 | 39% |
| 65 – 75 | 13 | 17% |

76% of reels score between 45–65 — a healthy spread for an engagement model. The top-scoring reel was **reel 100 (73.6)** at 24.5 s duration; the lowest was **reel 10 (36.8)** at 10.0 s.

---

## Cost Estimate

Based on 77 completed inferences at A40 pricing ($0.44 / hr):

| Metric | Value |
|---|---|
| Total inference wall time | 22,955 s ≈ **6.38 hr** |
| Estimated GPU cost (77 reels) | **$2.81** |
| Cost per reel (avg) | **$0.036** |
| Projected cost for all 174 reels | ~**$6.35** (extrapolated) |

> Cost per reel is well within the original $5–7 session budget. Extrapolating to the full 174-reel job at the observed mean elapsed time yields ~$6.35 — within budget and consistent with the checklist estimate.

---

## Bottleneck Breakdown (qualitative, from log)

Based on tqdm progress bars and timestamps in the log, per-reel time is dominated by:

1. **V-JEPA2 video encoding** — ~4.6 s per chunk, accounts for the bulk of variable latency. A 60 s clip produces ~92 chunks → ~7 min of encoding alone.
2. **Whisperx transcription** — ~24–31 s flat overhead per reel, independent of clip length (explains the ~70–80 s fixed base).
3. **TRIBE v2 prediction** — fast once encodings are ready (seconds, not minutes).
4. **Audio extraction (MoviePy)** — negligible (<1 s).

---

## Recommendations

**For UX:** A 60 s clip at P50 takes ~543 s (~9 min). This is incompatible with synchronous interactive UX. Recommended actions:

- **Cap clips at 30 s** in the product UI — reduces P50 latency to ~325–400 s and P95 to ~468 s.
- **Use async UX** with a job queue and push notification rather than a blocking spinner.
- **Consider a smaller V-JEPA variant** if sub-90 s warm latency is required — V-JEPA2-Base would dramatically cut video encoding time.

**For scaling:** At ~$0.036/reel on A40 serverless, the cost model is acceptable. With `max_workers=2` and 10–11× real-time processing, a single worker handles ~5–6 reels/hr at the current mean duration. For bursts, auto-scaling to 2 workers doubles throughput with no code changes.

**For the cap:** The 60 s `max_duration` is appropriate operationally. Raising it to 90 s would recover the 12 skipped reels but would push max elapsed toward ~900–1000 s — expose that only on batch/async paths, not interactive.

---

*Report generated from `run_20260502_012937.log` · 77 completed inferences · GPU: A40 48 GB · Model: TRIBE v2*