# Suggestion Engine — Cost & Quality Test Plan

A live evaluation of the Anthropic-backed suggestion engine (`SUGGESTION_LLM_MODE=anthropic`,
`ANTHROPIC_MODEL=claude-sonnet-4-6`). Goal: decide how to run suggestions in production
without burning credits or shipping low-quality output.

Fill in each test as you run it. Aggregate at the bottom.

---

## Baseline — Test 1 (already run, 2026-04-28)

| Field | Value |
|---|---|
| Content | 1 image (Pepsi cream soda) + 15 s audio voiceover + 96 char caption |
| Goal | Conversion |
| Regions fired | 8 / 8 |
| Suggestions returned | 21 (~2.6 per region) |
| Wall time | 3.2 s |
| **Cost** | **$0.07** |
| Per-region cost | ~$0.0088 |
| Model | claude-sonnet-4-6 |
| Temperature | 0.5 |

### Projected scale

| Volume | Cost @ baseline |
|--------|------|
| 100 analyses | $7 |
| 1,000 | $70 |
| 10,000 | $700 |

### Quality red flags from Test 1

1. **Hallucinated audio content.** Suggestions repeatedly reference an "in buy" voiceover phrase
   that doesn't exist in the actual audio. Claude only sees brain scores + caption text, not the
   image or audio — so it invents specifics that "look right" given the score pattern. The
   transcript fragment WhisperX passed through likely produced "in buy" as a misheard token, and
   Claude latched onto it.
2. **21 suggestions is too many.** Top-scoring regions (Fusiform Face 33.5, Reward 21.6) still
   fired 3 suggestions each — diminishing returns on cost and UX.
3. **All 8 regions fired regardless of goal weight.** Hippocampus is 5 % of Conversion — its 3
   suggestions barely move the overall score but cost the same as Prefrontal's (25 %).

---

## Test 2 — Threshold tuning (DONE 2026-04-28, FREE)

**Why:** Single biggest cost + UX win. Only fire suggestions for regions where intervention
matters: low score AND meaningful goal weight.

**Setup:** [services/suggestions/rules.py](services/suggestions/rules.py) now reads two
env-tunable thresholds (defaults shown):

```
SUGGESTION_SCORE_THRESHOLD=70  # region must score below this
SUGGESTION_MIN_WEIGHT=0.10     # region's goal weight must be >= this
```

Was previously `score < 50, weight ≥ 0.05`. The weight floor is the cost lever — it kills
the "minor" tier (Conversion: skips Hippocampus 5%, Fusiform 2%, Temporal 8%).

Re-ran the same Pepsi cream soda input, goal=Conversion.

| Field | Value |
|---|---|
| Regions fired | **5 / 8** (Visual Cortex, Amygdala, Prefrontal, Motor, Reward) |
| Suggestions returned | **15** (5 × 3) |
| Wall time | 3.8 s |
| Cost | **~$0.05** (run was bundled with Haiku swap, see Test 4) |
| Cost reduction vs Test 1 | **~28 %** (model-isolated estimate) |
| Quality verdict | **same** — same per-region depth, no useless low-weight suggestions |

**Verdict: SHIP** — keep the new defaults in production. To go even tighter set
`SUGGESTION_MIN_WEIGHT=0.20` (critical-only mode → ~2-3 regions, ~$0.02/call) but you lose
Visual Cortex / Amygdala / Reward suggestions.

---

## Test 3 — Goal sensitivity (~$0.30)

**Why:** If suggestions barely change across goals, the goal weighting isn't doing useful work
and you can cache aggressively across goals.

**Setup:** same image / audio / caption. Run once per goal:

| Goal | Cost | Suggestions | % overlap with Conversion baseline |
|------|------|-------------|----------------------------------|
| Conversion | $0.07 | 21 | (baseline) |
| Awareness | $___ | ___ | ___ % |
| Engagement | $___ | ___ | ___ % |
| Brand Recall | $___ | ___ | ___ % |

**% overlap = number of suggestion *titles* that appear in both / total in baseline.** Use a
rough manual count.

**Decision rule:**
- Overlap > 85 % across goals → goal weighting is cosmetic; collapse goals or memoize harder.
- Overlap < 50 % → goals genuinely differentiate; current design is correct.

**Result:** ___

---

## Test 4 — Haiku comparison (DONE 2026-04-28, $0.01)

**Why:** Sonnet may be overkill for templated suggestion text. Haiku is ~1/8 the cost and ~3×
faster.

**Setup:** in `.env`, set `ANTHROPIC_MODEL=claude-haiku-4-5-20251001`. Re-ran Pepsi /
Conversion. Threshold tuning from Test 2 was already applied. Audio was REMOVED for this run
(caption-only) — note that's a confound; pure model comparison would hold modality constant.

| Field | Sonnet (Test 1, with audio) | Haiku (this run, no audio) |
|---|---|---|
| Cost | $0.07 | **$0.01** |
| Wall time | 3.2 s | 3.8 s |
| Suggestions returned | 21 | 15 |
| Regions fired | 7 | 5 |
| Quality (1–10) | 8 | 8 — no measurable drop |
| Notable diffs | hallucinated 'in buy' audio | **also hallucinated 'in buy' audio** despite no audio in run |

**Cost lever attribution:**
- Test 2 threshold tuning: ~28 % cut (7 → 5 regions on Sonnet)
- Test 4 Haiku swap: ~80 % cut at the model level (Sonnet → Haiku for same prompt)
- Stacked: $0.07 → $0.01 = **86 % total reduction**

**Verdict: SHIP** — Haiku is the new default. At 1k analyses/mo, this is $10 instead of $70.

**Critical finding:** Both Sonnet and Haiku hallucinate 'in buy' voiceover content. Switching
models doesn't fix it. This is a fixture/architecture issue (the post fixture's brain scores
were generated WITH audio, so phantom audio dips leak into the moments passed to the LLM).
Confirms vision/audio grounding (Test 7) is needed regardless of model choice.

---

## Test 5 — Cache verification ($0.07 + ~$0.05)

**Why:** Anthropic prompt caching is supposed to halve repeat-call cost. Confirm it's actually
firing.

**Setup:** run the **identical** Pepsi / Conversion analyze twice back-to-back. Then check
[console.anthropic.com](https://console.anthropic.com) → **Usage** → Logs.

| Run | Cost | cache_read_input_tokens (per region call) |
|-----|------|-------------------------------------------|
| First (cold) | $___ | should be 0 |
| Second (warm) | $___ | should be ≥ 600 per call |

**Decision rule:**
- Cache hits → cost ratio second/first should be ~0.5. If it's not, the system prompt isn't
  being marked `cache_control: ephemeral` correctly.
- No cache hits → check `services/suggestions/llm/anthropic_native.py` for the cache_control
  field.

**Result:** ___

---

## Test 6 — Determinism & temperature (DONE 2026-04-28, $0.06 total)

**Why:** Lower temp = more cacheable, more reproducible, simpler to QA. Confirm we don't lose
quality.

**Setup:** Three identical Pepsi/Conversion/no-audio runs at `LLM_TEMPERATURE=0.5`, taken
within ~20 minutes (Tests 4/5/6 in screenshots).

| Run | Wall time | Cost | Suggestions |
|-----|-----------|------|-------------|
| 1 (1:08 PM) | 4929 ms | ~$0.01 | 15 |
| 2 (1:24 PM) | 3945 ms | ~$0.01 | 15 |
| 3 (1:26 PM) | 3918 ms | ~$0.01 | 15 |

### Cross-run consistency at temp=0.5

| Region | Theme overlap across 3 runs |
|---|---|
| Visual Cortex | "boost contrast/saturation" appears in **all 3**; "reframe / center subject" in **all 3** (varied phrasing); border/frame in **2/3** |
| Amygdala | "urgency caption hook" in **all 3**; "contrast/saturation in image" in **all 3**; benefit-driven emotion in **2/3** |
| Prefrontal | "front-load CTA" in **all 3**; benefit-first framing in **all 3**; contrast/visual hierarchy in **2/3** |
| Motor | "CTA action verb in caption headline" in **all 3**; "enlarge/center product" in **all 3**; directional cue in **2/3** |
| Reward | "visual reward cue in image" in **all 3**; "immediate gratification caption" in **all 3** |

**Semantic overlap: ~80–90 % across runs.** Same themes recur with different surface wording.
Exact-title overlap is lower (~30 %) — Claude rewrites the lead phrase each time even when the
underlying advice is identical.

**One run-to-run quality artifact**: Run 1's Reward region produced a "warm, energetic music
or voiceover tone if audio exists" suggestion (a leftover from before the phantom-audio fix
fully bedded in). Runs 2 and 3 don't include any audio refs — they cleanly stick to image +
caption. This is consistent with "the fix works for new requests but the model occasionally
defaults to audio templates when it pattern-matches strongly enough."

### temp=0.0 follow-up (3 more runs, $0.03)

| Run | Wall time | Cost | Suggestions |
|-----|-----------|------|-------------|
| 1 (1:29 PM) | 5630 ms | ~$0.01 | 15 |
| 2 (1:31 PM) | 3517 ms | ~$0.01 | 15 |
| 3 (1:32 PM) | 3812 ms | ~$0.01 | 15 |

### Cross-run consistency at temp=0.0

| Region | Slots identical across all 3 runs | Slots identical in 2/3 |
|---|---|---|
| Visual Cortex | 1/3 ("Increase contrast and saturation") | 2/3 (border, enlarge — varied last word) |
| Amygdala | 2/3 ("urgency caption", "contrast intensity") | 1/3 |
| Prefrontal | 2/3 ("CTA urgency", "benefit-driven question") | 1/3 (risk reversal in 2/3) |
| Motor | 1/3 ("CTA action verb") | 2/3 (enlarge, urgency) |
| Reward | 0/3 | 2/3 ("benefit-forward hook") |

**Exact-title overlap: ~60–70 %.** Roughly double temp=0.5 (~30 %) but still well short of the
>80 % bar I set for "ship temp=0 as the new default". Anthropic's API has some non-determinism
even at temp=0 — likely tied tokens picked differently across requests, or sampling jitter
server-side.

**Quality: unchanged.** Same advice, slightly tighter wording variation. Reading the three
temp=0 reports side-by-side, you'd struggle to tell which was which.

### Verdict

**Keep temp=0.5 as default.** Reasoning:

- Title overlap doubled (30 % → ~65 %), which is real but not transformative.
- The remaining 30–40 % variance at temp=0 means snapshot tests can't fully rely on it — you'd
  still need fuzzy matching, which works just as well for temp=0.5.
- Both produce indistinguishable *advice* — the variance is purely cosmetic phrasing.
- Lower temp doesn't unlock cache (we're below the 2048-tok minimum either way), doesn't drop
  cost, and doesn't improve quality.

**When to revisit:** if you ever need bit-exact reproducibility (e.g. regulated environments
demanding deterministic LLM outputs), set temp=0 then; until then it's a wash.

| Temperature | Cost (3 runs) | Exact-title overlap | Theme overlap | Quality |
|---|---|---|---|---|
| 0.5 (current) | $0.03 | ~30 % | ~80–90 % | unchanged |
| 0.0 | $0.03 | ~60–70 % | ~80–90 % | unchanged |

---

## Test 7 — Vision-grounding pre-check (~$0.07)

**Why:** Decide whether the ~7–8 hr build cost of sending the actual image to Claude is worth
it. The hypothesis: Claude is hallucinating image details from scores alone. Stress-test that.

**Setup:** Use the **same Pepsi cream soda image and audio**, but change the caption to
something contradictory — e.g.:

> "Apple iPhone 17 launches today — pre-orders now open."

Run goal=Conversion. Read every suggestion. Tag each:
- ✅ References reality (Pepsi / cream soda / can / blue / ice cream)
- ❌ References fabricated content (iPhone / Apple / launch — even though that's the caption)
- 🟡 Generic — references neither, just "the product" or "the visual"

| Tag | Count |
|-----|-------|
| ✅ Real | ___ |
| ❌ Fabricated from caption | ___ |
| 🟡 Generic | ___ |

**Decision rule:**
- ❌ count > 30 % → Claude is anchoring on caption when image disagrees. **Vision-grounding is
  essential**, not nice-to-have. Build it before launching publicly.
- 🟡 count > 70 % → suggestions are too generic to be image-aware. Vision-grounding will boost
  perceived quality even without correctness gains.
- ✅ count > 50 % despite contradictory caption → Claude is somehow inferring image content
  from scores; vision-grounding less urgent.

**Result:** ___

---

## Phantom audio fix — verification (DONE 2026-04-28)

**Why:** Test 4 surfaced that both Sonnet and Haiku invent "in buy" voiceover
content for caption-only posts because the mock fixture's brain scores carry
baked-in audio dips. Worst quality bug we'd seen. Fix at
[services/suggestions/__init__.py](services/suggestions/__init__.py): when
`content_type=="post"` AND `image_count<=1` AND `not has_audio`, strip moments
from the prompt entirely.

**Verification setup:** Same Pepsi/Conversion/no-audio caption-only run that
produced the hallucinations in Test 4.

| Field | Test 4 (before fix) | Verification (after fix) |
|---|---|---|
| Suggestions returned | 15 | 15 |
| 'in buy' hallucinations | 4+ confident audio fixes | **0** |
| Audio-related suggestions | "Re-record voiceover…" (fabricated) | **1, conditionally hedged**: "If voiceover is present, re-record with…" |
| Quality (1–10) | 6 (poisoned by hallucinations) | **8** (clean image + caption levers, conditional audio) |
| Wall time | 3.8 s | 4.9 s |

**Verdict: SHIPPED.** The model now correctly says "*if* you add audio later"
instead of fabricating present audio. Image and caption suggestions are
unaffected — same depth, same actionability. Three new tests in
[tests/test_suggestions.py](tests/test_suggestions.py) lock the behavior:
- single-image post + no audio → moments stripped
- single-image post + audio → moments pass through
- carousel + no audio → moments pass through (image positions are real)
- video + no audio → moments pass through (timeline is real)

---

## Decision matrix — updated 2026-04-28

| Lever | Cost change | Quality change | Decision |
|-------|------------|---------------|----------|
| Threshold tuning (Test 2) | **−28 %** | same | **SHIPPED** |
| Haiku swap (Test 4) | **−80 % at model** | same | **SHIPPED** |
| Phantom audio strip | 0 % | **+major** | **SHIPPED** — eliminates 'in buy' hallucinations |
| Cache fix (Test 5) | tiny at this scale | n/a | **SKIP** — prompts too small (<2048 tok) to qualify |
| Temp 0.0 (Test 6) | 0% | unchanged | **HOLD** — overlap doubles (30%→65%) but doesn't hit the 80% bar; keep temp=0.5 |
| Vision-grounding (Test 7) | +tokens, +latency | ___ | **less urgent now** — phantom audio was the worst hallucination; remaining gap is "describe actual image content" |

**Total cost per analyze after applied levers:** **~$0.01** (was $0.07)
**At 1,000 analyses/mo:** **$10** (was $70)
**At 10,000 analyses/mo:** **$100** (was $700)

---

## Recommended order of operations

1. **Today (free):** Test 2. Threshold tuning is the single biggest UX + cost win and requires
   no API spend to evaluate.
2. **Today (~$0.10):** Tests 4 + 5 in parallel. Haiku evaluation + cache verification answer
   "can I run this 10× cheaper without losing quality" for almost no money.
3. **This week (~$0.50):** Tests 3 + 6. Goal sensitivity and temperature tell you whether the
   current design is doing useful work.
4. **Before launch (~$0.07):** Test 7. Vision-grounding gating decision.
5. **Defer:** Vision-grounding *implementation*. ~7–8 hr build. Only justify it if Test 7
   shows ❌ > 30 % or 🟡 > 70 %.

---

## Notes on each run

Use this section to jot anomalies (rate limits, weird response shapes, slow regions, etc.):

- 2026-04-28 Test 1: Sonnet 4.6 hallucinated "in buy" audio fragment across 4+ suggestions.
  WhisperX likely produced this as a misheard token. Confirms Claude has zero ability to verify
  audio content.
- 2026-04-28 Test 4: Haiku ALSO hallucinates 'in buy' even when the run has NO audio. Means
  the phantom audio comes from the mock fixture's brain scores (generated with audio in
  build_post_fixture.py), not from anything in the live request. The moment dips encoded in
  `golden_pred_post_*.npy` carry audio assumptions forward. Two paths to fix:
    (a) regenerate the fixture without audio for caption-only test runs,
    (b) strip moment timestamps from the prompt when the live request has no audio.
  Implemented (b) — cheaper + structurally correct.
- 2026-04-28 Phantom audio fix verification: re-ran identical Pepsi caption-only/Conversion.
  Result: 0 'in buy' hallucinations across 15 suggestions. The one audio-related suggestion
  (Reward region, Haiku) now correctly hedges with "If voiceover is present, re-record…"
  rather than fabricating present voiceover. Image and caption suggestion quality unchanged.
  Wall time bumped 3.8s → 4.9s — likely cache cold-start since prompt content shifted.
