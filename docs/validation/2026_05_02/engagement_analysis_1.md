# TRIBE v2 ↔ Social Media Metrics: Correlation Analysis
## `run_20260502_012937` — 77 reels, 5 brands, goal=engagement

---

## TL;DR

There is a **real, positive signal**: TRIBE v2 scores correlate with view and like counts at r ≈ +0.36 pooled across all brands, rising to **r = +0.70 within Harvard** and **r = +0.48 within Marvel** — meaningful for an n of 17–20 per brand. However, the correlation is **entirely absent for engagement rate** (likes/views %), and the pooled number is **partially inflated by a brand-level confound**. The signal is real but noisy enough that you need more data and better-normalized metrics before drawing product conclusions.

---

## 1. Pooled Correlations (n = 77)

All brain regions vs. social metrics, Pearson r:

| Brain region | r vs log(views) | r vs log(likes) | r vs ER% |
|---|---|---|---|
| **reward** | **+0.512** | **+0.406** | -0.323 |
| **overall_score** | **+0.363** | **+0.365** | -0.119 |
| prefrontal | +0.361 | +0.298 | -0.232 |
| motor | +0.267 | +0.245 | -0.103 |
| visual_cortex | +0.236 | +0.276 | -0.082 |
| temporal_lang | +0.224 | +0.298 | -0.033 |
| amygdala | +0.200 | +0.111 | -0.084 |
| fusiform_face | +0.106 | +0.224 | +0.108 |
| hippocampus | +0.062 | +0.148 | +0.042 |

**What this says:**
- Volume metrics (views, likes) correlate positively with TRIBE scores at a weak-to-moderate level. This is directionally encouraging.
- **Reward** is the single strongest predictor (r = +0.51 vs log views) — content that registers high reward signal accrues views.
- **ER% shows no relationship** — TRIBE v2 does not predict whether people click like relative to how many saw the video. That's a separate behaviour (algorithmic sorting vs. volitional response).
- Log-transforming views/likes matters: raw-count correlations are dominated by NBA outliers (8.9M views for a Mavs clip). The log scale gives the true relationship across the range.

---

## 2. The Brand Confound

The pooled r is partly artifactual. Brands differ systematically in both audience size and content type:

| Brand | n | Avg TRIBE score | Avg log(views) | Median views | Avg ER% |
|---|---|---|---|---|---|
| Pepsi | 16 | 54.8 | 3.21 | 1,318 | **5.89%** |
| Harvard | 17 | 52.8 | 4.03 | 9,565 | 2.71% |
| NBA | 20 | 60.3 | 4.82 | 37,684 | 0.77% |
| Marvel | 20 | 57.2 | 5.16 | 136,432 | 0.27% |

Pepsi scores medium, gets tiny view counts, but crushes on engagement rate (small, loyal audience). NBA scores highest and gets large view counts. This creates a spurious positive r in the pooled data — the brands that have bigger audiences also happen to make higher-scoring content.

**You cannot interpret the pooled r as "TRIBE predicts virality" without first controlling for brand.** The within-brand numbers are what matter.

---

## 3. Within-Brand Correlations (the real test)

| Brand | n | r(score, log views) | r(score, log likes) | r(score, ER%) |
|---|---|---|---|---|
| **Harvard** | 17 | **+0.696** | **+0.583** | -0.091 |
| **Marvel** | 20 | **+0.483** | +0.365 | -0.311 |
| NBA | 20 | +0.256 | +0.301 | +0.148 |
| Pepsi | 16 | +0.176 | +0.130 | +0.037 |

**Harvard (r = +0.70)** is the strongest result. Within a fixed audience, higher TRIBE scores correspond meaningfully to more views and likes. The Harvard content range is diverse (animal videos, class announcements, archaeology digs, food content) — that variation probably gives TRIBE room to differentiate.

**Marvel (r = +0.48)** shows a moderate-to-strong within-brand signal. The catalogue has wide score spread (43.5–73.6) and the view counts vary enormously (35K to 711K), so the model is tracking something real about content quality within that franchise.

**NBA (r = +0.26)** is weak. NBA content is uniformly high-energy and high-scoring (scores 41–72), and views are driven by which game or player is featured rather than production quality. TRIBE v2 likely lacks a "star power" signal.

**Pepsi (r = +0.18)** is essentially noise. Pepsi's reel views are almost entirely driven by paid boosting and campaign windows, not organic content quality — which is exactly what TRIBE v2 cannot see.

---

## 4. What the Engagement Rate Result Means

ER% (likes ÷ views) is **negatively correlated** with TRIBE score in most buckets (especially within Marvel at r = -0.31). Higher-scoring content gets more views but proportionally *fewer* likes per view — the algorithm surfaces it broadly, but not to the core audience. This is not a TRIBE v2 failure; it reflects a real platform behaviour: widely-distributed content gets passive viewers, engaged content gets core fans who all like. These are different things.

For the product, this matters: if your customer goal is **engagement rate**, TRIBE v2 in its current form is not the right signal. If the goal is **reach / total volume**, the correlation is useful.

---

## 5. What You Don't Have Yet (Required for Validation)

The current dataset has several confounds that make it impossible to draw strong conclusions:

**Age normalization.** Pepsi reels are 26–285 days old; Harvard reels are 8–1383 days old. Older content simply had more time to accumulate views. You need `views_per_day` or age-corrected metrics before comparing across posts or brands. The `derived_age_days` column is in the CSV — divide views by it.

**Follower count normalization.** A reel from `pepsi` (unknown follower count in the data) and from `Harvard University` are not on the same reach footing. Engagement rate is the right denominator for cross-brand comparison; views is not. `plat_channel_follower_count` is in the CSV but mostly null — fill it in.

**No holdout split.** You're correlating scores on the same reels used to explore the model. You need a genuinely held-out set of content scored *before* the posts go live, then checked against 7-day performance — that's the real product validation test.

**n is too small per brand.** With n = 16–20, r = +0.20 is not statistically significant (p ≈ 0.45). Harvard at r = +0.70 with n = 17 *is* significant (p < 0.01), but it's one brand. You need at least 50 posts per brand to draw conclusions.

**Missing comments/reposts.** `plat_comment_count` and `plat_repost_count` are null in this run. Comments are often a better proxy for emotional engagement than likes. Wire those in.

---

## 6. Recommended Next Steps

**Immediate (can do with this dataset):**

1. **Add `views_per_day`** — divide `plat_view_count` by `derived_age_days` and re-run correlations. This alone will clean up the age confound and probably strengthen the within-brand r for Harvard and Marvel.
2. **Re-run ER% with `(likes + comments) / views`** once comment data is populated — comments correlate better with memorability, which TRIBE v2 is designed to predict.
3. **Plot scatter by brand** — a simple `score vs log(views_per_day)` scatter with brand colour-coding will make the within-brand pattern visually obvious. Run it now; it's the most persuasive artifact for stakeholders.

**Next data collection run:**

4. **Expand Pepsi n to 100+** with both organic and boosted posts tagged — then split them. TRIBE v2 should predict organic performance but not paid. If it does predict paid, something is off.
5. **Add a second influencer brand** alongside Serena Page (n=2 is useless). Influencer content has a tighter audience and less algorithmic noise than brand accounts.
6. **Score content before posting** — run TRIBE v2 on 20 pieces of unpublished content for one brand, post them all, collect 7-day metrics. This is the only experimental design that tests predictive validity rather than retrospective correlation.
7. **Vary goal** — re-run this batch with `goal=brand_recall` and `goal=conversion` and compare which brain regions become better predictors of which platform metrics. If `overall_brand_recall` (not `overall_engagement`) better predicts ER%, that's a product insight.

---

## 7. Summary Verdict

| Question | Answer |
|---|---|
| Does TRIBE v2 output correlate with social metrics? | **Yes, weakly to moderately** — r = +0.36 pooled, r = +0.48–0.70 within-brand |
| Is the pooled r trustworthy? | **Partially** — brand-level confound inflates it; within-brand numbers are the real test |
| Which brain region is most predictive? | **Reward** (r = +0.51 vs log views, pooled) |
| Does it predict engagement rate? | **No** — consistently near-zero or negative |
| Is the sample sufficient to ship a product claim? | **No** — need age-normalization, follower-normalization, larger per-brand n, and holdout validation |
| What's the fastest way to strengthen the signal? | **Age-normalize views, add comments, collect 50+ posts per brand** |

---

*Analysis based on 77 completed inferences from `run_20260502_012937` · brands: Pepsi, Harvard, NBA, Marvel, Disney, Serena Page, Spurs*