# Reference Ad Library

Pre-scored content the Stage 2 [suggestion engine](../../services/suggestions/) uses to back every suggestion with a concrete example. When a user's `visual_cortex` scored 32, the engine queries this library for ads that scored high in `visual_cortex`, weighted by goal fit and topical relevance, and surfaces the best one in the SuggestionCard expand UI.

Each `<name>.json` is a manifest written by [scripts/register_reference_ad.py](../../scripts/register_reference_ad.py). The Python query API lives in [services/examples/library.py](../../services/examples/library.py); the public `GET /examples` and `GET /examples/{name}` endpoints serve them to the frontend.

---

## How the suggestion engine picks an example

For each suggestion (region X, goal Y, user run with caption + brand-context), `library.best_examples(...)` ranks every registered ad by:

```
score(ad) = (0.7 × ad.region_scores[X] + 0.3 × ad.overall_by_goal[Y])
            × lexical_relevance(user_text, ad.description + ad.tags + ad.caption)
            + content_type_bonus(user_content_type, ad.content_type)
```

- **Region score** (70 %): how good this ad is at the region we're trying to fix
- **Goal-weighted overall** (30 %): tie-breaker, prefer ads that are a good fit for the user's goal
- **Lexical relevance** (multiplier 0.5x–1.5x): word-overlap between the user's caption + brand context and the ad's description / tags / caption. Stand-in for real CLIP / text-embedding similarity until that lands.
- **Content-type bonus** (+5 if matched): post-mode users see post examples first; video users see video examples first.

The top-2 names ship inline on each Suggestion as `Suggestion.examples`. The frontend lazy-loads the full `ExampleAd` (with thumbnail) when the user expands a card.

---

## Manifest shape

```json
{
  "name": "pepsi_cream_soda",
  "display_name": "Pepsi Cream Soda Reveal",
  "source_url": "https://instagram.com/p/...",
  "thumbnail_url": "https://cdn.cortyze.com/refs/pepsi-cream-soda.jpg",
  "description": "Static product post — soda can with ice-cream cone, blue background",
  "caption": "Treat yourself! Pepsi Cola Cream Soda is here to stay.",
  "tags": ["beverage", "cpg", "product-shot", "still-life", "blue"],
  "content_type": "post",
  "license": "Fair use — Pepsi Co",
  "predictions_path": "tests/fixtures/golden_pred_post_<hash>_T6.npy",
  "predictions_shape": [6, 20484],
  "region_scores": { "visual_cortex": 31.2, ... },
  "overall_by_goal": { "engagement": 34.4, ... },
  "registered_at": "2026-04-30T...Z"
}
```

| Field | Required? | Notes |
|---|---|---|
| `name` | ✅ | Slug; used as the manifest filename and lookup key |
| `display_name` | ✅ | Human-readable title shown in the card |
| `source_url` | ✅ | Where the original lives — opened by the "View source" link |
| `thumbnail_url` | recommended | 16:9 PNG/JPG. Without it the card shows a "No preview" placeholder. |
| `description` | recommended | One-line description; fed into the lexical-relevance ranker |
| `caption` | optional | Original caption / copy. Big relevance boost when set. |
| `tags` | recommended | 3-7 lowercase keywords. Used for relevance ranking. |
| `content_type` | recommended | `"video"` / `"post"` / `"image"`. +5 ranking bonus when matched. |
| `license` | recommended | License/usage terms shown in the card |
| `predictions_path` | ✅ | Path to the `.npy` (auto-set by `register_reference_ad.py`) |
| `region_scores` | ✅ | Per-region calibrated 0-100 (auto-computed) |
| `overall_by_goal` | ✅ | Per-goal weighted overall (auto-computed) |

---

## How to add a new reference ad

### Path A — registering a video reference

```bash
# 1. Generate the (T, 20484) fixture from the video URL.
#    Wall time: 30-90 min on M3 Mac (CPU + bf16 mode).
/Users/kirby/Documents/cortyze/tribev2/.venv/bin/python \
  scripts/build_fixture.py \
  --video-url https://example.com/cool-ad.mp4

# 2. Register it. Look at the printed shape from build_fixture
#    (something like `golden_pred_video_<hash>_T53.npy`) and pass that.
uv run python scripts/register_reference_ad.py \
  tests/fixtures/golden_pred_video_<hash>_T53.npy \
  --name cool_ad \
  --display-name "Cool Brand Hero Spot" \
  --content-type video \
  --source-url https://example.com/cool-ad.mp4 \
  --thumbnail-url https://cdn.example.com/cool-ad-thumb.jpg \
  --description "30-second hero spot, fast cuts, character close-ups" \
  --tags lifestyle,fashion,fast-cuts,faces \
  --license "© Cool Brand 2026 — used with permission"

# 3. Restart uvicorn (the library is lru_cached at import time).
```

### Path B — registering a post reference (image + caption)

```bash
# 1. Generate the post fixture from the image (+ optional audio + caption).
/Users/kirby/Documents/cortyze/tribev2/.venv/bin/python \
  scripts/build_post_fixture.py \
  --image-url https://example.com/post-image.jpg \
  --caption "Your caption here"

# 2. Register the resulting fixture. Note --content-type post.
uv run python scripts/register_reference_ad.py \
  tests/fixtures/golden_pred_post_<hash>_T6.npy \
  --name brand_x_summer_drop \
  --display-name "Brand X Summer Drop Reveal" \
  --content-type post \
  --source-url https://instagram.com/p/abc123 \
  --thumbnail-url https://example.com/post-image.jpg \
  --caption "Your caption here" \
  --description "Static product post — minimalist, pastel palette" \
  --tags fashion,minimalist,pastel,product-shot \
  --license "Fair use"

# 3. Restart uvicorn.
```

The post fixture build is faster than video (~30-50 min on M3 vs 30-90 min for video) and is the right call for most marketing references — most ads creators see in their feed are static posts or carousels.

---

## Choosing thumbnails

The thumbnail shown in the SuggestionCard is **96 × 96 pixels rendered at object-cover**. Aim for an image where the subject is well-centered and recognizable at that size:

- **Video ads**: pick a representative still — usually the hero frame around the 1/3 mark, or the brand-logo end card. Don't use the very first frame; openers are often blank.
- **Post ads**: use the post image itself (the file you uploaded to `--image-url`).
- **Carousels**: use the first image; or composite the first 2-3 images into one square if you want to hint at the carousel.
- **Format**: JPG or PNG, 400 × 400 minimum. Avoid transparency.
- **Hosting**: any public URL works (Imgur, your own CDN, R2 with public access). The frontend just loads it via `<img src>`.

If you skip `--thumbnail-url`, the card falls back to a "No preview" placeholder. The system still functions, just less rich.

---

## Choosing tags

Tags are pure-text relevance keys. The lexical-relevance ranker tokenizes both the user's caption + brand context and the ad's description + tags + caption, and bumps ads with overlapping vocabulary.

Pick **3-7 lowercase tags**, separated by commas in the CLI flag:

```bash
--tags beverage,cpg,product-shot,still-life,blue,minimal
```

Categories that work well:

| Axis | Examples |
|---|---|
| Vertical | `beverage`, `apparel`, `tech`, `cpg`, `automotive`, `fashion`, `beauty`, `fintech` |
| Format / shape | `product-shot`, `lifestyle`, `testimonial`, `unboxing`, `still-life`, `fast-cuts`, `vlog` |
| Visual character | `minimalist`, `maximalist`, `pastel`, `high-contrast`, `monochrome`, `vibrant`, `cinematic` |
| Subject | `face`, `hands`, `text-overlay`, `product`, `crowd`, `landscape` |
| Tone | `playful`, `aspirational`, `urgent`, `informative`, `emotional` |

**Don't** include the product/brand name — those are usually one-off and won't help relevance ranking; put them in `description` or `caption` instead.

---

## Curated URL list — building out coverage

The library should span a range of brain-activation profiles so the ranker has diverse examples to choose from. The user's content varies wildly; if every reference is a Sintel trailer, every suggestion shows a Sintel trailer.

**Aim for breadth across** content type × goal × top region:

| Coverage axis | What you want |
|---|---|
| Content type | At least 2 video, 2 single-image post, 1 carousel |
| Goal × top region | One ad that's strong in each (region, goal) combo you care about — e.g., a high-`amygdala` Conversion ad, a high-`hippocampus` Brand-Recall ad |
| Vertical | Don't have all 5 ads be fashion. Mix verticals so tags help. |

### Free-license videos (CC BY 3.0, no licensing risk)

| Slug | URL | Why pick this |
|---|---|---|
| `sintel_trailer` ✅ | <https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4> | Music, fantasy CGI, brief dialogue. Top regions: visual_cortex, motor |
| `bigbuckbunny_trailer` | <https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_480p_h264.mov> | No dialogue — low temporal_language, isolates visual/audio contribution |
| `tears_of_steel` | <https://download.blender.org/mango/download.blender.org/demo/movies/ToS/ToS-4k-1920.mov> | Live action + heavy speech — high temporal_language |
| `cosmoslaundromat` | <https://download.blender.org/cosmoslaundromat/trailer/cosmoslaundromat_trailer_h264_1080p.mp4> | Surreal visuals — different fusiform_face profile |
| `agent327_barbershop` | <https://download.blender.org/agent327/01-Barbershop/barbershop_h264.mp4> | Comedy, character close-ups — high fusiform_face |
| `caminandes_llamigos` | <https://download.blender.org/durian/Caminandes_Llamigos.mp4> | Slapstick action, no dialogue — high motor |
| `spring` | <https://download.blender.org/demo/movies/Spring/Spring_2019_Open_Movie_FullHD.mp4> | Atmospheric, slow-paced — distinctive reward / amygdala profile |

### Real ad references (require permission for redistribution)

For real marketing references, **don't redistribute the source files**. Use a permanent CDN URL the brand has already published (Instagram CDN URLs expire, host on your own R2 or use the brand's own marketing site). Tag the manifest with `"license": "Fair use — <brand>"` to flag that the asset isn't yours.

A useful mix:

- **CPG / beverage**: 1 still-life product shot, 1 video spot
- **Fashion / apparel**: 1 lifestyle still, 1 motion ad
- **Tech / SaaS**: 1 demo video, 1 testimonial post
- **Auto**: 1 hero-spot video (cinematic, music-heavy)

You don't need all of them — **5-10 ads is the right size** for a well-functioning library. Past ~30, ranking saturates and storage starts mattering.

---

## What's still mock-quality

1. **Calibration is single-clip.** Per-region scores come from the calibration in `core/scoring/calibration.json`, which was fit against the sintel + Pepsi fixtures. Once we have ~30 reference predictions and run cross-clip calibration on RunPod, scores will shift. **Re-register all ads after recalibration** using the same `register_reference_ad.py` flags + `--force`.
2. **Lexical relevance is keyword-overlap, not semantic.** A user posting about "running shoes" gets a relevance boost from any ad with `"running"` in its description, but won't match `"footwear"` even though they're semantically related. Stage 2-3 swaps `_lexical_relevance` in [library.py](../../services/examples/library.py) for real CLIP (visual) + `text-embedding-3-small` (text) similarity. The ranking function signature stays the same.
3. **Library lives on disk.** Each manifest is a JSON file in this directory. Stage 2 production moves them to Postgres (with `is_reference=true` flag on the existing `reports` table). The Python query API stays unchanged; only `_load_all` swaps.
4. **No per-creator favorites.** Once Stage 4 audience profiling lands, references should weight by similarity to the creator's *own* high-performing content, not just generic top-N. That's audience-data-driven and out of scope here.

---

## Operational notes

**Re-register after recalibration:**

```bash
# Hypothetical batch script (TODO: write this when needed)
for fixture in tests/fixtures/golden_pred_*.npy; do
  name=$(...)  # extract from manifest matching the fixture
  uv run python scripts/register_reference_ad.py "$fixture" --name "$name" --force ...
done
```

**Library cache** is `@lru_cache(maxsize=1)` so any uvicorn worker only loads the JSON files once at import. After registering a new ad, **restart uvicorn** or call `services.examples.library.reload()` in tests / a one-off REPL.

**Adding the same name twice** is rejected by `register_reference_ad.py` unless you pass `--force`. This catches typos that would silently overwrite.
