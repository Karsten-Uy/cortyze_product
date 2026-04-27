# Reference Ad Library

Pre-scored content the [Stage 2 suggestion engine](../../IMPLEMENTATION_PLAN.md) uses to back every suggestion with a concrete example. When a user's `visual_cortex` scored 32, the engine queries this library for ads that scored 88+ in `visual_cortex` and surfaces them alongside the suggestion.

Each `<name>.json` is a manifest written by [scripts/register_reference_ad.py](../../scripts/register_reference_ad.py). The Python query API lives in [services/examples/library.py](../../services/examples/library.py).

## Manifest shape

```json
{
  "name": "sintel_trailer",
  "display_name": "Sintel Trailer",
  "source_url": "https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4",
  "description": "Blender Foundation open-movie trailer, fantasy animation",
  "license": "CC BY 3.0",
  "predictions_path": "tests/fixtures/golden_pred_sintel_T53.npy",
  "predictions_shape": [53, 20484],
  "region_scores": { "visual_cortex": 64.2, "motor": 61.5, ... },
  "overall_by_goal": { "engagement": 52.7, ... },
  "registered_at": "2026-04-25T22:00:00Z"
}
```

## How to add a new reference ad

1. **Generate the fixture** — pick a video URL, run `build_fixture.py`. Each run is ~2 hours on M3 Mac CPU (most of it is one-time HF weight loading + WhisperX transcription, which gets faster after the first run as caches warm up):
   ```bash
   /Users/kirby/Documents/cortyze/tribev2/.venv/bin/python scripts/build_fixture.py \
     --video-url https://download.blender.org/cosmoslaundromat/trailer/cosmoslaundromat_trailer_h264_1080p.mp4 \
     --output-stem tests/fixtures/golden_pred_cosmoslaundromat
   ```

2. **Register it** — computes scores from the fixture, writes the manifest:
   ```bash
   uv run python scripts/register_reference_ad.py \
     tests/fixtures/golden_pred_cosmoslaundromat_T53.npy \
     --name cosmoslaundromat \
     --display-name "Cosmos Laundromat" \
     --source-url https://download.blender.org/cosmoslaundromat/trailer/cosmoslaundromat_trailer_h264_1080p.mp4 \
     --description "Blender open-movie short, surreal animation" \
     --license "CC BY 3.0"
   ```

3. **Verify** — `services.examples.library.all_ads()` will now include it. The query API picks it up automatically (no code changes).

## Curated URL list — varied content for spectrum coverage

For Stage 2 the library should cover a range of brain-activation profiles so suggestions can match diverse weak spots. All Blender open movies are CC BY 3.0 and well-encoded MP4s.

| Name slug | URL | Why pick this |
|---|---|---|
| `sintel_trailer` ✅ | <https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4> | Already done. Music, fantasy CGI, brief dialogue. Top regions: visual_cortex (64), motor (62) |
| `bigbuckbunny_trailer` | <https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_480p_h264.mov> | No dialogue — should score lower in `temporal_language`, isolating visual/auditory contribution |
| `tears_of_steel` | <https://download.blender.org/mango/download.blender.org/demo/movies/ToS/ToS-4k-1920.mov> | Live action + heavy speech — should score high in `temporal_language` |
| `cosmoslaundromat` | <https://download.blender.org/cosmoslaundromat/trailer/cosmoslaundromat_trailer_h264_1080p.mp4> | Surreal, unusual visuals — different `fusiform_face` profile |
| `agent327_barbershop` | <https://download.blender.org/agent327/01-Barbershop/barbershop_h264.mp4> | Comedy, character close-ups — high `fusiform_face` (faces) |
| `caminandes_llamigos` | <https://download.blender.org/durian/Caminandes_Llamigos.mp4> | Slapstick action, no dialogue — high `motor` |
| `spring` | <https://download.blender.org/demo/movies/Spring/Spring_2019_Open_Movie_FullHD.mp4> | Atmospheric, slow-paced — different `reward` and `amygdala` profile |

You don't need all 7 — even **3-5 varied clips** gives the suggestion engine enough to match weak regions to strong examples. Aim for one clip that scores high in each of the regions you most care about for marketing (visual_cortex, amygdala, hippocampus = brand recall).

## What's still mock-quality about this

- Per-region scores come from the **single-clip calibration** in `core/scoring/calibration.json`. Once you have RunPod and run real cross-clip calibration on ~30 clips, scores will recalibrate but the manifest schema stays the same — re-register with the new scores.
- The `predictions_path` field points at a local fixture file. Stage 2 production replaces this with R2 URIs (where saved `(T, 20484)` arrays from real `/analyze` calls live).
- `is_reference=true` flagging in Postgres is the eventual long-term home — for now, a JSON file is the library.

## Bulk re-register after recalibration

When `calibration.json` changes (Stage 2 cross-clip calibration), all manifests need their scores recomputed. Re-run the register script for each fixture; the script overwrites `<name>.json` in place. A batch script for this lives in your future as `scripts/recalibrate_library.py` once it's needed.
