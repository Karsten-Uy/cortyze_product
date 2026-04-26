# Test fixtures

Real `(T, 20484)` brain-prediction arrays from Meta's TRIBE v2, used as golden inputs for atlas-mapper and scoring tests. Each `.npy` is paired with a `.meta.json` sidecar recording the video URL, SHA-256, model revision, and generation timestamp.

## Files

- `golden_pred_sintel_T<n>.npy` — Sintel trailer (Blender), T<n> 1-second timesteps, generated on Mac CPU + bf16.
- `golden_pred_sintel_T<n>.meta.json` — sidecar metadata.

## Regenerating

Requires `HF_TOKEN` with access to the gated `facebook/tribev2` model (accept terms at <https://huggingface.co/facebook/tribev2>).

```bash
cd /Users/kirby/Documents/cortyze/cortyze_product
export HF_TOKEN=hf_...
/Users/kirby/Documents/cortyze/tribev2/.venv/bin/python scripts/build_fixture.py
```

Wall time: ~10–20 min on an 8GB M3 Mac. Peak RAM: ~5 GB. The script is idempotent — re-running reuses the HuggingFace weight cache (`~/.cache/huggingface/`) and the local feature cache.

See [scripts/build_fixture.py](../../scripts/build_fixture.py) for the full pipeline and [STAGE1_PLAN.md](../../STAGE1_PLAN.md) for the broader context.

## Why committed to git

Each `.npy` is ~250 KB to ~5 MB depending on event count. Committing them keeps golden tests working from a fresh clone with zero extra setup. `.gitattributes` marks `*.npy` as binary so diffs stay quiet.
