"""Compute per-region (mu, sigma) calibration from the reference fixture pool.

Replaces the placeholder mu=0/sigma=1 in core/scoring/calibration.json with
values derived from ALL `golden_pred_*.npy` files in tests/fixtures/.

Approach — multi-clip pooled calibration:

  mu_region    = mean of per-clip region_means across the fixture pool.
                 Per-region (different for every region). This is the
                 "expected raw activation for this region across our
                 reference library."

  sigma_region = mean of per-clip within-region vertex stds.
                 Per-region. Reflects how heterogeneously the region
                 fires within a clip; averaged across clips for stability.

Why mean-of-vertex-stds and not cross-clip std of region_means?
  With only n=2 fixtures (typical Stage 1), cross-clip std is a single
  difference and produces a tiny, brittle sigma that pushes scores to
  0/100 saturation. Vertex-std averaged across clips is a stable
  regularizer that captures spatial heterogeneity, which is the dominant
  source of variation when n is small. As the pool grows past ~10 clips,
  switch to true cross-clip std (commented at the bottom).

Score formula recap (core/scoring/scoring.py):
  z = (raw_region_mean - mu) / sigma
  score = sigmoid(z) * 100

So a clip whose region_mean equals mu scores exactly 50 — i.e., the
"average reference activation". Above-mean → > 50, below → < 50.

Usage:
    uv run python scripts/calibrate_from_fixture.py
"""

import json
import sys
from pathlib import Path

import numpy as np

from core.atlas.mapper import REGION_VERTICES


def main() -> int:
    here = Path(__file__).resolve().parent
    fixtures_dir = here.parent / "tests" / "fixtures"
    calibration_path = here.parent / "core" / "scoring" / "calibration.json"

    fixtures = sorted(fixtures_dir.glob("golden_pred_*.npy"))
    if not fixtures:
        print("ERROR: no golden_pred_*.npy in tests/fixtures/", file=sys.stderr)
        return 1

    print(f"Pooling {len(fixtures)} reference fixture(s):")
    per_clip_region_means: dict[str, list[float]] = {r: [] for r in REGION_VERTICES}
    per_clip_region_sigmas: dict[str, list[float]] = {r: [] for r in REGION_VERTICES}

    for fixture in fixtures:
        preds = np.load(fixture)
        if preds.ndim != 2 or preds.shape[1] != 20484:
            print(f"  SKIP {fixture.name}: shape {preds.shape}", file=sys.stderr)
            continue
        print(f"  {fixture.name}  shape={preds.shape}")
        time_mean = preds.mean(axis=0)
        for region, vertex_idx in REGION_VERTICES.items():
            region_vals = time_mean[vertex_idx]
            per_clip_region_means[region].append(float(region_vals.mean()))
            per_clip_region_sigmas[region].append(float(region_vals.std()))

    calibration: dict[str, dict[str, float]] = {}
    for region in REGION_VERTICES:
        means = per_clip_region_means[region]
        sigmas = per_clip_region_sigmas[region]
        mu = float(np.mean(means))
        sigma = float(np.mean(sigmas))
        if sigma == 0.0:
            sigma = 1e-3
        calibration[region] = {"mu": mu, "sigma": sigma}

    calibration_path.write_text(json.dumps(calibration, indent=2) + "\n")
    print(f"\nWrote {calibration_path}\n")

    # Show what each fixture would score under the new calibration so you
    # can spot extreme bias before shipping.
    from math import exp

    def _score(raw: float, mu: float, sigma: float) -> float:
        z = (raw - mu) / sigma
        return 100.0 / (1.0 + exp(-z)) if z >= 0 else 100.0 * exp(z) / (1.0 + exp(z))

    print(f"  {'region':20s} {'mu':>10s} {'sigma':>10s}  scores per clip")
    for region, c in calibration.items():
        per_clip = [
            _score(m, c["mu"], c["sigma"])
            for m in per_clip_region_means[region]
        ]
        score_str = "  ".join(f"{s:5.1f}" for s in per_clip)
        print(f"  {region:20s} {c['mu']:>+10.4f} {c['sigma']:>10.4f}  {score_str}")

    print(
        f"\n  (column order = sorted fixture names: "
        f"{', '.join(f.stem for f in fixtures)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Stage 2 hook — once the reference pool has >= 10 clips, replace
# `sigma = mean(per-clip vertex stds)` with `sigma = std(per-clip
# region_means)`. That converts the calibration from "spatial
# heterogeneity regularizer" to "true cross-clip distribution std",
# which is what you actually want when n is large enough to estimate it.
