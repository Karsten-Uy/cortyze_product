"""Compute per-region (mu, sigma) calibration constants from a real fixture.

Replaces the placeholder mu=0/sigma=1 in core/scoring/calibration.json with
values derived from the fixture's (T, 20484) array.

Approach (Stage 1 expedient — single clip):

  mu     = mean of per-region means across all 8 regions (i.e., this clip's
           overall cortical activation baseline). One value, shared.
  sigma  = std of activations across vertices WITHIN that region (its
           spatial heterogeneity). Per-region.

Why not mu=region_mean: would force every region on this clip to score
exactly 50, since z = (region_mean - region_mean) = 0 → sigmoid → 50. No
inter-region differentiation. The snapshot tests would be all-50 forever.

Why not mu=0: defensible, but ignores the actual signal scale of THIS
model on real data. Using the clip's overall baseline as the centerpoint
means regions *above* baseline (e.g. visual cortex on a video) score >50
and regions *below* baseline score <50 — which is what you want.

Stage 2 (with ~30 reference clips on RunPod) replaces both fields with
proper per-region cross-clip statistics.

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
    fixture = fixtures[0]
    print(f"Loading {fixture}")
    preds = np.load(fixture)
    if preds.ndim != 2 or preds.shape[1] != 20484:
        print(f"ERROR: fixture has unexpected shape {preds.shape}", file=sys.stderr)
        return 1

    time_mean = preds.mean(axis=0)

    per_region_mean = {
        region: float(time_mean[idx].mean())
        for region, idx in REGION_VERTICES.items()
    }
    clip_baseline = float(np.mean(list(per_region_mean.values())))

    calibration: dict[str, dict[str, float]] = {}
    for region, vertex_idx in REGION_VERTICES.items():
        sigma = float(time_mean[vertex_idx].std())
        if sigma == 0.0:
            sigma = 1e-3
        calibration[region] = {"mu": clip_baseline, "sigma": sigma}

    calibration_path.write_text(json.dumps(calibration, indent=2) + "\n")
    print(f"\nClip baseline mu = {clip_baseline:+.4f} (used for all 8 regions)")
    print(f"Wrote {calibration_path}\n")
    print(f"  {'region':20s} {'region_mean':>12s} {'sigma':>10s}  → score on this clip")
    from math import exp

    def _score(raw: float, mu: float, sigma: float) -> float:
        z = (raw - mu) / sigma
        return 100.0 / (1.0 + exp(-z)) if z >= 0 else 100.0 * exp(z) / (1.0 + exp(z))

    for region, c in calibration.items():
        rm = per_region_mean[region]
        s = _score(rm, c["mu"], c["sigma"])
        print(f"  {region:20s} {rm:>+12.4f} {c['sigma']:>10.4f}  → {s:5.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
