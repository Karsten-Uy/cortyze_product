"""Convert raw cortical activations (TRIBE v2 output) to 0-100 region scores.

Stage 1 minimum: per-region z-score followed by sigmoid scaled to [0, 100].

    z = (raw - mu) / sigma
    score = 100 / (1 + exp(-z))

Calibration constants (mu, sigma per region) live in `calibration.json` next
to this module so they can be updated without code changes — Stage 2 simply
rewrites the JSON once the reference ad library has run through TRIBE v2.

The placeholder constants (mu=0, sigma=1) are deliberate: absolute scores
are meaningless until Stage 2 lands real calibration data, but **rank order
across content is preserved**, which is what diagnosis needs.

# TODO(stage 2): replace placeholder mu=0/sigma=1 with real per-region
# mean/std from ~30 reference clips through TRIBE v2.
"""

import json
import math
from pathlib import Path

_CALIBRATION_PATH = Path(__file__).parent / "calibration.json"
with open(_CALIBRATION_PATH) as _f:
    CALIBRATION: dict[str, dict[str, float]] = json.load(_f)


def _sigmoid(z: float) -> float:
    # Branch on sign so we never compute exp(large positive) — that overflows
    # for |z| > ~709 with double precision. Both branches are mathematically
    # equivalent to 1 / (1 + exp(-z)).
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def normalize(raw_region_activations: dict[str, float]) -> dict[str, float]:
    """Map raw activations to 0-100 region scores via per-region z + sigmoid.

    Input and output share the 8 region keys (see core.atlas.regions).
    """
    out: dict[str, float] = {}
    for region, raw in raw_region_activations.items():
        cal = CALIBRATION[region]
        z = (raw - cal["mu"]) / cal["sigma"]
        out[region] = 100.0 * _sigmoid(z)
    return out
