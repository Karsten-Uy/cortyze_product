"""Per-timestep cortical aggregation.

`aggregate()` in mapper.py collapses time — useful for the overall report
but loses the diagnostic signal Stage 2 needs ("amygdala dipped at 0:14").
This module preserves the time axis: for each region, return its mean
activation per second of input.

Pure NumPy. No I/O. No LLM. Used by services/suggestions/moments.py.
"""

import json
from pathlib import Path

import numpy as np

from .mapper import REGION_VERTICES

_CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "scoring" / "calibration.json"
with open(_CALIBRATION_PATH) as _f:
    _CALIBRATION: dict[str, dict[str, float]] = json.load(_f)


def aggregate_per_timestep(preds: np.ndarray) -> dict[str, np.ndarray]:
    """Collapse vertices into 8 region time-series, keep time axis.

    Input: `(T, 20484)` float32, 1 timestep = 1 second per TRIBE v2.
    Output: dict mapping each region key to a `(T,)` array of mean
            activation across that region's vertices at each timestep.
    """
    if preds.ndim != 2 or preds.shape[1] != 20484:
        raise ValueError(f"Expected (T, 20484), got {preds.shape}")
    return {
        region: preds[:, idx].mean(axis=1).astype(np.float32)
        for region, idx in REGION_VERTICES.items()
    }


def normalize_per_timestep(
    raw_per_region: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Sigmoid-normalize each region's time series to 0–100.

    Same z-score-then-sigmoid as `core.scoring.normalize.normalize`,
    vectorized over the time axis. Numerically stable for extreme z.
    """
    out: dict[str, np.ndarray] = {}
    for region, raw_series in raw_per_region.items():
        cal = _CALIBRATION[region]
        z = (raw_series.astype(np.float64) - cal["mu"]) / cal["sigma"]
        # Branch on sign for numerical stability with large |z|.
        positive = z >= 0
        scores = np.empty_like(z)
        scores[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        e = np.exp(z[~positive])
        scores[~positive] = e / (1.0 + e)
        out[region] = (100.0 * scores).astype(np.float32)
    return out
