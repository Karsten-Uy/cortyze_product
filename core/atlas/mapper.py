"""Vertex-to-region aggregation for the 8 marketing brain regions.

Loads the precomputed fsaverage5 DK label array (built by
scripts/build_atlas_labels.py) and exposes `aggregate(preds)` which
collapses a (T, 20484) prediction array into 8 raw mean activations.

Per-region vertex indices are pre-computed at import so `aggregate()` is
just `mean(axis=0)` + a per-region masked mean per request.
"""

import json
from pathlib import Path

import numpy as np

from .regions import REGIONS

_HERE = Path(__file__).resolve().parent
_LABELS_NPY = _HERE / "data" / "fsaverage5_dk_labels.npy"
_LABELS_JSON = _HERE / "data" / "fsaverage5_dk_labels.json"

VERTEX_LABELS: np.ndarray = np.load(_LABELS_NPY)
with open(_LABELS_JSON) as _f:
    DK_LABEL_TO_ID: dict[str, int] = json.load(_f)["label_to_id"]


def _build_region_vertex_indices() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for region_key, dk_names in REGIONS.items():
        ids = [DK_LABEL_TO_ID[n] for n in dk_names if n in DK_LABEL_TO_ID]
        if not ids:
            raise ValueError(
                f"Region {region_key!r} has no DK labels in the labels file. "
                f"None of {dk_names!r} found in {_LABELS_JSON}. "
                "Re-run scripts/build_atlas_labels.py."
            )
        idx = np.where(np.isin(VERTEX_LABELS, ids))[0]
        if len(idx) == 0:
            raise ValueError(f"Region {region_key!r} resolves to zero vertices.")
        out[region_key] = idx
    return out


REGION_VERTICES: dict[str, np.ndarray] = _build_region_vertex_indices()


def aggregate(preds: np.ndarray) -> dict[str, float]:
    """Aggregate (T, 20484) cortical predictions to 8 raw region activations.

    Mean across time, then per-region mean across that region's vertices.
    """
    if preds.ndim != 2 or preds.shape[1] != 20484:
        raise ValueError(f"Expected (T, 20484), got {preds.shape}")
    time_mean = preds.mean(axis=0)
    return {region: float(time_mean[idx].mean()) for region, idx in REGION_VERTICES.items()}
