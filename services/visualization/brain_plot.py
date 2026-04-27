"""Render `(T, 20484)` cortical predictions as a base64 PNG brain heatmap.

Uses nilearn's surface plotting (already in the dep tree) on the fsaverage5
mesh that nilearn caches at ~/nilearn_data after the first call. The
per-vertex map is the time-mean of predictions, masked to vertices that
belong to one of the 8 marketing regions in core.atlas — so the
visualization shows exactly what the BrainReport scores cover.

Two hemispheres rendered side-by-side, lateral view, hot colormap with
a subtle sulcal background. ~1-2s on first call (surface mesh download),
~0.5s after caching. PNG is small (~50-80 KB), inlined in BrainReport
as base64 so the frontend renders it without a second round-trip.
"""

import base64
import io
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")  # headless — must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from core.atlas.mapper import REGION_VERTICES  # noqa: E402


@lru_cache(maxsize=1)
def _fsaverage():
    from nilearn import datasets

    return datasets.fetch_surf_fsaverage("fsaverage5")


@lru_cache(maxsize=1)
def _region_mask() -> np.ndarray:
    """(20484,) bool — True for vertices in any of the 8 marketing regions."""
    mask = np.zeros(20484, dtype=bool)
    for idx in REGION_VERTICES.values():
        mask[idx] = True
    return mask


def render_brain_png(preds: np.ndarray) -> str:
    """Return a base64-encoded PNG of the cortical activation map."""
    from nilearn import plotting

    fsaverage = _fsaverage()
    mask = _region_mask()

    time_mean = preds.mean(axis=0).astype(np.float64)
    masked = np.where(mask, time_mean, np.nan)
    lh, rh = masked[:10242], masked[10242:]

    finite = time_mean[mask]
    vmax = float(np.nanpercentile(np.abs(finite), 99)) if finite.size else 1.0
    vmin = -vmax

    fig, axes = plt.subplots(
        1, 2, figsize=(10, 4), subplot_kw={"projection": "3d"}
    )
    fig.patch.set_facecolor("#0a0a0a")
    for ax in axes:
        ax.set_facecolor("#0a0a0a")

    plotting.plot_surf_stat_map(
        fsaverage["pial_left"],
        lh,
        hemi="left",
        view="lateral",
        bg_map=fsaverage["sulc_left"],
        cmap="hot",
        colorbar=False,
        vmax=vmax,
        threshold=vmax * 0.05,
        axes=axes[0],
        figure=fig,
    )
    plotting.plot_surf_stat_map(
        fsaverage["pial_right"],
        rh,
        hemi="right",
        view="lateral",
        bg_map=fsaverage["sulc_right"],
        cmap="hot",
        colorbar=False,
        vmax=vmax,
        threshold=vmax * 0.05,
        axes=axes[1],
        figure=fig,
    )

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        facecolor="#0a0a0a",
        dpi=80,
        pad_inches=0.1,
    )
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")
