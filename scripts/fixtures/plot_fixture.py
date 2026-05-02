"""Render a brain heatmap PNG from the (T, 20484) fixture.

Uses tribev2.plotting which requires the tribev2 [plotting] extras
(matplotlib, nilearn, etc.). Run from the tribev2 venv:

    /Users/kirby/Documents/cortyze/tribev2/.venv/bin/python \\
        scripts/plot_fixture.py

Output: docs/brain_demo.png — a marketing-grade still of cortical
activation predicted on the sintel trailer. Suitable for the Stage 3
landing page.
"""

import sys
from pathlib import Path

import numpy as np

# matplotlib backend before pyplot import — headless on Mac without display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tribev2.plotting import PlotBrain  # noqa: E402


def main() -> int:
    here = Path(__file__).resolve().parent
    fixtures_dir = here.parent / "tests" / "fixtures"
    output_dir = here.parent / "docs"
    output_dir.mkdir(parents=True, exist_ok=True)

    fixtures = sorted(fixtures_dir.glob("golden_pred_*.npy"))
    if not fixtures:
        print("ERROR: no golden_pred_*.npy in tests/fixtures/", file=sys.stderr)
        return 1
    fixture = fixtures[0]
    preds = np.load(fixture)
    print(f"Loaded {fixture.name}: {preds.shape} {preds.dtype}")

    plotter = PlotBrain(mesh="fsaverage5")

    # Time-averaged activation across all timesteps -> single (20484,) map.
    # Using plot_timesteps with first 3 frames is more dynamic but needs
    # `segments` metadata which we didn't persist; the time-mean is a
    # cleaner single-frame summary anyway.
    n_frames = min(3, preds.shape[0])
    fig = plotter.plot_timesteps(
        preds[:n_frames],
        cmap="fire",
        norm_percentile=99,
        vmin=0.6,
        alpha_cmap=(0, 0.2),
        show_stimuli=False,
    )

    out_path = output_dir / "brain_demo.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
