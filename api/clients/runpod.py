"""RunPod GPU worker client.

Stage 1 ships with a `MockRunPodClient` that returns a deterministic
synthetic (T, 20484) array — the API can serve real BrainReports without
a GPU. The real `RunPodClient` lands in Stage 1 Phase 1.2 and matches the
same protocol so swapping is one env-var flip.
"""

import os
from pathlib import Path
from typing import Protocol

import numpy as np


class RunPodClientProtocol(Protocol):
    def predict(self, content_url: str, content_type: str) -> np.ndarray:
        """Return (T, 20484) float32 cortical predictions."""
        ...


class MockRunPodClient:
    """Synthetic predictions for development.

    If a real fixture from `tests/fixtures/golden_pred_*.npy` exists
    (produced by scripts/build_fixture.py), it is returned as-is — that's
    real-data mode.

    Otherwise, generates a fresh random `(T, 20484)` array on every call.
    Per-region biases are drawn from N(0, 2) per call, so the 8 region
    means actually differ from each other and from one call to the next.
    Without that, averaging hundreds of N(0, 1) vertices per region
    converges to ~0 (LLN) and sigmoid maps that to ~50 across the board —
    which looks like a constant value in the UI.
    """

    def __init__(self, fixtures_dir: Path | None = None):
        self.fixtures_dir = fixtures_dir or (
            Path(__file__).resolve().parents[2] / "tests" / "fixtures"
        )

    def predict(self, content_url: str, content_type: str) -> np.ndarray:
        for fixture in sorted(self.fixtures_dir.glob("golden_pred_*.npy")):
            preds = np.load(fixture)
            if preds.ndim == 2 and preds.shape[1] == 20484:
                return preds.astype(np.float32)

        # Imported lazily so loading the API doesn't pay the atlas-labels
        # read cost in mock-fixture mode.
        from core.atlas.mapper import REGION_VERTICES

        rng = np.random.default_rng()
        T = 24
        preds = rng.normal(0.0, 0.3, size=(T, 20484)).astype(np.float32)
        for vertex_idx in REGION_VERTICES.values():
            preds[:, vertex_idx] += float(rng.normal(0.0, 2.0))
        return preds


def get_client() -> RunPodClientProtocol:
    """Return mock client unless RUNPOD_ENDPOINT_ID is set (Stage 1.2)."""
    if os.environ.get("RUNPOD_ENDPOINT_ID"):
        raise NotImplementedError(
            "Real RunPodClient is Stage 1 Phase 1.2. Unset RUNPOD_ENDPOINT_ID for mock mode."
        )
    return MockRunPodClient()
