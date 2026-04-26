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
    """Deterministic synthetic predictions for development.

    Loads any real fixture from `tests/fixtures/golden_pred_*.npy` if one
    is present (produced by scripts/build_fixture.py). Falls back to an
    on-the-fly synthetic array seeded by hash(content_url) so repeat calls
    return the same data.
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

        seed = abs(hash(content_url)) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.normal(0.0, 1.0, size=(24, 20484)).astype(np.float32)


def get_client() -> RunPodClientProtocol:
    """Return mock client unless RUNPOD_ENDPOINT_ID is set (Stage 1.2)."""
    if os.environ.get("RUNPOD_ENDPOINT_ID"):
        raise NotImplementedError(
            "Real RunPodClient is Stage 1 Phase 1.2. Unset RUNPOD_ENDPOINT_ID for mock mode."
        )
    return MockRunPodClient()
