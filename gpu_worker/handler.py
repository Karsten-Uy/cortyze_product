"""GPU worker entry point — pod mode (FastAPI) or serverless mode.

Selects mode at startup via the RUNPOD_SERVERLESS env var. Both modes use
the same BrainPredictor instance (loaded once on import), so swapping
between Pod-for-iteration and Serverless-for-prod is a one-flag flip.

Pod mode (default):
    python -u handler.py        →  FastAPI on :8000

Serverless mode (RunPod runtime sets RUNPOD_SERVERLESS=1):
    python -u handler.py        →  runpod.serverless.start(...)

Response shape (both modes):
    {
      "shape": [T, 20484],
      "dtype": "float16",
      "data_b64": "<base64 of compressed npz>",
      "events": [{"type": "Word", "start_s": 1.2, "duration_s": 0.4, "text": "..."}, ...],
      "elapsed_ms": <int>
    }
"""

from __future__ import annotations

import base64
import io
import os
import time

import numpy as np

from gpu_worker.inference import BrainPredictor
from services.suggestions.moments import serialize_events_dataframe


_predictor: BrainPredictor | None = None


def _get_predictor() -> BrainPredictor:
    global _predictor
    if _predictor is None:
        _predictor = BrainPredictor()
    return _predictor


def _serialize(arr: np.ndarray, events_df) -> dict:
    """Compress (T, 20484) to base64 NPZ + serialize events for JSON transport."""
    buf = io.BytesIO()
    np.savez_compressed(buf, preds=arr.astype(np.float16))
    events = [e.model_dump() for e in serialize_events_dataframe(events_df)]
    return {
        "shape": list(arr.shape),
        "dtype": "float16",
        "data_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "events": events,
    }


def _predict_payload(input_data: dict) -> dict:
    t0 = time.monotonic()
    arr, events_df = _get_predictor().predict(
        input_data["content_url"],
        input_data.get("content_type", "video"),
    )
    payload = _serialize(arr, events_df)
    payload["elapsed_ms"] = int((time.monotonic() - t0) * 1000)
    return payload


# ---- Serverless mode (RunPod calls this) -----------------------------

def handler(event: dict) -> dict:
    """RunPod serverless entry point."""
    return _predict_payload(event["input"])


# ---- Pod mode (FastAPI) ----------------------------------------------

def _build_app():
    from fastapi import FastAPI

    app = FastAPI(title="Cortyze GPU Worker", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "device": _get_predictor().device,
        }

    @app.post("/predict")
    def predict(req: dict) -> dict:
        return _predict_payload(req)

    return app


# Lazy app reference for `uvicorn handler:app`.
def __getattr__(name: str):
    if name == "app":
        return _build_app()
    raise AttributeError(name)


if __name__ == "__main__":
    if os.environ.get("RUNPOD_SERVERLESS") == "1":
        import runpod

        # Eagerly load the model so cold start happens before the first
        # invocation, not during it.
        _get_predictor()
        runpod.serverless.start({"handler": handler})
    else:
        import uvicorn

        # Build the app eagerly (loads model now, before HTTP server starts).
        app = _build_app()
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
