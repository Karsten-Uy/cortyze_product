"""TRIBE v2 inference, GPU-backed.

`BrainPredictor` wraps `tribev2.demo_utils.TribeModel`. The heavy work
(loading ~10 GB of weights onto GPU, building the multimodal Transformer)
happens once at import; per-request work is just download + forward pass.

Designed to be importable from both pod-mode (FastAPI) and serverless-mode
(`def handler(event)`) entry points in handler.py. Same predictor instance
serves both.

# TODO(stage 2): handle content_type="image" (single-frame video) and
# "text" (TTS or text events). Stage 1.2 ships video-only.
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from tribev2.demo_utils import TribeModel, download_file


class BrainPredictor:
    def __init__(self, cache_folder: str | None = None):
        cache_folder = cache_folder or os.environ.get("HF_HOME", "/opt/hf_cache")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print(
                "WARNING: no CUDA detected. Falling back to CPU — inference will be "
                "minutes per call instead of seconds. This worker is designed for "
                "GPU runtimes (RunPod A40 / A100).",
                flush=True,
            )
        print(f"Loading TRIBE v2 onto {device} (cache={cache_folder})...", flush=True)
        self.model = TribeModel.from_pretrained(
            "facebook/tribev2",
            cache_folder=cache_folder,
            device=device,
        )
        self.device = device
        print("Model ready.", flush=True)

    def predict(
        self, content_url: str, content_type: str = "video"
    ) -> tuple[np.ndarray, "object"]:
        """Download content, run TRIBE v2, return (predictions, events_df).

        `predictions` is the (T, 20484) float32 cortical activation array.
        `events_df` is the timestamped events DataFrame tribev2 builds
        during preprocessing (Words, Sentences, Audio/Video chunks); the
        Stage 2 suggestion engine consumes it for moment annotation. The
        caller (handler.py) serializes both for transport.
        """
        if content_type != "video":
            raise NotImplementedError(
                f"content_type={content_type!r} not supported in Stage 1.2 "
                "(video only — image/text land in Stage 2)"
            )
        local_path = Path(f"/tmp/{uuid4().hex}.mp4")
        try:
            download_file(content_url, local_path)
            events_df = self.model.get_events_dataframe(video_path=local_path)
            # verbose=False disables tqdm progress bars. Long inference can
            # outlive the HTTP client (Cloudflare 100s proxy timeout); when
            # FastAPI's threadpool worker loses its stderr handle, tqdm
            # crashes with "I/O operation on closed file." mid-batch.
            preds, _segments = self.model.predict(events=events_df, verbose=False)
        finally:
            local_path.unlink(missing_ok=True)
        if preds.ndim != 2 or preds.shape[1] != 20484:
            raise RuntimeError(
                f"unexpected preds shape {preds.shape}; want (T, 20484)"
            )
        return preds.astype(np.float32), events_df
