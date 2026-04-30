"""TRIBE v2 inference, GPU-backed.

`BrainPredictor` wraps `tribev2.demo_utils.TribeModel`. The heavy work
(loading ~10 GB of weights onto GPU, building the multimodal Transformer)
happens once at import; per-request work is just download + forward pass.

Designed to be importable from both pod-mode (FastAPI) and serverless-mode
(`def handler(event)`) entry points in handler.py. Same predictor instance
serves both.

Two content shapes:

- **video** (legacy): single MP4 URL → tribev2 runs its full pipeline.
- **post** (static or carousel): 1-20 image URLs + optional audio +
  optional caption. Each image is held for `seconds_per_image` seconds
  in the synthesized MP4 (so 1-image posts and N-image carousels share
  the same code path). V-JEPA reads transitions between images as scene
  changes — meaningful signal for memorability + attention regions.
  See SCALING.md for the latency math.
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch

from tribev2.demo_utils import TribeModel, download_file

from gpu_worker.post_assembly import (
    assemble_post_video,
    caption_to_word_events,
)


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
        self,
        content_url: str | None = None,
        content_type: str = "video",
        *,
        image_urls: list[str] | None = None,
        audio_url: str | None = None,
        caption: str | None = None,
        seconds_per_image: float = 2.5,
    ) -> tuple[np.ndarray, "object"]:
        """Download content, run TRIBE v2, return (predictions, events_df).

        Dispatches on `content_type`:
        - `"video"` (default) → `_predict_video(content_url)`
        - `"post"` → `_predict_post(image_urls, audio_url, caption, seconds_per_image)`
                     handles 1-20 images uniformly
        - anything else → NotImplementedError

        Returns `(predictions, events_df)` where predictions is `(T, 20484)`
        float32 cortical activations and events_df is the timestamped
        events DataFrame the Stage 2 suggestion engine consumes for
        moment annotation. handler.py serializes both for transport.
        """
        if content_type == "video":
            if not content_url:
                raise ValueError("content_type='video' requires content_url")
            return self._predict_video(content_url)
        if content_type == "post":
            if not image_urls or len(image_urls) < 1:
                raise ValueError(
                    "content_type='post' requires image_urls with >= 1 entry"
                )
            return self._predict_post(
                image_urls, audio_url, caption, seconds_per_image
            )
        raise NotImplementedError(
            f"content_type={content_type!r} not supported. "
            "Use 'video' for MP4 URLs or 'post' for image-based content."
        )

    def _predict_video(self, content_url: str) -> tuple[np.ndarray, "object"]:
        """Legacy single-MP4 path. Unchanged from Stage 1.2."""
        local_path = Path(f"/tmp/{uuid4().hex}.mp4")
        try:
            download_file(content_url, local_path)
            events_df = self.model.get_events_dataframe(video_path=local_path)
            preds, events_df = self._run_predict(events_df)
        finally:
            local_path.unlink(missing_ok=True)
        return preds, events_df

    def _predict_post(
        self,
        image_urls: list[str],
        audio_url: str | None,
        caption: str | None,
        seconds_per_image: float,
    ) -> tuple[np.ndarray, "object"]:
        """Post path — 1 to 20 images, concatenated into a single MP4.

        Pipeline (same for single-image and carousel posts):
          1. Download all images (and audio if present)
          2. Assemble synthetic MP4 with each image held for
             `seconds_per_image` seconds; mux audio if supplied
          3. Run tribev2's `get_events_dataframe(video_path=)` —
             WhisperX transcribes any speech in the audio
          4. Append synthetic Word events from the caption (if any) at
             timestamps past the end of any audio-derived events
          5. Run TRIBE forward pass

        Cleanup of /tmp files is best-effort in `finally`.
        """
        image_paths: list[Path] = []
        audio_path: Path | None = None
        post_video_path: Path | None = None

        try:
            for i, url in enumerate(image_urls):
                p = Path(f"/tmp/{uuid4().hex}_p{i:02d}_image")
                download_file(url, p)
                image_paths.append(p)

            if audio_url:
                audio_path = Path(f"/tmp/{uuid4().hex}_p_audio")
                download_file(audio_url, audio_path)

            post_video_path, duration_s = assemble_post_video(
                image_paths, audio_path, seconds_per_image
            )

            events_df = self.model.get_events_dataframe(
                video_path=post_video_path
            )

            if caption:
                # Place caption events after any audio-derived events so
                # synthetic timestamps don't collide with whisperx-
                # transcribed words. If the audio had no speech,
                # base_time_s=0 just populates events from the start.
                base_time_s = self._next_free_time(events_df, fallback=duration_s)
                caption_df = caption_to_word_events(
                    caption, base_time_s=base_time_s
                )
                events_df = self._merge_events(events_df, caption_df)

            preds, events_df = self._run_predict(events_df)
        finally:
            for path in (*image_paths, audio_path, post_video_path):
                if path is not None:
                    path.unlink(missing_ok=True)

        return preds, events_df

    def _run_predict(self, events_df) -> tuple[np.ndarray, "object"]:
        """Common forward pass + shape validation."""
        # verbose=False disables tqdm progress bars. Long inference can
        # outlive the HTTP client (Cloudflare 100s proxy timeout); when
        # FastAPI's threadpool worker loses its stderr handle, tqdm
        # crashes with "I/O operation on closed file." mid-batch.
        preds, _segments = self.model.predict(events=events_df, verbose=False)
        if preds.ndim != 2 or preds.shape[1] != 20484:
            raise RuntimeError(
                f"unexpected preds shape {preds.shape}; want (T, 20484)"
            )
        return preds.astype(np.float32), events_df

    @staticmethod
    def _next_free_time(events_df, fallback: float) -> float:
        """Find the latest end-time across existing events, or fallback if none.

        tribev2's events DataFrame uses 'start' and 'duration' columns. We
        place caption Word events strictly after this so the synthetic
        timestamps don't collide with audio-derived ones.
        """
        if events_df is None or len(events_df) == 0:
            return float(fallback)
        if "start" not in events_df.columns:
            return float(fallback)
        ends = events_df["start"].astype(float)
        if "duration" in events_df.columns:
            ends = ends + events_df["duration"].astype(float)
        return float(max(ends.max(), fallback))

    @staticmethod
    def _merge_events(base_df, extra_df):
        """Concat two events DataFrames safely.

        Tolerates column-set differences: extra_df only carries the raw
        event columns (type/start/duration/text), but base_df may have
        additional embedding columns from the extractors. pd.concat with
        sort=False preserves both column sets, filling gaps with NaN.

        If the model's predict() rejects rows missing embedding columns,
        the runtime error will be visible in logs and is the signal to
        invoke a text-extractor on the caption rows explicitly.
        """
        if base_df is None or len(base_df) == 0:
            return extra_df
        if extra_df is None or len(extra_df) == 0:
            return base_df
        merged = pd.concat([base_df, extra_df], ignore_index=True, sort=False)
        if "start" in merged.columns:
            merged = merged.sort_values("start", kind="mergesort").reset_index(
                drop=True
            )
        return merged
