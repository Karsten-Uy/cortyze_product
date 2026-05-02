"""Run the post pipeline (image + optional audio + optional caption)
through TRIBE v2 in CPU+bf16 mode and save the result as a fixture.

Mirrors `build_fixture.py` but for static-post inputs. Use this when you
want mock mode to return real-data brain scores derived from an image
you actually care about, instead of synthetic noise or sintel-derived
video scores.

Run from the existing tribev2 venv (the only one with neuralset /
transformers / torch / pandas installed):

    cd /Users/kirby/Documents/cortyze/cortyze_product
    export HF_TOKEN=hf_...
    /Users/kirby/Documents/cortyze/tribev2/.venv/bin/python \\
      scripts/build_post_fixture.py --image-url https://...

Wall time on 8GB M3 Mac:
  - image-only (caption optional, no audio):  ~30-50 min (1 V-JEPA chunk)
  - image + 15s audio (with WhisperX):        ~50-90 min
  - image + 60s audio:                        ~90-150 min

Outputs go to tests/fixtures/. Mock mode automatically picks up any
golden_pred_*.npy in that directory.

Prereqs (besides the tribev2 venv):
  - ffmpeg on PATH:    brew install ffmpeg
  - HF_TOKEN env var with access to facebook/tribev2 + meta-llama/Llama-3.2-3B
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

# Make the cortyze_product package importable from the tribev2 venv: this
# script lives in cortyze_product/scripts/, so its parent is the package
# root that contains gpu_worker/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Reuse the same CPU+bf16 monkey-patches as build_fixture.py — they have
# to run BEFORE any tribev2 import (those imports trigger neuralset
# extractor class definitions that we're patching).
from cortyze_product.scripts.fixtures.build_fixture import (  # noqa: E402
    apply_mac_cpu_patches,
    sha256_file,
    get_tribev2_commit,
    get_model_revision,
)


DEFAULT_CACHE_DIR = "../tribev2/cache"


def _content_hash(image_url: str, audio_url: str | None, caption: str | None) -> str:
    """Short hash of all inputs so unique posts write to unique files."""
    blob = "\n".join([image_url, audio_url or "", caption or ""])
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


def _download_url(url: str, dest: Path) -> None:
    """Lightweight downloader that doesn't require neuralset."""
    import urllib.request

    req = urllib.request.Request(
        url, headers={"User-Agent": "cortyze-build-post-fixture/1.0"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-url",
        required=True,
        help="URL to a single image (JPG/PNG/WebP). Required.",
    )
    parser.add_argument(
        "--audio-url",
        default=None,
        help="Optional. URL to an audio track (MP3/WAV/M4A). When set, "
        "TRIBE runs WhisperX over it for word timestamps.",
    )
    parser.add_argument(
        "--caption",
        default=None,
        help="Optional. Caption text — synthesized into Word events at "
        "150 wpm reading rate, appended to whatever audio produces.",
    )
    parser.add_argument(
        "--seconds-per-image",
        type=float,
        default=2.5,
        help="How long the image is held in the synthesized video. "
        "Only matters for V-JEPA chunking; >2.6s splits into 2 chunks.",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help=(
            "Path prefix; _T<n>.npy and _T<n>.meta.json are appended. "
            "Defaults to tests/fixtures/golden_pred_post_<contenthash> so "
            "different posts (image/audio/caption combos) write to distinct "
            "files (no overwrites)."
        ),
    )
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing fixture at the resolved output path.",
    )
    args = parser.parse_args()

    content_hash = _content_hash(args.image_url, args.audio_url, args.caption)
    output_stem_value = (
        args.output_stem
        or f"tests/fixtures/golden_pred_post_{content_hash}"
    )

    if not os.environ.get("HF_TOKEN"):
        print(
            "ERROR: HF_TOKEN env var not set.\n"
            "  1. Create a token at https://huggingface.co/settings/tokens\n"
            "  2. Accept gated terms at https://huggingface.co/facebook/tribev2\n"
            "                       and https://huggingface.co/meta-llama/Llama-3.2-3B\n"
            "  3. export HF_TOKEN=hf_...",
            file=sys.stderr,
        )
        return 1

    if shutil.which("ffmpeg") is None:
        print(
            "ERROR: ffmpeg not found on PATH. Install via `brew install ffmpeg`.",
            file=sys.stderr,
        )
        return 1

    if not args.caption and not args.audio_url:
        print(
            "WARNING: no caption or audio supplied. The post will use 1 of 3 "
            "modalities — Engagement / Brand Recall scores will be very weak. "
            "Pass --caption or --audio-url for a more representative fixture.",
            file=sys.stderr,
        )

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_stem = Path(output_stem_value)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    print("Applying CPU + bf16 patches...", flush=True)
    apply_mac_cpu_patches()

    # post_assembly imports pandas, which isn't loaded until after patches —
    # so this import is below the patch call by design.
    from gpu_worker.post_assembly import (
        assemble_post_video,
        caption_to_word_events,
    )
    from tribev2.demo_utils import TribeModel

    # --- Download inputs --------------------------------------------------
    image_path = cache_dir / f"post_image_{uuid4().hex[:8]}"
    print(f"Downloading {args.image_url} -> {image_path}", flush=True)
    _download_url(args.image_url, image_path)
    image_sha = sha256_file(image_path)

    audio_path: Path | None = None
    audio_sha: str | None = None
    if args.audio_url:
        audio_path = cache_dir / f"post_audio_{uuid4().hex[:8]}"
        print(f"Downloading {args.audio_url} -> {audio_path}", flush=True)
        _download_url(args.audio_url, audio_path)
        audio_sha = sha256_file(audio_path)

    # --- Assemble the synthetic post video -------------------------------
    print(
        f"Assembling post video (1 image, hold={args.seconds_per_image}s"
        f"{', + audio mux' if audio_path else ''})...",
        flush=True,
    )
    post_video_path, duration_s = assemble_post_video(
        [image_path], audio_path, args.seconds_per_image, out_dir=cache_dir
    )

    # --- Load TRIBE + run -------------------------------------------------
    print(
        "Loading TribeModel (first run pulls ~10GB to ~/.cache/huggingface/)...",
        flush=True,
    )
    t0 = time.monotonic()
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=cache_dir,
        device="cpu",
    )

    print(
        "Extracting events from the synthetic post video"
        f"{' (WhisperX over audio)' if audio_path else ' (silent video)'}...",
        flush=True,
    )
    df = model.get_events_dataframe(video_path=post_video_path)

    if args.caption:
        # Place caption events past whatever audio produced so they don't
        # collide on the timeline. Mirrors gpu_worker.inference._predict_post.
        if "start" in df.columns and len(df) > 0:
            ends = df["start"].astype(float)
            if "duration" in df.columns:
                ends = ends + df["duration"].astype(float)
            base_time_s = float(max(ends.max(), duration_s))
        else:
            base_time_s = duration_s
        caption_df = caption_to_word_events(args.caption, base_time_s=base_time_s)
        print(
            f"Appending {len(caption_df)} caption Word events at base_time={base_time_s:.1f}s",
            flush=True,
        )
        # Concat tolerates column-set differences (caption rows lack
        # extractor-emitted embedding columns; pd.concat fills NaN).
        import pandas as pd

        df = pd.concat([df, caption_df], ignore_index=True, sort=False)
        if "start" in df.columns:
            df = df.sort_values("start", kind="mergesort").reset_index(drop=True)

    # Balance the events to keep RAM in check on M3 (one of each type).
    # Drop any rows with NaN in 'type' (caption rows that didn't get extractor
    # processing, if the model rejects them).
    if "type" in df.columns:
        balanced_df = df.groupby("type").head(1)
    else:
        balanced_df = df.head(4)
    events_kept = (
        balanced_df["type"].value_counts().to_dict()
        if "type" in balanced_df.columns
        else {}
    )
    print(f"  Events kept: {events_kept}", flush=True)

    print(
        f"Running predictions on {len(balanced_df)} events (slow part)...",
        flush=True,
    )
    preds, _segments = model.predict(events=balanced_df)
    wall_time = int(time.monotonic() - t0)

    if preds.ndim != 2 or preds.shape[1] != 20484:
        print(
            f"ERROR: unexpected preds shape {preds.shape}; want (T, 20484)",
            file=sys.stderr,
        )
        return 1

    # --- Save fixture + sidecar ------------------------------------------
    T = preds.shape[0]
    npy_path = Path(f"{output_stem}_T{T}.npy")
    meta_path = Path(f"{output_stem}_T{T}.meta.json")

    if npy_path.exists() and not args.force:
        print(
            f"ERROR: {npy_path} already exists. Pass --force to overwrite, "
            f"or change --output-stem.",
            file=sys.stderr,
        )
        return 1

    np.save(npy_path, preds)

    meta = {
        "content_type": "post",
        "image_url": args.image_url,
        "image_sha256": image_sha,
        "audio_url": args.audio_url,
        "audio_sha256": audio_sha,
        "caption": args.caption,
        "seconds_per_image": args.seconds_per_image,
        "synthetic_video_duration_s": duration_s,
        "shape": list(preds.shape),
        "dtype": str(preds.dtype),
        "events_kept": {str(k): int(v) for k, v in events_kept.items()},
        "tribev2_commit": get_tribev2_commit(),
        "model_repo": "facebook/tribev2",
        "model_revision": get_model_revision(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_on": f"{platform.system()} {platform.machine()} (CPU+bf16)",
        "wall_time_seconds": wall_time,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    # --- Cleanup the synthetic video; keep originals for re-runs ---------
    post_video_path.unlink(missing_ok=True)

    print(f"\nSaved {tuple(preds.shape)} {preds.dtype} -> {npy_path}")
    print(f"Saved sidecar -> {meta_path}")
    print(f"Wall time: {wall_time}s")
    print(
        "\nMock mode picks up the first golden_pred_*.npy by sorted order. "
        "If you have a sintel fixture committed, temporarily move it aside "
        "or rename this output to sort first."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
