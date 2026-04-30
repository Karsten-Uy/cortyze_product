"""Build a real (T, 20484) brain-prediction fixture using TRIBE v2 in CPU+bf16 mode.

Run from the existing tribev2 venv (the only one with neuralset/transformers/torch
installed):

    cd /Users/kirby/Documents/cortyze/cortyze_product
    export HF_TOKEN=hf_...
    /Users/kirby/Documents/cortyze/tribev2/.venv/bin/python scripts/build_fixture.py

Wall time ~10-20 min on 8GB M3 Mac. Outputs go to tests/fixtures/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

DEFAULT_VIDEO_URL = "https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4"
DEFAULT_CACHE_DIR = "../tribev2/cache"
TRIBEV2_REPO = Path("/Users/kirby/Documents/cortyze/tribev2")


def _url_hash(url: str) -> str:
    """Short content-derived suffix so different URLs cache + write separately."""
    return hashlib.sha256(url.encode()).hexdigest()[:8]


def apply_mac_cpu_patches() -> None:
    """Force tribev2's heavy submodels onto CPU + bf16 + low_cpu_mem_usage.

    Inlined from tribev2/run.py:11-44. Must run BEFORE importing TribeModel —
    these monkey-patches mutate neuralset.extractors classes that TribeModel
    instantiates lazily.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from neuralset.extractors.audio import HuggingFaceAudio
    from neuralset.extractors.image import HuggingFaceImage
    from neuralset.extractors.text import HuggingFaceText
    from neuralset.extractors.video import HuggingFaceVideo

    HuggingFaceImage.device = property(lambda self: "cpu")

    def force_text_load(self, **kwargs):
        import transformers
        return (
            transformers.AutoModel.from_pretrained(
                self.model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
            )
            .cpu()
            .eval()
        )

    HuggingFaceText._load_model = force_text_load
    HuggingFaceText.device = property(lambda self: "cpu")

    def force_audio_load(self, model_name):
        import transformers
        return (
            transformers.Wav2Vec2BertModel.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
            )
            .cpu()
            .eval()
        )

    HuggingFaceAudio._get_sound_model = force_audio_load
    HuggingFaceAudio.device = property(lambda self: "cpu")

    def force_video_load(self, **kwargs):
        import transformers
        return (
            transformers.AutoModel.from_pretrained(
                self.model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
            )
            .cpu()
            .eval()
        )

    HuggingFaceVideo._load_model = force_video_load
    HuggingFaceVideo.device = property(lambda self: "cpu")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_tribev2_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(TRIBEV2_REPO), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def get_model_revision() -> str | None:
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info("facebook/tribev2").sha
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-url", default=DEFAULT_VIDEO_URL)
    parser.add_argument(
        "--output-stem",
        default=None,
        help=(
            "Path prefix; _T<n>.npy and _T<n>.meta.json will be appended. "
            "Defaults to tests/fixtures/golden_pred_video_<urlhash> so each "
            "unique --video-url writes to its own file (no overwrites)."
        ),
    )
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing fixture at the resolved output path.",
    )
    args = parser.parse_args()

    url_hash = _url_hash(args.video_url)
    output_stem_value = args.output_stem or f"tests/fixtures/golden_pred_video_{url_hash}"

    if not os.environ.get("HF_TOKEN"):
        print(
            "ERROR: HF_TOKEN env var not set.\n"
            "  1. Create a token at https://huggingface.co/settings/tokens\n"
            "  2. Accept gated terms at https://huggingface.co/facebook/tribev2\n"
            "  3. export HF_TOKEN=hf_...",
            file=sys.stderr,
        )
        return 1

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_stem = Path(output_stem_value)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    print("Applying CPU + bf16 patches...", flush=True)
    apply_mac_cpu_patches()

    from tribev2.demo_utils import TribeModel, download_file

    # Cache filename includes URL hash so different videos don't shadow
    # each other (the original `sample_video.mp4` had a quiet-reuse bug).
    video_path = cache_dir / f"video_{url_hash}.mp4"
    if video_path.exists():
        print(f"Reusing cached video {video_path}", flush=True)
    else:
        print(f"Downloading {args.video_url} -> {video_path}", flush=True)
        download_file(args.video_url, video_path)

    print("Hashing video for sidecar...", flush=True)
    video_sha = sha256_file(video_path)

    print("Loading TribeModel (first run pulls ~10GB to ~/.cache/huggingface/)...", flush=True)
    t0 = time.monotonic()
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=cache_dir,
        device="cpu",
    )

    print("Extracting events (audio + transcription)...", flush=True)
    df = model.get_events_dataframe(video_path=video_path)
    balanced_df = df.groupby("type").head(1)
    events_kept = balanced_df["type"].value_counts().to_dict()
    print(f"  Events kept: {events_kept}", flush=True)

    print(f"Running predictions on {len(balanced_df)} events (slow part)...", flush=True)
    preds, _segments = model.predict(events=balanced_df)
    wall_time = int(time.monotonic() - t0)

    if preds.ndim != 2 or preds.shape[1] != 20484:
        print(f"ERROR: unexpected preds shape {preds.shape}; want (T, 20484)", file=sys.stderr)
        return 1

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
        "video_url": args.video_url,
        "video_sha256": video_sha,
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

    print(f"\nSaved {tuple(preds.shape)} {preds.dtype} -> {npy_path}")
    print(f"Saved sidecar -> {meta_path}")
    print(f"Wall time: {wall_time}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
