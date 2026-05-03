"""Edit the VIDEOS list below, then run me. No CLI args needed.

    cd /Users/kirby/Documents/cortyze/cortyze_product
    export HF_TOKEN=hf_...
    /Users/kirby/Documents/cortyze/tribev2/.venv/bin/python \\
        scripts/run_tiktok_inference.py

Results are written to a timestamped folder under RESULTS_DIR so each run
is preserved. The JSONL inside it is line-buffered — a crash mid-batch
keeps prior videos' results.

For the CLI / file-driven variant, see scripts/analyze_tiktok_videos.py.

TikTok URL formats accepted by yt-dlp:
  https://www.tiktok.com/@username/video/1234567890123456789
  https://vm.tiktok.com/ZMxxxxxxxx/          (short link — auto-resolved)

NOTE: TikTok aggressively rate-limits unauthenticated yt-dlp. Set
COOKIES_FROM_BROWSER = "chrome" (or "firefox") if you start seeing
403 / "Sign in" errors, or export a cookies.txt and point yt-dlp at it.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# EDIT THIS BLOCK
# ---------------------------------------------------------------------------

VIDEOS: list[str] = [
    # Paste TikTok video URLs here, one per line. Example:
    # "https://www.tiktok.com/@username/video/1234567890123456789",
    # "https://vm.tiktok.com/ZMxxxxxxxx/",
]

# Goal that drives the overall_score weighting. Region scores are emitted
# regardless; this only changes the headline number.
# One of: "conversion", "awareness", "engagement", "brand_recall".
GOAL = "engagement"

# Where per-run output folders are created. Each run gets a timestamped
# subdirectory holding the JSONL of results.
RESULTS_DIR = "results_tiktok"

# Compression knobs. Compression cuts upload size ~3-5x but does NOT
# speed up inference (V-JEPA2 downsamples to 256x256 internally).
NO_COMPRESS = False
TARGET_HEIGHT = 360

# Set True to skip R2 upload (offline / no Cloudflare creds in .env).
NO_UPLOAD = False

# Set True to discard the caption (video-only ablation).
NO_CAPTION = False

# Keep the downloaded mp4 in CACHE_DIR after inference. Default deletes
# them post-run. R2 still has them if NO_UPLOAD is False.
KEEP_VIDEOS = False

# Where TribeModel caches weights and where TikTok mp4s land during a run.
CACHE_DIR = os.environ.get(
    "CORTYZE_CACHE_DIR",
    "/Users/kirby/Documents/cortyze/tribev2/cache",
)

# TikTok frequently requires auth cookies for non-public / FYP content.
# Set to "chrome", "firefox", "safari", etc. None = no cookies.
# If you see 403 errors or "This video is unavailable", set this first.
COOKIES_FROM_BROWSER: str | None = "chrome"

# Skip videos longer than this many seconds.
MAX_VIDEO_DURATION_S: float | None = 60

# ---------------------------------------------------------------------------
# Don't edit below — orchestration only.
# ---------------------------------------------------------------------------

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cortyze_product.scripts.validation.analyze_tiktok_videos import run_batch  # noqa: E402


def main() -> int:
    if not VIDEOS:
        print(
            "ERROR: VIDEOS list is empty. Edit "
            f"{Path(__file__).resolve()} and add TikTok URLs.",
            file=sys.stderr,
        )
        return 1

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(RESULTS_DIR) / f"run_{timestamp}"
    output_path = run_dir / "results.jsonl"
    print(f"Run folder: {run_dir.resolve()}", flush=True)

    return run_batch(
        VIDEOS,
        output_path=output_path,
        goal=GOAL,
        cache_dir=CACHE_DIR,
        no_upload=NO_UPLOAD,
        no_caption=NO_CAPTION,
        no_compress=NO_COMPRESS,
        target_height=TARGET_HEIGHT,
        keep_videos=KEEP_VIDEOS,
        cookies_browser=COOKIES_FROM_BROWSER,
        max_duration_s=MAX_VIDEO_DURATION_S,
    )


if __name__ == "__main__":
    sys.exit(main())