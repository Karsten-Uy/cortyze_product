"""Run TRIBE v2 inference on URLs read from a CSV. YouTube + Instagram both work.

Edit CSV_PATH (and optionally URL_COLUMN) below, then run me. The CSV
just needs a column of video URLs — yt-dlp handles youtube.com,
instagram.com/reel/, instagram.com/p/, vm.tiktok.com, etc. The two
sample CSVs under cortyze_product/data/ both work out of the box:

  - data/cortyze-superbowl-test-ads.csv  (column: youtube_url)
  - data/cortyze-test-urls-100.csv       (column: reel_url)

Run from the cortyze_product directory with the tribev2 venv. On a
RunPod box, point CORTYZE_CACHE_DIR at a writable workspace path so
the ~10GB of HF weights survive between invocations:

    cd /workspace/cortyze/cortyze_product
    export HF_TOKEN=hf_...
    export CORTYZE_CACHE_DIR=/workspace/cache
    /workspace/cortyze/tribev2/.venv/bin/python \\
        scripts/validation/run_csv_inference.py

Results are written to a timestamped folder under RESULTS_DIR. The
JSONL inside it is line-buffered — a crash mid-batch keeps prior
results. CUDA is auto-detected; on a RunPod GPU box the Mac CPU
patches are skipped automatically.
"""
from __future__ import annotations

import csv
import os

# ---------------------------------------------------------------------------
# EDIT THIS BLOCK
# ---------------------------------------------------------------------------

# Path to the input CSV. Relative paths resolve against the cwd you
# launch the script from (typically cortyze_product/).
CSV_PATH = "data/cortyze-superbowl-test-ads.csv"

# Column name that holds the URL. Set to None to auto-detect — the
# script will pick the first column matching one of:
#   url, reel_url, youtube_url, video_url, link, tiktok_url, post_url
URL_COLUMN: str | None = None

# Process only the first N rows of the CSV. None = all rows.
LIMIT: int | None = None

# Goal that drives the overall_score weighting. Region scores are emitted
# regardless; this only changes the headline number.
# One of: "conversion", "awareness", "engagement", "brand_recall".
GOAL = "engagement"

# Where per-run output folders are created. Each run gets a timestamped
# subdirectory holding the JSONL of results.
RESULTS_DIR = "results_csv"

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

# Where TribeModel caches weights and where mp4s land during a run.
# On RunPod, export CORTYZE_CACHE_DIR=/workspace/cache before running
# so weights survive between pod restarts.
CACHE_DIR = os.environ.get(
    "CORTYZE_CACHE_DIR",
    "/Users/kirby/Documents/cortyze/tribev2/cache",
)

# Pass to yt-dlp for restricted videos (age-gated, login-walled, etc.).
# Set to "firefox", "chrome", "safari", etc. None = no cookies.
# IG and TikTok hit this more often than YT.
COOKIES_FROM_BROWSER: str | None = None

# Skip videos longer than this many seconds. None = no limit.
MAX_DURATION_S: float | None = 90

# ---------------------------------------------------------------------------
# Don't edit below — orchestration only.
# ---------------------------------------------------------------------------

import sys
import time
from pathlib import Path

# Make sibling script importable when run as a file.
sys.path.insert(0, str(Path(__file__).parent))

from analyze_instagram_reels import run_batch  # noqa: E402

_URL_COLUMN_CANDIDATES = (
    "url",
    "reel_url",
    "youtube_url",
    "video_url",
    "link",
    "tiktok_url",
    "post_url",
)


def _detect_url_column(fieldnames: list[str]) -> str:
    lookup = {name.lower(): name for name in fieldnames}
    for candidate in _URL_COLUMN_CANDIDATES:
        if candidate in lookup:
            return lookup[candidate]
    raise ValueError(
        "Could not auto-detect URL column in CSV. Looked for one of "
        f"{list(_URL_COLUMN_CANDIDATES)}; CSV has {fieldnames}. "
        "Set URL_COLUMN explicitly at the top of this script."
    )


def _read_urls(csv_path: Path, url_column: str | None) -> tuple[list[str], str]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV {csv_path} has no header row.")
        column = url_column or _detect_url_column(list(reader.fieldnames))
        if column not in reader.fieldnames:
            raise ValueError(
                f"URL_COLUMN={column!r} not found in CSV. Available: "
                f"{list(reader.fieldnames)}"
            )
        urls: list[str] = []
        for row in reader:
            value = (row.get(column) or "").strip()
            if value and not value.startswith("#"):
                urls.append(value)
    return urls, column


def main() -> int:
    csv_path = Path(CSV_PATH)
    if not csv_path.exists():
        print(
            f"ERROR: CSV not found: {csv_path.resolve()}\n"
            f"  cwd={Path.cwd()}",
            file=sys.stderr,
        )
        return 1

    try:
        urls, column = _read_urls(csv_path, URL_COLUMN)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if LIMIT is not None:
        urls = urls[:LIMIT]

    if not urls:
        print(
            f"ERROR: no URLs found in column {column!r} of {csv_path}.",
            file=sys.stderr,
        )
        return 1

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(RESULTS_DIR) / f"run_{timestamp}_{csv_path.stem}"
    output_path = run_dir / "results.jsonl"
    print(f"CSV: {csv_path.resolve()}", flush=True)
    print(f"URL column: {column}  (rows: {len(urls)})", flush=True)
    print(f"Run folder: {run_dir.resolve()}", flush=True)

    return run_batch(
        urls,
        output_path=output_path,
        goal=GOAL,
        cache_dir=CACHE_DIR,
        no_upload=NO_UPLOAD,
        no_caption=NO_CAPTION,
        no_compress=NO_COMPRESS,
        target_height=TARGET_HEIGHT,
        keep_videos=KEEP_VIDEOS,
        cookies_browser=COOKIES_FROM_BROWSER,
        max_duration_s=MAX_DURATION_S,
    )


if __name__ == "__main__":
    sys.exit(main())
