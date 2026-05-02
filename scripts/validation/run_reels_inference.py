"""Edit the REELS list below, then run me. No CLI args needed.

    cd /Users/kirby/Documents/cortyze/cortyze_product
    export HF_TOKEN=hf_...
    /Users/kirby/Documents/cortyze/tribev2/.venv/bin/python \\
        scripts/run_reels_inference.py

Results are written to a timestamped folder under RESULTS_DIR so each run
is preserved. The JSONL inside it is line-buffered — a crash mid-batch
keeps prior reels' results.

For the CLI / file-driven variant, see scripts/analyze_instagram_reels.py.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# EDIT THIS BLOCK
# ---------------------------------------------------------------------------

REELS: list[str] = [
    # Paste Instagram Reel URLs here, one per line. Example:
    # "https://www.instagram.com/reel/ABCDEF12345/",
    # "https://www.instagram.com/reel/GHIJKL67890/",

    # Loreal + Cannon
    # "https://www.instagram.com/reel/DXFMZWdCekH/",
    # "https://www.instagram.com/reels/DWE-WonAQ4n/",
    # "https://www.instagram.com/reel/DDKpXbFPpqz/",
    # "https://www.instagram.com/reels/DF-vtiiCqru/",
    # "https://www.instagram.com/reels/DUXK7NJDL-Y/",

    # Pepsi
    "https://www.instagram.com/reel/DWzdk2-DoRY/",
    "https://www.instagram.com/reel/DVJRdhbjnJB/",
    "https://www.instagram.com/reel/DUhKE91jhxE/",
    "https://www.instagram.com/reel/DUd_IkWjufX/",
    "https://www.instagram.com/reel/DTigJ7Bjvy4/",
    "https://www.instagram.com/reel/DSSx3jbjleL/",
    "https://www.instagram.com/reel/DRmkvHZDmfr/",
    "https://www.instagram.com/reel/DRhooZfDsXf/",
    "https://www.instagram.com/reel/DQhX3hcjuUg/",
    "https://www.instagram.com/reel/DQCzFLKDu0Y/",
    "https://www.instagram.com/reel/DPph6vSDt7u/",
    "https://www.instagram.com/reel/DPCUIh7jlYk/",
    "https://www.instagram.com/reel/DOq9x7KjprE/",
    "https://www.instagram.com/reel/DOXFafdDjOm/",
    "https://www.instagram.com/reel/DMVy-9GOL-9/",
    "https://www.instagram.com/reel/DL-MYPEuZjN/",
    "https://www.instagram.com/reel/DLiOUxIuuVJ/",
    "https://www.instagram.com/reel/DLDMPNDJwGP/",

    # Harvard
    "https://www.instagram.com/reel/DXzJvH8kz2U/",
    "https://www.instagram.com/reel/DXhFvAIjhD-/",
    "https://www.instagram.com/reel/DXZX6tjjv7G/",
    "https://www.instagram.com/reel/DWosah5FKzB/",
    "https://www.instagram.com/reel/DWjbg_UEdZ1/",
    "https://www.instagram.com/reel/DWXZbmrjXdO/",
    "https://www.instagram.com/reel/DWRYfHUkb6x/",
    "https://www.instagram.com/reel/DV_Y5qZCARe/",
    "https://www.instagram.com/reel/DVTon0VjlvW/",
    "https://www.instagram.com/reel/DUnzdcKDgrp/",
    "https://www.instagram.com/reel/DSbKZIDETLF/",
    "https://www.instagram.com/reel/DSXrPtPFVug/",
    "https://www.instagram.com/reel/DQHp7WDDpnO/",
    "https://www.instagram.com/reel/DN0_We-wqW_/",
    "https://www.instagram.com/reel/DMvVd00iBGG/",
    "https://www.instagram.com/reel/DKsGdkkiETa/",
    "https://www.instagram.com/reel/DI38P0OIIjI/",
    "https://www.instagram.com/reel/DHG72HjJPji/",
    "https://www.instagram.com/reel/DG3H5jqpsPu/",
    "https://www.instagram.com/reel/DE0QaHRhA59/",
    "https://www.instagram.com/reel/DA3wDv-pHiX/",
    "https://www.instagram.com/reel/C8ezrh_OKJg/",
    "https://www.instagram.com/reel/C5E14cGR3sJ/",
    "https://www.instagram.com/reel/CyyE1MDOzcj/",
    "https://www.instagram.com/reel/CqbkgZhMnWy/",
    "https://www.instagram.com/reel/CkUCC7Cs4kI/",
    "https://www.instagram.com/reel/CgM1855tf2H/",

    # NBA
    "https://www.instagram.com/reel/DXicdtaklS4/",
    "https://www.instagram.com/reel/C4zKaNdRT5Y/",
    "https://www.instagram.com/reel/C1ihm2ir54I/",
    "https://www.instagram.com/reel/DXzwXSIBVkr/",
    "https://www.instagram.com/reel/DXyHz-rS5hg/",
    "https://www.instagram.com/reel/DXx8YZqyG5C/",
    "https://www.instagram.com/reel/DXxrg-Lpmj3/",
    "https://www.instagram.com/reel/DXwrxn3xCJV/",
    "https://www.instagram.com/reel/DXvbvL-yM0W/",
    "https://www.instagram.com/reel/DXvZMkkMKOA/",
    "https://www.instagram.com/reel/DXuKMiBDMSm/",
    "https://www.instagram.com/reel/DXtBegKkoEo/",
    "https://www.instagram.com/reel/DXs7kzukmpj/",
    "https://www.instagram.com/reel/DXsp0DeEkn6/",
    "https://www.instagram.com/reel/DXrguVShbAR/",
    "https://www.instagram.com/reel/DXqQ3gFEiP7/",
    "https://www.instagram.com/reel/DXqEr3mEmRw/",
    "https://www.instagram.com/reel/DXlRXRCkmbX/",
    "https://www.instagram.com/reel/DXkiQA3kvTS/",
    "https://www.instagram.com/reel/DXjIUR7jOxS/",
    "https://www.instagram.com/reel/DXioq-WkvLe/",
    "https://www.instagram.com/reel/DXgMJrJko_K/",
    "https://www.instagram.com/reel/DXfnObjktkF/",
    "https://www.instagram.com/reel/DXeiYf4kmYq/",
    "https://www.instagram.com/reel/DXeoYyWEoav/",
    "https://www.instagram.com/reel/DXa8Zd4ElhM/",
    "https://www.instagram.com/reel/DXZUlsFka_x/",
    "https://www.instagram.com/reel/DXYONN3EltT/",
    "https://www.instagram.com/reel/DXVpujQxhu-/",
    "https://www.instagram.com/reel/DXVCkt1kZ8_/",

    # Marvel
    "https://www.instagram.com/reel/DXwwHJFEt1N/",
    "https://www.instagram.com/reel/DXvGfCnj5xy/",
    "https://www.instagram.com/reel/DXujPUNiCti/",
    "https://www.instagram.com/reel/DXuYctfEStv/",
    "https://www.instagram.com/reel/DXuEmQgCMRk/",
    "https://www.instagram.com/reel/DXh1uLTAoEP/",
    "https://www.instagram.com/reel/DXkxqYuJh68/",
    "https://www.instagram.com/reel/DXcy4iDFC9O/",
    "https://www.instagram.com/reel/DXaOF02jq32/",
    "https://www.instagram.com/reel/DXZ1FWiFMls/",
    "https://www.instagram.com/reel/DW6rooFj1h7/",
    "https://www.instagram.com/reel/DW4LxTPgTQl/",
    "https://www.instagram.com/reel/DWsW3bPkaMy/",
    "https://www.instagram.com/reel/DWrPFA0iGew/",
    "https://www.instagram.com/reel/DWjSvQbDNgt/",
    "https://www.instagram.com/reel/DWZBKU6Fd87/",
    "https://www.instagram.com/reel/DWRrBiMjO9m/",
    "https://www.instagram.com/reel/DWQOtvvibAw/",
    "https://www.instagram.com/reel/DWQLZFWCVVB/",
    "https://www.instagram.com/reel/DWPcctiiafb/",
    "https://www.instagram.com/reel/DWMVRJ0jJ9E/",
    "https://www.instagram.com/reel/DWHGghTgYE9/",
    "https://www.instagram.com/reel/DWExHflAdnZ/",
    "https://www.instagram.com/reel/DV_InRXCQMe/",
    "https://www.instagram.com/reel/DVuXZFmj-CR/",
    "https://www.instagram.com/reel/DVcbyTKE4_R/",
    
    # Dove
    "https://www.instagram.com/reel/DXy5aQ8ChZn/",
    "https://www.instagram.com/reel/DXrhvAQE5Av/",
    "https://www.instagram.com/reel/DXo1tbYjHY9/",
    "https://www.instagram.com/reel/DXes2jBiOZQ/",
    "https://www.instagram.com/reel/DXcW0-BEc2s/",
    "https://www.instagram.com/reel/DXWkqHsD48S/",
    "https://www.instagram.com/reel/DXKErkEj6_X/",
    "https://www.instagram.com/reel/DWjZlFXCQGl/",
    "https://www.instagram.com/reel/DWMG9UCgfqT/",
    "https://www.instagram.com/reel/DWEY-vujjD6/",
    "https://www.instagram.com/reel/DVd3YtDkSYo/",
    "https://www.instagram.com/reel/DVLopy_AtCs/",
    "https://www.instagram.com/reel/DUdm5mhjXzm/",
    "https://www.instagram.com/reel/DTh8vthD4Cq/",
    "https://www.instagram.com/reel/DRSJrnciGGN/",
    "https://www.instagram.com/reel/DP6zKykETc4/",
    "https://www.instagram.com/reel/DQKSI9GEW2D/",
    "https://www.instagram.com/reel/DPeLAsaCpI4/",
    "https://www.instagram.com/reel/DOMBtdXCHK9/",
    "https://www.instagram.com/reel/DNqmALuoyvG/",
    "https://www.instagram.com/reel/DNWpsa1xGdV/",
    "https://www.instagram.com/reel/DNhMIb1xOBw/",

    # Mr beast
    "https://www.instagram.com/reel/DV6JzBFiLcx/",
    "https://www.instagram.com/reel/DUf9drTiPuQ/",
    "https://www.instagram.com/reel/DVTihY-CKsH/",
    "https://www.instagram.com/reel/DXmQSmbCLuB/",
    "https://www.instagram.com/reel/DXmPzsOCC0M/",
    "https://www.instagram.com/reel/DXhIS8-CCqM/",
    "https://www.instagram.com/reel/DXhH_CbCPiD/",
    "https://www.instagram.com/reel/DXULEnNiMnR/",
    "https://www.instagram.com/reel/DXRo0GLCMY-/",
    "https://www.instagram.com/reel/DWrHDFnCMQQ/",
    "https://www.instagram.com/reel/DWrFZ8piMmD/",
    "https://www.instagram.com/reel/DWbnAIxiJmQ/",
    "https://www.instagram.com/reel/DWT3uSHiIS4/",
    "https://www.instagram.com/reel/DV6JLN-CDFB/",
    "https://www.instagram.com/reel/DVOY0-hCO3v/",
    "https://www.instagram.com/reel/DUVu9IJiJrK/",
    "https://www.instagram.com/reel/DTvM4G1CM-x/",
    "https://www.instagram.com/reel/DFLS-Hfo9_h/",
    "https://www.instagram.com/reel/DFH_laQoQ1U/",

    # evbo
    "https://www.instagram.com/reel/DEKI4ROScjH/",
    "https://www.instagram.com/reel/DEKHskUNAjZ/",
    "https://www.instagram.com/reel/C_X8hVQtcTl/",
    "https://www.instagram.com/reel/C_UbjHRSdRd/",
    "https://www.instagram.com/reel/C-kZU0Zyh4i/",
    "https://www.instagram.com/reel/C9gjhNhK_-q/",
    "https://www.instagram.com/reel/C9UzeTANSs9/",
    "https://www.instagram.com/reel/C9LugpGSS0o/",
    "https://www.instagram.com/reel/C9Ag5_KNaHO/",
    "https://www.instagram.com/reel/C82Dt52NvUl/",
    "https://www.instagram.com/reel/C8syRztS-jd/",
    "https://www.instagram.com/reel/C8nqgL_IVpC/",
    "https://www.instagram.com/reel/C8kLg2oIHax/",

    # nmplol

    "https://www.instagram.com/reel/DKcvqyjCvRH/",
    "https://www.instagram.com/reel/C8iERtTNe99/",
    "https://www.instagram.com/reel/DXyl0zbjNwv/",
    "https://www.instagram.com/reel/DXrK7imjrFZ/",
    "https://www.instagram.com/reel/DXw0ZNyDMhO/",
    "https://www.instagram.com/reel/DXnBqyVkSRv/",
    "https://www.instagram.com/reel/DXg2LcqDLq1/",
    "https://www.instagram.com/reel/DXcDB6VFM-l/",
    "https://www.instagram.com/reel/DXX8NvbiYff/",
    "https://www.instagram.com/reel/DXRkhcBlOkM/",
    "https://www.instagram.com/reel/DXPtTN2DTdl/",
    "https://www.instagram.com/reel/DXKu_v-k7wx/",
    "https://www.instagram.com/reel/DXIREXHgW5h/",
    "https://www.instagram.com/reel/DXE0VVsgdc9/",
    "https://www.instagram.com/reel/DWmTyMDiKKS/",
    "https://www.instagram.com/reel/DWYtyMIkua-/",
    "https://www.instagram.com/reel/DWTs62SGP52/",
    "https://www.instagram.com/reel/DV92dr_lNQJ/",
    "https://www.instagram.com/reel/DVq2sLSka-z/",

]

# Goal that drives the overall_score weighting. Region scores are emitted
# regardless; this only changes the headline number.
# One of: "conversion", "awareness", "engagement", "brand_recall".
GOAL = "engagement"

# Where per-run output folders are created. Each run gets a timestamped
# subdirectory holding the JSONL of results.
RESULTS_DIR = "results"

# Compression knobs. Compression cuts upload size ~3-5x but does NOT
# speed up inference (V-JEPA2 downsamples to 256x256 internally).
NO_COMPRESS = False
TARGET_HEIGHT = 360

# Set True to skip R2 upload (offline / no Cloudflare creds in .env).
# When True, raw predictions only live as long as you keep the JSONL.
NO_UPLOAD = False

# Set True to discard the caption (video-only ablation). Useful for
# comparing language-region scores with vs without the caption signal.
NO_CAPTION = False

# Keep the downloaded mp4 in CACHE_DIR after inference. Default deletes
# them post-run. R2 still has them if NO_UPLOAD is False.
KEEP_VIDEOS = False

# Where TribeModel caches weights and where reel mp4s land during a run.
# Override on RunPod (or any non-Mac box) by exporting CORTYZE_CACHE_DIR
# before running, e.g. `export CORTYZE_CACHE_DIR=/workspace/cache`.
CACHE_DIR = os.environ.get(
    "CORTYZE_CACHE_DIR",
    "/Users/kirby/Documents/cortyze/tribev2/cache",
)

# Pass to yt-dlp for IG-restricted reels (age-gated, region-locked, etc.).
# Set to "firefox", "chrome", "safari", etc. None = no cookies.
COOKIES_FROM_BROWSER: str | None = None

# Skip reels longer than this many seconds. Long reels (multi-minute
# audio + long captions) blow up whisperx + text-extractor cost
# super-linearly on Mac CPU. Set to None for no limit.
MAX_REEL_DURATION_S: float | None = 60

# ---------------------------------------------------------------------------
# Don't edit below — orchestration only.
# ---------------------------------------------------------------------------

import sys
import time
from pathlib import Path

# Make sibling script importable when run as a file.
sys.path.insert(0, str(Path(__file__).parent))

from cortyze_product.scripts.validation.analyze_instagram_reels import run_batch  # noqa: E402


def main() -> int:
    if not REELS:
        print(
            "ERROR: REELS list is empty. Edit "
            f"{Path(__file__).resolve()} and add Reel URLs.",
            file=sys.stderr,
        )
        return 1

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(RESULTS_DIR) / f"run_{timestamp}"
    output_path = run_dir / "results.jsonl"
    print(f"Run folder: {run_dir.resolve()}", flush=True)

    return run_batch(
        REELS,
        output_path=output_path,
        goal=GOAL,
        cache_dir=CACHE_DIR,
        no_upload=NO_UPLOAD,
        no_caption=NO_CAPTION,
        no_compress=NO_COMPRESS,
        target_height=TARGET_HEIGHT,
        keep_videos=KEEP_VIDEOS,
        cookies_browser=COOKIES_FROM_BROWSER,
        max_duration_s=MAX_REEL_DURATION_S,
    )


if __name__ == "__main__":
    sys.exit(main())
