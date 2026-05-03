"""Batch-analyze TikTok videos: download, run TRIBE v2 locally, score.

For each TikTok URL in the input file: download via yt-dlp, optionally
upload the mp4 to R2, run TRIBE v2 on the video + caption, score the
predictions through the standard atlas + goal pipeline, append the
result to a JSONL output file. Failures on individual videos do not
abort the batch; they're logged and the script moves on.

Run from the cortyze_product directory with the tribev2 venv:

    cd /Users/kirby/Documents/cortyze/cortyze_product
    export HF_TOKEN=hf_...
    /Users/kirby/Documents/cortyze/tribev2/.venv/bin/python \\
        scripts/analyze_tiktok_videos.py videos.txt --goal engagement

Input file format (`videos.txt`): one TikTok URL per line.
Lines starting with `#` and blank lines are ignored.

TikTok URL formats accepted:
  https://www.tiktok.com/@username/video/1234567890123456789
  https://vm.tiktok.com/ZMxxxxxxxx/          (short links auto-resolved)

NOTE: TikTok frequently returns 403 for unauthenticated yt-dlp requests.
Pass --cookies-from-browser chrome (or firefox) if you hit auth errors.

Wall time: ~10-20 minutes per video on Mac CPU (~5 GB RAM).

Required venv deps (one-time, into the tribev2 venv):
    /Users/kirby/Documents/cortyze/tribev2/.venv/bin/pip install yt-dlp boto3
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from uuid import uuid4

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_VENV_BIN = Path(sys.executable).parent
if str(_VENV_BIN) not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = f"{_VENV_BIN}{os.pathsep}{os.environ.get('PATH', '')}"


# ---------------------------------------------------------------------------
# CPU + bf16 patches — must run BEFORE importing TribeModel.
# ---------------------------------------------------------------------------
def _apply_mac_cpu_patches() -> None:
    import torch

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


# ---------------------------------------------------------------------------
# Input file parsing
# ---------------------------------------------------------------------------
def _read_video_urls(input_path: Path) -> list[str]:
    urls: list[str] = []
    for raw in input_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


# ---------------------------------------------------------------------------
# yt-dlp wrapper
# ---------------------------------------------------------------------------
class VideoTooLongError(Exception):
    """Video duration exceeded the configured cap. Treated as a clean skip."""


# TikTok-specific yt-dlp metadata fields.
# TikTok surfaces more engagement signals than IG (collect_count / share_count
# are usually populated; bookmark_count is the "saves" metric).
_PLATFORM_FIELDS = (
    # --- engagement ---
    "view_count",
    "like_count",
    "comment_count",
    "repost_count",          # shares / reposts
    "collect_count",         # TikTok "saves" / bookmarks
    # --- account ---
    "uploader",              # display name  (e.g. "MrBeast")
    "uploader_id",           # @handle       (e.g. "@mrbeast")
    "uploader_url",
    "channel_follower_count",
    # --- post context ---
    "upload_date",           # YYYYMMDD
    "timestamp",             # unix epoch — renamed to upload_timestamp below
    "id",                    # TikTok video id (numeric string)
    "tags",                  # list of hashtag strings (without #)
    # --- sound / music ---
    "track",                 # TikTok sound name / song title
    "artist",                # artist / creator of the sound
    "album",                 # album (often None for TikTok originals)
    "music_url",             # https://www.tiktok.com/music/<slug>-<id>
    # --- technical ---
    "width",
    "height",
    "fps",
    "language",
    "duration",              # yt-dlp also puts this here; we extract separately
)


def _parse_music_id(music_url: str | None) -> str | None:
    """Extract the numeric TikTok sound ID from a music_url.

    TikTok music URLs look like:
      https://www.tiktok.com/music/original-sound-7312345678901234567
    The trailing numeric segment after the last hyphen is the canonical
    sound ID that maps to a unique audio asset in TikTok's library.
    Returns None if music_url is absent or doesn't match the pattern.
    """
    if not music_url:
        return None
    # The ID is the last hyphen-separated token in the path.
    slug = music_url.rstrip("/").rsplit("/", 1)[-1]   # e.g. "original-sound-7312..."
    parts = slug.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[1]
    return None


def _parse_hashtags(info: dict) -> list[str]:
    """Return a clean list of hashtags for this post.

    yt-dlp populates `tags` as a list of strings (without the #).
    As a fallback we also parse #words from the description so nothing
    is lost if yt-dlp's tag extraction regresses.
    """
    import re

    tags: list[str] = list(info.get("tags") or [])
    if not tags:
        desc = info.get("description") or ""
        tags = re.findall(r"#(\w+)", desc)
    # Deduplicate while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for t in tags:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            deduped.append(t)
    return deduped


def _extract_platform_metadata(info: dict) -> dict:
    out = {k: info.get(k) for k in _PLATFORM_FIELDS}
    # Rename for clarity in downstream CSV columns.
    out["upload_timestamp"] = out.pop("timestamp")
    out["tiktok_video_id"] = out.pop("id")
    out.pop("duration", None)   # already captured separately as duration_s

    # Enrich with derived fields.
    out["hashtags"] = _parse_hashtags(info)
    out["hashtag_count"] = len(out["hashtags"])
    out["music_id"] = _parse_music_id(out.get("music_url"))

    return out


def _download_video(
    url: str,
    out_dir: Path,
    cookies_browser: str | None,
) -> tuple[Path, str, float | None, dict]:
    """Download a TikTok video. Returns (local_mp4_path, caption, duration_s, platform_metadata).

    TikTok's `description` field is the caption / text overlay (often
    includes hashtags). Returns empty string if missing.

    yt-dlp IG notes don't apply here, but TikTok-specific quirks:
      - Short links (vm.tiktok.com) are resolved automatically.
      - Some region-locked or age-gated videos require cookies.
      - yt-dlp may choose a watermarked format; format string below
        prefers the no-watermark stream when available.
    """
    import yt_dlp

    out_template = str(out_dir / f"tiktok_{uuid4().hex}.%(ext)s")
    opts: dict = {
        "outtmpl": out_template,
        # Prefer no-watermark H.264 stream; fall back to best available.
        "format": "bestvideo[ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        # TikTok sometimes serves a slideshow or photo post — skip those.
        "match_filter": yt_dlp.utils.match_filter_func("!is_live"),
    }
    if cookies_browser:
        opts["cookiesfrombrowser"] = (cookies_browser,)

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    local_path = Path(ydl.prepare_filename(info))
    if not local_path.exists():
        candidates = sorted(
            out_dir.glob(local_path.stem + ".*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise RuntimeError(
                f"yt-dlp reported success but no file at {local_path} or sibling."
            )
        local_path = candidates[0]

    caption = (info.get("description") or "").strip()
    duration = info.get("duration")
    platform_metadata = _extract_platform_metadata(info)
    return local_path, caption, duration, platform_metadata


# ---------------------------------------------------------------------------
# Video compression
# ---------------------------------------------------------------------------
def _compress_video(input_path: Path, target_height: int) -> Path:
    output_path = input_path.parent / f"{input_path.stem}_h{target_height}.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(input_path),
                "-vf", f"scale=-2:{target_height}",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "28",
                "-c:a", "copy",
                "-movflags", "+faststart",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr_tail = exc.stderr.decode(errors="replace")[-1000:] if exc.stderr else ""
        raise RuntimeError(
            f"ffmpeg compression failed (code={exc.returncode}):\n{stderr_tail}"
        ) from exc
    return output_path


# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------
def _next_free_time(events_df, fallback: float) -> float:
    if events_df is None or len(events_df) == 0:
        return float(fallback)
    if "start" not in events_df.columns:
        return float(fallback)
    ends = events_df["start"].astype(float)
    if "duration" in events_df.columns:
        ends = ends + events_df["duration"].astype(float)
    return float(max(ends.max(), fallback))


def _merge_events(base_df, extra_df):
    import pandas as pd

    if base_df is None or len(base_df) == 0:
        return extra_df
    if extra_df is None or len(extra_df) == 0:
        return base_df
    merged = pd.concat([base_df, extra_df], ignore_index=True, sort=False)
    if "start" in merged.columns:
        merged = merged.sort_values("start", kind="mergesort").reset_index(drop=True)
    return merged


# ---------------------------------------------------------------------------
# R2 upload
# ---------------------------------------------------------------------------
def _upload_video_to_r2(r2_client, request_id: str, mp4_path: Path) -> str:
    key = f"uploads/tiktok/{request_id}.mp4"   # <-- tiktok/ prefix vs reels/
    r2_client._client.put_object(
        Bucket=r2_client.uploads_bucket,
        Key=key,
        Body=mp4_path.read_bytes(),
        ContentType="video/mp4",
    )
    return f"r2://{r2_client.uploads_bucket}/{key}"


# ---------------------------------------------------------------------------
# Per-video pipeline
# ---------------------------------------------------------------------------
def _analyze_one(
    video_url: str,
    *,
    model,
    r2_client,
    goal: str,
    cache_dir: Path,
    no_upload: bool,
    no_caption: bool,
    keep_videos: bool,
    no_compress: bool,
    target_height: int,
    cookies_browser: str | None,
    predictions_dir: Path | None,
    max_duration_s: float | None,
) -> dict:
    import numpy as np

    from core.atlas.mapper import aggregate
    from core.scoring.goals import Goal, overall_score
    from core.scoring.normalize import normalize
    from gpu_worker.post_assembly import caption_to_word_events

    request_id = str(uuid4())
    t0 = time.monotonic()

    print(f"  [download] {video_url}", flush=True)
    mp4_path, caption_raw, duration_s, platform_metadata = _download_video(
        video_url, cache_dir, cookies_browser
    )
    caption = "" if no_caption else caption_raw
    original_size_kb = mp4_path.stat().st_size // 1024
    duration_str = f"{duration_s:.1f}s" if duration_s is not None else "unknown"
    print(
        f"  [downloaded] {mp4_path.name} "
        f"({original_size_kb} KB), caption={len(caption)} chars, "
        f"duration={duration_str}",
        flush=True,
    )

    if (
        max_duration_s is not None
        and duration_s is not None
        and duration_s > max_duration_s
    ):
        mp4_path.unlink(missing_ok=True)
        raise VideoTooLongError(
            f"video duration {duration_s:.1f}s exceeds "
            f"MAX_VIDEO_DURATION_S={max_duration_s}s"
        )

    try:
        compressed_to_height_actual: int | None = None
        if not no_compress:
            print(f"  [compress] -> {target_height}p ...", flush=True)
            compressed_path = _compress_video(mp4_path, target_height)
            compressed_size_kb = compressed_path.stat().st_size // 1024

            if compressed_size_kb >= original_size_kb:
                ratio = (compressed_size_kb / original_size_kb) if original_size_kb else 1.0
                print(
                    f"  [compress] result was larger ({original_size_kb} -> "
                    f"{compressed_size_kb} KB, {ratio:.0%}); keeping original.",
                    flush=True,
                )
                compressed_path.unlink(missing_ok=True)
                final_size_kb = original_size_kb
            else:
                ratio = (compressed_size_kb / original_size_kb) if original_size_kb else 1.0
                print(
                    f"  [compress] {original_size_kb} KB -> "
                    f"{compressed_size_kb} KB ({ratio:.0%})",
                    flush=True,
                )
                mp4_path.unlink(missing_ok=True)
                mp4_path = compressed_path
                final_size_kb = compressed_size_kb
                compressed_to_height_actual = target_height
        else:
            final_size_kb = original_size_kb

        r2_video_uri: str | None = None
        if not no_upload and r2_client is not None:
            print("  [r2 video] uploading...", flush=True)
            r2_video_uri = _upload_video_to_r2(r2_client, request_id, mp4_path)
            print(f"  [r2 video] {r2_video_uri}", flush=True)

        print("  [events] extracting (audio + transcription)...", flush=True)
        events_df = model.get_events_dataframe(video_path=mp4_path)

        injected_word_count = 0
        if caption.strip():
            base_t = _next_free_time(events_df, fallback=0.0)
            cap_df = caption_to_word_events(caption, base_time_s=base_t)
            if not events_df.empty:
                word_rows = events_df[events_df.get("type") == "Word"] \
                    if "type" in events_df.columns else events_df.iloc[:0]
                template_row = (
                    word_rows.iloc[0] if not word_rows.empty
                    else events_df.iloc[0]
                )
                for col in events_df.columns:
                    if col not in cap_df.columns:
                        cap_df[col] = template_row[col]
                cap_words = cap_df["text"].tolist()
                window = 10
                cap_contexts = [
                    " ".join(cap_words[max(0, i - window):min(len(cap_words), i + window + 1)])
                    for i in range(len(cap_words))
                ]
                if "context" in cap_df.columns:
                    cap_df["context"] = cap_contexts
                if "sentence" in cap_df.columns:
                    cap_df["sentence"] = cap_contexts
            injected_word_count = len(cap_df)
            events_df = _merge_events(events_df, cap_df)
            print(
                f"  [events] +{injected_word_count} caption Word events "
                f"(base_time={base_t:.1f}s)",
                flush=True,
            )

        print(f"  [predict] {len(events_df)} events (slow part)...", flush=True)
        preds, _segments = model.predict(events=events_df, verbose=False)
        if preds.ndim != 2 or preds.shape[1] != 20484:
            raise RuntimeError(
                f"unexpected preds shape {preds.shape}; want (T, 20484)"
            )
        preds = preds.astype(np.float32)

        region_raw = aggregate(preds)
        region_scores = normalize(region_raw)
        goal_enum = Goal(goal)
        overall = overall_score(region_scores, goal_enum)
        overall_by_goal = {
            g.value: round(overall_score(region_scores, g), 2) for g in Goal
        }

        local_predictions_path: str | None = None
        if predictions_dir is not None:
            predictions_dir.mkdir(parents=True, exist_ok=True)
            npy_path = predictions_dir / f"{request_id}.npy"
            np.save(npy_path, preds)
            local_predictions_path = str(npy_path)
            print(
                f"  [predictions] saved {tuple(preds.shape)} {preds.dtype} "
                f"-> {npy_path} ({npy_path.stat().st_size // 1024} KB)",
                flush=True,
            )

        r2_predictions_uri: str | None = None
        if not no_upload and r2_client is not None:
            r2_predictions_uri = r2_client.store_predictions(request_id, preds)
            print(f"  [r2 predictions] {r2_predictions_uri}", flush=True)

        elapsed = round(time.monotonic() - t0, 2)
        return {
            "request_id": request_id,
            "video_url": video_url,
            "caption": caption,
            "caption_word_events_injected": injected_word_count,
            "goal": goal,
            "region_scores": {k: round(v, 2) for k, v in region_scores.items()},
            "overall_score": round(overall, 2),
            "overall_by_goal": overall_by_goal,
            "r2_video_uri": r2_video_uri,
            "r2_predictions_uri": r2_predictions_uri,
            "local_predictions_path": local_predictions_path,
            "predictions_shape": list(preds.shape),
            "events_count": int(len(events_df)),
            "original_size_kb": original_size_kb,
            "final_size_kb": final_size_kb,
            "compressed_to_height": compressed_to_height_actual,
            "duration_s": duration_s,
            "platform": platform_metadata,
            "elapsed_s": elapsed,
            "error": None,
        }
    finally:
        if not keep_videos:
            mp4_path.unlink(missing_ok=True)


def _failure_record(video_url: str, goal: str, exc: BaseException) -> dict:
    return {
        "request_id": None,
        "video_url": video_url,
        "caption": None,
        "caption_word_events_injected": None,
        "goal": goal,
        "region_scores": None,
        "overall_score": None,
        "overall_by_goal": None,
        "r2_video_uri": None,
        "r2_predictions_uri": None,
        "local_predictions_path": None,
        "predictions_shape": None,
        "events_count": None,
        "original_size_kb": None,
        "final_size_kb": None,
        "compressed_to_height": None,
        "duration_s": None,
        "platform": None,
        "elapsed_s": None,
        "error": f"{type(exc).__name__}: {exc}",
    }


def run_batch(
    urls: list[str],
    *,
    output_path: Path,
    goal: str = "engagement",
    cache_dir: Path | str = "/Users/kirby/Documents/cortyze/tribev2/cache",
    no_upload: bool = False,
    no_caption: bool = False,
    no_compress: bool = False,
    target_height: int = 360,
    keep_videos: bool = False,
    cookies_browser: str | None = "chrome",
    predictions_dir: Path | str | None = None,
    max_duration_s: float | None = None,
) -> int:
    """Validate env, load TRIBE, process URLs, write JSONL. Returns an
    exit-code-style int: 0 = all ok, 1 = setup failed, 2 = partial failure."""
    if not os.environ.get("HF_TOKEN"):
        print(
            "ERROR: HF_TOKEN env var not set.\n"
            "  1. Create a token at https://huggingface.co/settings/tokens\n"
            "  2. Accept gated terms at https://huggingface.co/facebook/tribev2\n"
            "  3. export HF_TOKEN=hf_...",
            file=sys.stderr,
        )
        return 1

    if not no_compress and shutil.which("ffmpeg") is None:
        print(
            "ERROR: ffmpeg not on PATH but compression is enabled.\n"
            "  Install with `brew install ffmpeg`, or pass no_compress=True.",
            file=sys.stderr,
        )
        return 1

    if not urls:
        print("ERROR: empty URL list passed to run_batch().", file=sys.stderr)
        return 1

    keep_videos = keep_videos or no_upload
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions_dir_resolved: Path | None
    if predictions_dir is None:
        predictions_dir_resolved = output_path.parent / "predictions"
    elif str(predictions_dir).lower() in ("", "off", "none", "false"):
        predictions_dir_resolved = None
    else:
        predictions_dir_resolved = Path(predictions_dir)

    print(f"Found {len(urls)} TikTok URL(s) to process.", flush=True)
    print(f"Cache dir:        {cache_dir}", flush=True)
    print(f"Output JSONL:     {output_path}", flush=True)
    print(
        f"Predictions .npy: "
        f"{predictions_dir_resolved if predictions_dir_resolved else '(disabled)'}",
        flush=True,
    )
    max_dur_str = (
        "no limit" if max_duration_s is None else f"{max_duration_s:.0f}s"
    )
    print(
        f"Mode: goal={goal} upload={'off' if no_upload else 'on'} "
        f"caption={'off' if no_caption else 'on'} "
        f"compress={'off' if no_compress else f'{target_height}p'} "
        f"max_duration={max_dur_str} "
        f"keep_videos={keep_videos} "
        f"cookies={cookies_browser or 'none'}",
        flush=True,
    )

    r2_client = None
    if not no_upload:
        from services.storage.r2 import get_client as get_r2

        try:
            r2_client = get_r2()
        except Exception as exc:
            print(
                f"WARNING: R2 partially configured "
                f"({type(exc).__name__}: {exc}). Continuing without uploads.",
                flush=True,
            )
            r2_client = None
        else:
            if r2_client is None:
                print(
                    "WARNING: STORAGE_MODE=off — R2 upload disabled.",
                    flush=True,
                )

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print("\nApplying Mac CPU + bf16 patches...", flush=True)
        _apply_mac_cpu_patches()
    else:
        print(
            f"\nCUDA detected ({torch.cuda.get_device_name(0)}); "
            "skipping Mac CPU patches.",
            flush=True,
        )

    from tribev2.demo_utils import TribeModel

    hf_cache = os.environ.get("HF_HOME", "~/.cache/huggingface/")
    print(
        f"Loading TribeModel onto {device} "
        f"(first run pulls ~10GB to {hf_cache})...",
        flush=True,
    )
    t_load = time.monotonic()
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=cache_dir,
        device=device,
    )
    print(f"Model ready ({int(time.monotonic() - t_load)}s).\n", flush=True)

    n_ok = 0
    n_skipped = 0
    n_err = 0
    with output_path.open("w", buffering=1) as out_f:
        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}] {url}", flush=True)
            try:
                result = _analyze_one(
                    url,
                    model=model,
                    r2_client=r2_client,
                    goal=goal,
                    cache_dir=cache_dir,
                    no_upload=no_upload,
                    no_caption=no_caption,
                    keep_videos=keep_videos,
                    no_compress=no_compress,
                    target_height=target_height,
                    cookies_browser=cookies_browser,
                    predictions_dir=predictions_dir_resolved,
                    max_duration_s=max_duration_s,
                )
                n_ok += 1
                print(
                    f"  [done] overall={result['overall_score']} "
                    f"elapsed={result['elapsed_s']}s",
                    flush=True,
                )
            except VideoTooLongError as exc:
                n_skipped += 1
                print(f"  [skipped] {exc}", flush=True)
                result = _failure_record(url, goal, exc)
            except Exception as exc:
                n_err += 1
                print(f"  [ERROR] {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc(file=sys.stderr)
                result = _failure_record(url, goal, exc)
            out_f.write(json.dumps(result) + "\n")

    print(
        f"\nDone. {n_ok} ok, {n_skipped} skipped, {n_err} failed. "
        f"Results -> {output_path}",
        flush=True,
    )
    return 0 if n_err == 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        help="Text file with one TikTok URL per line. # comments and blank lines are ignored.",
    )
    parser.add_argument(
        "--goal",
        default="engagement",
        choices=["conversion", "awareness", "engagement", "brand_recall"],
        help="Goal that drives the overall_score weighting (default: engagement).",
    )
    parser.add_argument(
        "--output",
        default="tiktok_results.jsonl",
        help="JSONL output path. Flushed after every video.",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip R2 upload (offline mode). Implies --keep-videos.",
    )
    parser.add_argument(
        "--no-caption",
        action="store_true",
        help="Discard the video caption (video-only ablation).",
    )
    parser.add_argument(
        "--keep-videos",
        action="store_true",
        help="Keep downloaded mp4 in --cache-dir after processing.",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Skip 360p ffmpeg downscale before R2 upload.",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=360,
        help="Output height (px) for compressed video (default: 360).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N URLs from the input file.",
    )
    parser.add_argument(
        "--cache-dir",
        default="/Users/kirby/Documents/cortyze/tribev2/cache",
        help="Where TribeModel caches weights and where mp4s land.",
    )
    parser.add_argument(
        "--cookies-from-browser",
        default="chrome",
        help='Browser to pull TikTok auth cookies from. E.g. "chrome" or "firefox". '
             'Default: chrome. Pass "none" to disable.',
    )
    parser.add_argument(
        "--predictions-dir",
        default=None,
        help="Directory to write per-video raw (T, 20484) brain activations "
             "as <request_id>.npy. Default: <output dir>/predictions/. "
             "Pass 'off' to disable.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Skip videos longer than this many seconds.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 1

    urls = _read_video_urls(input_path)
    if args.limit:
        urls = urls[: args.limit]
    if not urls:
        print(
            f"ERROR: no URLs in {input_path} (after stripping comments/blanks)",
            file=sys.stderr,
        )
        return 1

    cookies = None if args.cookies_from_browser.lower() == "none" else args.cookies_from_browser

    return run_batch(
        urls,
        output_path=Path(args.output),
        goal=args.goal,
        cache_dir=args.cache_dir,
        no_upload=args.no_upload,
        no_caption=args.no_caption,
        no_compress=args.no_compress,
        target_height=args.target_height,
        keep_videos=args.keep_videos,
        cookies_browser=cookies,
        predictions_dir=args.predictions_dir,
        max_duration_s=args.max_duration,
    )


if __name__ == "__main__":
    sys.exit(main())