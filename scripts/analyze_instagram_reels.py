"""Batch-analyze Instagram Reels: download, run TRIBE v2 locally, score.

For each Reel URL in the input file: download via yt-dlp, optionally
upload the mp4 to R2, run TRIBE v2 on the video + caption, score the
predictions through the standard atlas + goal pipeline, append the
result to a JSONL output file. Failures on individual reels do not
abort the batch; they're logged and the script moves on.

Run from the cortyze_product directory with the tribev2 venv (the
only one with neuralset/transformers/torch installed):

    cd /Users/kirby/Documents/cortyze/cortyze_product
    export HF_TOKEN=hf_...
    /Users/kirby/Documents/cortyze/tribev2/.venv/bin/python \\
        scripts/analyze_instagram_reels.py reels.txt --goal engagement

Input file format (`reels.txt`): one Instagram Reel URL per line.
Lines starting with `#` and blank lines are ignored.

Wall time: ~10-20 minutes per reel on Mac CPU (~5 GB RAM). A 50-reel
batch is overnight. Use `--limit N` while iterating.

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

# When invoked as `python scripts/<file>.py`, Python only adds the
# scripts/ directory to sys.path — not the project root — so the lazy
# `from core...`, `from services...`, `from gpu_worker...` imports later
# would fail. Inserting the project root here keeps the script self-
# contained no matter where it's invoked from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Make tools installed in the running venv (notably `whisperx`, which
# TRIBE shells out to via subprocess.run) findable on PATH. When the
# script is invoked as `/path/to/.venv/bin/python ...` without
# `source activate`, the venv's bin/ is NOT on PATH and subprocess
# bails with FileNotFoundError on tools like whisperx.
#
# NOTE: do NOT .resolve() — that follows the venv's python symlink to
# the underlying homebrew/system python's bin/, which won't have the
# venv-installed tools. Plain .parent gives us the venv's own bin/.
_VENV_BIN = Path(sys.executable).parent
if str(_VENV_BIN) not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = f"{_VENV_BIN}{os.pathsep}{os.environ.get('PATH', '')}"


# ---------------------------------------------------------------------------
# CPU + bf16 patches — must run BEFORE importing TribeModel.
# Inlined from scripts/build_fixture.py::apply_mac_cpu_patches so this
# script stays runnable as a single file.
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
def _read_reel_urls(input_path: Path) -> list[str]:
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
class ReelTooLongError(Exception):
    """Reel duration exceeded the configured cap. Treated as a clean
    skip — no traceback printed, just a one-line message."""


# yt-dlp keys we keep into the JSONL `platform` dict for downstream
# correlation analysis (engagement vs brain activations). Any of these
# can be None depending on the post's privacy / login state.
_PLATFORM_FIELDS = (
    "view_count",
    "like_count",
    "comment_count",
    "repost_count",          # shares (when surfaced)
    "uploader",              # display name
    "uploader_id",
    "uploader_url",
    "channel_follower_count",
    "upload_date",           # YYYYMMDD
    "timestamp",             # unix epoch — renamed to upload_timestamp below
    "id",                    # IG post id (e.g. DQccBjtj64i)
    "width",
    "height",
    "fps",
    "language",
)


def _extract_platform_metadata(info: dict) -> dict:
    """Subset yt-dlp's info dict to the engagement / context fields we
    persist for downstream analysis. Renames `timestamp` -> `upload_timestamp`
    and `id` -> `ig_post_id` for clarity in CSV columns."""
    out = {k: info.get(k) for k in _PLATFORM_FIELDS}
    out["upload_timestamp"] = out.pop("timestamp")
    out["ig_post_id"] = out.pop("id")
    return out


def _download_reel(
    url: str,
    out_dir: Path,
    cookies_browser: str | None,
) -> tuple[Path, str, float | None, dict]:
    """Download a Reel. Returns (local_mp4_path, caption, duration_s, platform_metadata).

    Caption is the post's `description` field (often includes hashtags).
    Returns empty string if missing. Duration is yt-dlp's metadata field
    (None if yt-dlp couldn't determine it — proceed without checking).
    `platform_metadata` is the engagement / uploader / posting-context dict.
    """
    import yt_dlp

    out_template = str(out_dir / f"reel_{uuid4().hex}.%(ext)s")
    opts: dict = {
        "outtmpl": out_template,
        "format": "best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    if cookies_browser:
        # yt-dlp accepts a tuple: (browser_name,) or (browser, profile, ...)
        opts["cookiesfrombrowser"] = (cookies_browser,)

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    local_path = Path(ydl.prepare_filename(info))
    if not local_path.exists():
        # yt-dlp sometimes resolves to a different extension after merging.
        # Pick the most-recent matching file in out_dir as a fallback.
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
    duration = info.get("duration")  # seconds, or None if unknown
    platform_metadata = _extract_platform_metadata(info)
    return local_path, caption, duration, platform_metadata


# ---------------------------------------------------------------------------
# Video compression — shrink before R2 upload. V-JEPA2 downsamples to 256x256
# internally so this doesn't speed up inference, but it cuts upload time and
# R2 storage materially. Filter pattern mirrors gpu_worker/post_assembly.py.
# ---------------------------------------------------------------------------
def _compress_video(input_path: Path, target_height: int) -> Path:
    """Re-encode `input_path` to `target_height`p (preserving aspect).

    Output is written next to the input as `<stem>_h<H>.mp4`. Audio is
    stream-copied (no re-encode). Returns the new path. Raises
    `RuntimeError` if ffmpeg fails.

    `scale=-2:H` keeps aspect ratio and rounds width to even (libx264
    requires even dims). Reels are usually portrait (e.g. 1080x1920);
    after scale this becomes ~202x360 — below V-JEPA's 256x256 internal,
    but acceptable for v1. Bump `--target-height` if quality matters.

    `-preset medium -crf 28`: a saner default than ultrafast. Mac CPU
    encode time on a ~30s reel is single-digit seconds either way, and
    `ultrafast` produces inefficient bitstreams that can grow when the
    source is already H.265 / well-compressed.
    """
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
        # Surface ffmpeg's stderr — usually says exactly what's wrong
        # (codec mismatch, audio stream missing, etc.).
        stderr_tail = exc.stderr.decode(errors="replace")[-1000:] if exc.stderr else ""
        raise RuntimeError(
            f"ffmpeg compression failed (code={exc.returncode}):\n{stderr_tail}"
        ) from exc
    return output_path


# ---------------------------------------------------------------------------
# Event helpers — mirror gpu_worker/inference.py::_next_free_time/_merge_events
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
# R2 video upload — R2Client doesn't expose a video helper today, so we
# reach into its boto3 handle. If this becomes load-bearing, promote it
# to a public R2Client.store_uploaded_video method.
# ---------------------------------------------------------------------------
def _upload_video_to_r2(r2_client, request_id: str, mp4_path: Path) -> str:
    key = f"uploads/reels/{request_id}.mp4"
    r2_client._client.put_object(
        Bucket=r2_client.uploads_bucket,
        Key=key,
        Body=mp4_path.read_bytes(),
        ContentType="video/mp4",
    )
    return f"r2://{r2_client.uploads_bucket}/{key}"


# ---------------------------------------------------------------------------
# Per-reel pipeline
# ---------------------------------------------------------------------------
def _analyze_one(
    reel_url: str,
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

    print(f"  [download] {reel_url}", flush=True)
    mp4_path, caption_raw, duration_s, platform_metadata = _download_reel(
        reel_url, cache_dir, cookies_browser
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

    # Skip long reels before paying compression / whisperx / TRIBE costs.
    # Long captions and 6-min audio tracks blow up text-extractor time
    # quadratically; the max_duration cap is the simplest gate.
    if (
        max_duration_s is not None
        and duration_s is not None
        and duration_s > max_duration_s
    ):
        mp4_path.unlink(missing_ok=True)
        raise ReelTooLongError(
            f"reel duration {duration_s:.1f}s exceeds "
            f"MAX_REEL_DURATION_S={max_duration_s}s"
        )

    try:
        compressed_to_height_actual: int | None = None
        if not no_compress:
            print(f"  [compress] -> {target_height}p ...", flush=True)
            compressed_path = _compress_video(mp4_path, target_height)
            compressed_size_kb = compressed_path.stat().st_size // 1024

            if compressed_size_kb >= original_size_kb:
                # Source was likely already efficiently encoded (e.g. H.265).
                # Re-encoding made it larger — keep the original to honor
                # the spirit of "compression should shrink".
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
                # Original mp4 is no longer needed; the compressed version
                # is what gets uploaded and what TRIBE reads.
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
            # caption_to_word_events emits minimal rows (type/start/
            # duration/text); TRIBE's Pydantic Word schema requires more
            # fields and the text extractor requires non-empty context.
            # Two fixes:
            #   1. Inherit metadata (timeline, subject, language, etc.)
            #      from a Word-type row in events_df. iloc[0] may be an
            #      Audio event with NaN context — explicitly prefer Word.
            #   2. Override context/sentence with the actual caption text
            #      so each caption word's context literally contains it
            #      (TRIBE's text extractor computes contextual embeddings
            #      and rejects rows where the target word isn't in
            #      context). Single full-caption value is fine — that's
            #      semantically what the surrounding text is.
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
                # Per-word context: a sliding window of ±N surrounding
                # caption words. The earlier "set context = full caption"
                # approach was correct for short captions but quadratic-
                # blew up the Llama text extractor on long ones (a 1500-
                # char caption pushed each batch from ~40s to ~35min).
                # Mirror what TRIBE does for audio words.
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

        # Save raw (T, 20484) brain activations locally as float32 .npy.
        # This is the actual TRIBE v2 output — load with `np.load(path)`.
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
            "reel_url": reel_url,
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


def _failure_record(reel_url: str, goal: str, exc: BaseException) -> dict:
    return {
        "request_id": None,
        "reel_url": reel_url,
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
    cookies_browser: str | None = None,
    predictions_dir: Path | str | None = None,
    max_duration_s: float | None = None,
) -> int:
    """Validate env, load TRIBE, process URLs, write JSONL. Returns an
    exit-code-style int: 0 = all ok, 1 = setup failed (no work done),
    2 = partial failure (some reels errored). The JSONL is line-buffered
    so a crash mid-batch keeps prior results.

    Importable from wrapper scripts (e.g. scripts/run_reels_inference.py)
    that want to drive the same pipeline from a hardcoded URL list.
    """
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

    # Default: save raw (T, 20484) predictions next to the JSONL so the
    # actual TRIBE v2 brain activations are on disk, not just in R2.
    predictions_dir_resolved: Path | None
    if predictions_dir is None:
        predictions_dir_resolved = output_path.parent / "predictions"
    elif str(predictions_dir).lower() in ("", "off", "none", "false"):
        predictions_dir_resolved = None
    else:
        predictions_dir_resolved = Path(predictions_dir)

    print(f"Found {len(urls)} reel URL(s) to process.", flush=True)
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
        f"keep_videos={keep_videos}",
        flush=True,
    )

    r2_client = None
    if not no_upload:
        from services.storage.r2 import get_client as get_r2

        # R2 init can raise KeyError when only some of the R2_* env vars
        # are set (e.g., R2_ACCESS_KEY present but R2_BUCKET_UPLOADS not).
        # Treat that as "uploads off" rather than aborting the whole run —
        # the local .npy + JSONL outputs are the primary deliverable.
        try:
            r2_client = get_r2()
        except Exception as exc:
            print(
                f"WARNING: R2 partially configured "
                f"({type(exc).__name__}: {exc}). Continuing without uploads — "
                "predictions save locally only. Fix the missing R2_* env "
                "var in .env to enable uploads, or pass no_upload=True to "
                "silence.",
                flush=True,
            )
            r2_client = None
        else:
            if r2_client is None:
                print(
                    "WARNING: STORAGE_MODE=off — R2 upload disabled despite "
                    "no_upload=False. Set STORAGE_MODE=r2 + R2_* env vars to "
                    "enable, or pass no_upload=True to silence.",
                    flush=True,
                )

    print("\nApplying Mac CPU + bf16 patches...", flush=True)
    _apply_mac_cpu_patches()

    from tribev2.demo_utils import TribeModel  # noqa: E402

    print("Loading TribeModel (first run pulls ~10GB to ~/.cache/huggingface/)...", flush=True)
    t_load = time.monotonic()
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=cache_dir,
        device="cpu",
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
            except ReelTooLongError as exc:
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
        help="Text file with one Instagram Reel URL per line. "
             "# comments and blank lines are ignored.",
    )
    parser.add_argument(
        "--goal",
        default="engagement",
        choices=["conversion", "awareness", "engagement", "brand_recall"],
        help="Goal that drives the overall_score weighting (default: engagement). "
             "All four goal-weighted overalls are emitted regardless.",
    )
    parser.add_argument(
        "--output",
        default="reels_results.jsonl",
        help="JSONL output path. Overwritten on each invocation; flushed "
             "after every reel so a crash mid-batch keeps prior results.",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip R2 upload of video and predictions (offline mode). "
             "Implies --keep-videos unless overridden.",
    )
    parser.add_argument(
        "--no-caption",
        action="store_true",
        help="Discard the reel caption (video-only ablation). Useful for "
             "comparing language-region scores with and without caption.",
    )
    parser.add_argument(
        "--keep-videos",
        action="store_true",
        help="Keep the downloaded mp4 in --cache-dir after processing. "
             "Default behavior deletes them post-inference.",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Skip the 360p ffmpeg downscale step before R2 upload. "
             "Compression cuts upload size ~3-5x but does NOT speed up "
             "inference (V-JEPA2 downsamples internally regardless).",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=360,
        help="Output height (px) for the compressed video (default: 360). "
             "Width is auto-scaled to preserve aspect ratio. Ignored when "
             "--no-compress is set.",
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
        help="Where TribeModel caches weights and where reel mp4s land.",
    )
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        help='Pass to yt-dlp for IG-restricted reels. E.g., "firefox" or "chrome".',
    )
    parser.add_argument(
        "--predictions-dir",
        default=None,
        help="Directory to write per-reel raw (T, 20484) brain activations "
             "as <request_id>.npy. Default: <output dir>/predictions/. "
             "Pass 'off' to disable local saving (R2 only).",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Skip reels longer than this many seconds. yt-dlp returns "
             "duration in metadata so the check happens immediately after "
             "download — no wasted compression / inference time.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 1

    urls = _read_reel_urls(input_path)
    if args.limit:
        urls = urls[: args.limit]
    if not urls:
        print(
            f"ERROR: no URLs in {input_path} (after stripping comments/blanks)",
            file=sys.stderr,
        )
        return 1

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
        cookies_browser=args.cookies_from_browser,
        predictions_dir=args.predictions_dir,
        max_duration_s=args.max_duration,
    )


if __name__ == "__main__":
    sys.exit(main())
