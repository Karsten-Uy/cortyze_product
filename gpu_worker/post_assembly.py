"""Helpers for assembling post inputs (1-20 images + optional audio +
optional caption) into something tribev2 understands.

tribev2 only ingests video files. To analyze a post we concatenate the N
images into a single short MP4 (each image held for `seconds_per_image`
seconds), optionally mux in audio, and optionally append synthetic Word
events from a caption.

The assembly path is image-count-agnostic: a 1-image post and a 20-image
carousel use the same `assemble_post_video` entry point. ffmpeg's
concat demuxer accepts a single-file manifest cleanly, so the
single-image case is just N=1 of the general path.

Why ffmpeg subprocess instead of moviepy: tribev2 already shells out to
ffmpeg for audio extraction, so we know it's on the path. moviepy adds a
heavy dependency for what's a 2-line ffmpeg invocation.

Why downscale to 360p: see SCALING.md tier 1 — V-JEPA processes at
256x256 internally and 360p is the smallest source resolution that
doesn't visibly degrade. Dropping from 1080p sources cuts encoder time
~40%.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import pandas as pd


# Reading rate for synthetic Word events from a caption. 150 wpm = 0.4 s/word
# is normal silent reading; on the conservative side so events stay within
# the audio duration when both are supplied.
_WORDS_PER_SECOND = 2.5

# Frame rate for the synthesized video. tribev2 / V-JEPA expects a
# reasonable rate; 24 fps matches sintel and most film/web video.
_FPS = 24


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Required for post assembly "
            "(image→video and optional audio mux). Add it to docker/runpod.Dockerfile."
        )


def _ffprobe_duration(path: Path) -> float:
    """Return the duration in seconds of a media file via ffprobe."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def images_to_video(
    image_paths: list[Path],
    seconds_per_image: float,
    out_dir: Path = Path("/tmp"),
    fps: int = _FPS,
) -> Path:
    """Concatenate N still images into a single MP4.

    Each image is held for `seconds_per_image` seconds. All frames are
    scaled+padded to a common 360p canvas so ffmpeg's concat demuxer
    doesn't bail on dimension mismatches across mixed-aspect images
    (square + landscape mixed is normal in real-world carousels).

    Works for any N >= 1 — a single-image post is just a 1-element list.
    The transitions between images (when N >= 2) are hard cuts. V-JEPA
    reads them as scene changes, which is what we want — the model's
    memorability signal benefits from clean visual segmentation.
    """
    _require_ffmpeg()
    if not image_paths:
        raise ValueError("images_to_video requires at least one image_path")

    # Build a concat manifest. Each image referenced with its hold
    # duration; the final image is repeated without a duration line per
    # the ffmpeg concat demuxer spec (works fine when N=1, the
    # repetition just produces a slightly longer single-image clip).
    manifest_path = out_dir / f"{uuid4().hex}_concat.txt"
    with manifest_path.open("w") as f:
        for path in image_paths:
            f.write(f"file '{path.as_posix()}'\n")
            f.write(f"duration {seconds_per_image:.3f}\n")
        f.write(f"file '{image_paths[-1].as_posix()}'\n")

    out_path = out_dir / f"{uuid4().hex}_post.mp4"
    # scale=-2:360 → height 360, width preserves aspect (rounded to even)
    # pad=ceil(iw/2)*2:ceil(ih/2)*2 → pad to even dims for libx264
    # fps filter normalizes frame rate so the concat output is uniform
    vf_chain = (
        f"scale=-2:360,"
        f"pad=ceil(iw/2)*2:ceil(ih/2)*2:color=black,"
        f"fps={fps}"
    )
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(manifest_path),
                "-vf", vf_chain,
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                "-an",
                str(out_path),
            ],
            capture_output=True,
            check=True,
        )
    finally:
        manifest_path.unlink(missing_ok=True)
    return out_path


def mux_audio(
    video_path: Path,
    audio_path: Path,
    out_dir: Path = Path("/tmp"),
) -> Path:
    """Mux an external audio file into an existing video.

    The output is trimmed to whichever stream is shorter (-shortest) so
    the result stays consistent if the caller built the video from
    images at one duration and the audio doesn't quite match.
    """
    _require_ffmpeg()
    out_path = out_dir / f"{uuid4().hex}_muxed.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(out_path),
        ],
        capture_output=True,
        check=True,
    )
    return out_path


def assemble_post_video(
    image_paths: list[Path],
    audio_path: Path | None,
    seconds_per_image: float,
    out_dir: Path = Path("/tmp"),
) -> tuple[Path, float]:
    """Build a single MP4 from N images + optional audio. Returns
    `(path, duration_s)`.

    If `audio_path` is given, it plays continuously across all images;
    the muxed output is trimmed to the shorter stream.

    `duration_s` is the effective duration of the synthesized video,
    used by callers to time-align supplemental caption events past the
    end of any audio-derived events.
    """
    silent_video = images_to_video(
        image_paths, seconds_per_image, out_dir=out_dir
    )
    video_duration = len(image_paths) * seconds_per_image
    if audio_path is None:
        return silent_video, video_duration

    muxed = mux_audio(silent_video, audio_path, out_dir=out_dir)
    silent_video.unlink(missing_ok=True)
    audio_duration = _ffprobe_duration(audio_path)
    return muxed, min(audio_duration, video_duration)


def caption_to_word_events(
    caption: str,
    base_time_s: float = 0.0,
    words_per_second: float = _WORDS_PER_SECOND,
) -> pd.DataFrame:
    """Synthesize a Word-event DataFrame from a caption string.

    Each word becomes a row with (type, start, duration, text) columns,
    matching the canonical tribev2 events schema. `base_time_s` lets
    callers offset the caption events past any audio-derived events
    (e.g. caption read after voiceover).

    Note: this DataFrame contains only the raw event columns. tribev2's
    `predict()` pipeline expects events that have already passed through
    its extractors (text/audio/video embeddings populated). For
    image+caption-only flows these synthesized events join the events
    DataFrame *after* `get_events_dataframe()` has run on the silent
    video, so the extractors get a chance to embed them on the next pass.
    Validation against real GPU runs may show this needs a follow-up to
    invoke text-extractor preparation explicitly.
    """
    words = [w for w in caption.strip().split() if w]
    if not words:
        return pd.DataFrame(columns=["type", "start", "duration", "text"])

    word_duration = 1.0 / words_per_second
    rows = [
        {
            "type": "Word",
            "start": base_time_s + i * word_duration,
            "duration": word_duration,
            "text": word,
        }
        for i, word in enumerate(words)
    ]
    return pd.DataFrame(rows)


__all__ = [
    "assemble_post_video",
    "images_to_video",
    "mux_audio",
    "caption_to_word_events",
]
