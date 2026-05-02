"""Flatten run_*/results.jsonl into a wide CSV ready for correlation analysis.

One row per reel. Columns:

  request_id, reel_url, caption (truncated), caption_chars, goal,
  events_count, duration_s, elapsed_s, error,
  region_<8 names>          — the 8 marketing-region brain scores (0–100)
  overall_<4 goals>         — same scores reweighted under each goal
  plat_<14 fields>          — yt-dlp engagement / uploader / posting context
  derived_<5 fields>        — engagement_rate, age_days, likes_per_view, etc.
  pred_path                 — where the (T, 20484) .npy lives on disk

Drop the resulting CSV into pandas / Excel / sheets and run pairwise
correlations:

    df = pd.read_csv("results/run_*/results.csv")
    df[[c for c in df.columns if c.startswith("region_")] +
       ["derived_engagement_rate"]].corr()

Usage:

    python scripts/reels_to_csv.py results/run_20260501_062734/results.jsonl
    python scripts/reels_to_csv.py results/run_20260501_062734/results.jsonl -o my.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# Stable column ordering — keep this in sync with core/atlas/regions.py
# and core/scoring/goals.py so the CSV is reproducible across runs.
REGION_KEYS = (
    "visual_cortex",
    "fusiform_face",
    "amygdala",
    "prefrontal",
    "temporal_language",
    "hippocampus",
    "motor",
    "reward",
)

GOAL_KEYS = ("conversion", "awareness", "engagement", "brand_recall")

PLATFORM_KEYS = (
    "view_count",
    "like_count",
    "comment_count",
    "repost_count",
    "uploader",
    "uploader_id",
    "uploader_url",
    "channel_follower_count",
    "upload_date",
    "upload_timestamp",
    "ig_post_id",
    "width",
    "height",
    "fps",
    "language",
)


def _safe_div(num, denom):
    if num is None or denom is None or denom == 0:
        return None
    return num / denom


def _flatten(row: dict, now_ts: float) -> dict:
    flat: dict = {}
    flat["request_id"] = row.get("request_id")
    flat["reel_url"] = row.get("reel_url")
    caption = row.get("caption") or ""
    flat["caption_chars"] = len(caption)
    # Inline a truncated caption for at-a-glance scanning. Strip newlines
    # so the CSV stays one row per reel.
    flat["caption"] = caption.replace("\n", " ").replace("\r", " ").strip()[:500]
    flat["goal"] = row.get("goal")
    flat["overall_score"] = row.get("overall_score")
    flat["events_count"] = row.get("events_count")
    flat["caption_word_events_injected"] = row.get("caption_word_events_injected")
    flat["duration_s"] = row.get("duration_s")
    flat["elapsed_s"] = row.get("elapsed_s")
    flat["error"] = row.get("error")

    region_scores = row.get("region_scores") or {}
    for k in REGION_KEYS:
        flat[f"region_{k}"] = region_scores.get(k)

    obg = row.get("overall_by_goal") or {}
    for k in GOAL_KEYS:
        flat[f"overall_{k}"] = obg.get(k)

    platform = row.get("platform") or {}
    for k in PLATFORM_KEYS:
        flat[f"plat_{k}"] = platform.get(k)

    likes = platform.get("like_count")
    comments = platform.get("comment_count")
    shares = platform.get("repost_count")
    views = platform.get("view_count")
    followers = platform.get("channel_follower_count")
    ts = platform.get("upload_timestamp")

    engagement_parts = [x for x in (likes, comments, shares) if x is not None]
    total_engagement = sum(engagement_parts) if engagement_parts else None

    flat["derived_total_engagement"] = total_engagement
    flat["derived_engagement_rate"] = _safe_div(total_engagement, followers)
    flat["derived_likes_per_view"] = _safe_div(likes, views)
    flat["derived_comments_per_view"] = _safe_div(comments, views)
    flat["derived_engagement_per_view"] = _safe_div(total_engagement, views)
    flat["derived_age_days"] = (
        round((now_ts - ts) / 86400, 2) if ts is not None else None
    )

    flat["pred_path"] = row.get("local_predictions_path")
    return flat


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        help="Path to results.jsonl from a run (e.g. results/run_<ts>/results.jsonl)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="CSV output path. Default: alongside input as results.csv",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found", file=sys.stderr)
        return 1

    output_path = (
        Path(args.output) if args.output else input_path.with_suffix(".csv")
    )

    rows: list[dict] = []
    now_ts = time.time()
    with input_path.open() as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(_flatten(json.loads(line), now_ts))
            except json.JSONDecodeError as exc:
                print(
                    f"WARNING: line {i} is not valid JSON: {exc}",
                    file=sys.stderr,
                )

    if not rows:
        print(f"ERROR: no rows in {input_path}", file=sys.stderr)
        return 1

    # Take fieldnames from the first row — order is fixed by _flatten().
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Wrote {len(rows)} rows × {len(fieldnames)} cols -> {output_path}",
        flush=True,
    )

    n_with_engagement = sum(
        1 for r in rows
        if r.get("plat_like_count") is not None
        or r.get("plat_view_count") is not None
    )
    n_errored = sum(1 for r in rows if r.get("error"))
    print(
        f"  {n_with_engagement}/{len(rows)} rows have at least one engagement metric.",
        flush=True,
    )
    if n_errored:
        print(
            f"  {n_errored} rows are error/skip records (no brain data).",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
