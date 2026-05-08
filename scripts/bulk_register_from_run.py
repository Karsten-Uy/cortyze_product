"""Register every successful ad in a results.jsonl as a reference ad.

Joins the per-reel JSONL produced by run_csv_inference.py with the
input CSV (brand / ad_name / context columns) on URL, then shells out
to register_reference_ad.py for each successful row. Failed rows
(error != None) and rows whose .npy is missing are skipped.

    cd cortyze_product
    uv run python scripts/bulk_register_from_run.py \\
        results/superbowl/results_csv/run_20260506_154321_cortyze-superbowl-test-ads/results.jsonl \\
        data/cortyze-superbowl-test-ads.csv

Pass --dry-run to print the commands without executing them.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


def _slugify(text: str) -> str:
    """brand + ad_name → 'brand_ad_name'-style slug."""
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s or "unnamed"


_YOUTUBE_ID_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([A-Za-z0-9_-]{11})"
)


def _derive_thumbnail(url: str) -> str:
    """Best-effort thumbnail URL for the source.

    YouTube provides public stills at img.youtube.com — use the highest
    quality variant; the frontend falls back to the SVG placeholder if
    the image 404s. Instagram has no public thumbnail URL we can derive
    statically (would need yt-dlp to scrape one), so return ''.
    """
    m = _YOUTUBE_ID_RE.search(url)
    if m:
        return f"https://img.youtube.com/vi/{m.group(1)}/maxresdefault.jpg"
    return ""


def _load_csv_index(csv_path: Path, url_col: str) -> dict[str, dict[str, str]]:
    """Map URL → CSV row dict so we can pull brand/ad_name/etc per result."""
    out: dict[str, dict[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if url_col not in (reader.fieldnames or []):
            raise SystemExit(
                f"--url-column {url_col!r} not in CSV. Available: "
                f"{reader.fieldnames}"
            )
        for row in reader:
            url = (row.get(url_col) or "").strip()
            if url:
                out[url] = row
    return out


def _build_args(
    *,
    npy_path: Path,
    result: dict,
    csv_row: dict[str, str] | None,
    license_default: str,
) -> list[str]:
    """Translate one JSONL row + matching CSV row into a register CLI invocation."""
    url = result["reel_url"]

    # Prefer brand+ad_name from CSV when available; fall back to the URL slug.
    brand = (csv_row or {}).get("brand", "").strip()
    ad_name = (csv_row or {}).get("ad_name", "").strip()
    name_slug = _slugify(f"{brand}_{ad_name}") if (brand and ad_name) else _slugify(url.rsplit("/", 1)[-1])

    display_parts = [p for p in (brand, ad_name) if p]
    display_name = " — ".join(display_parts) if display_parts else url

    description = (csv_row or {}).get("context", "")
    tags_field = ",".join(
        filter(
            None,
            [
                (csv_row or {}).get("brand", ""),
                (csv_row or {}).get("campaign", ""),
                (csv_row or {}).get("goal_to_test", ""),
                (csv_row or {}).get("expected_tier", ""),
            ],
        )
    )
    tags = ",".join(_slugify(t) for t in tags_field.split(",") if t.strip())

    thumbnail_url = _derive_thumbnail(url)

    cmd = [
        "uv", "run", "python", "scripts/register_reference_ad.py",
        str(npy_path),
        "--name", name_slug,
        "--display-name", display_name,
        "--content-type", "video",
        "--source-url", url,
        "--description", description,
        "--caption", (result.get("caption") or "").strip()[:500],
        "--tags", tags,
        "--license", license_default,
        "--force",
    ]
    if thumbnail_url:
        cmd.extend(["--thumbnail-url", thumbnail_url])
    return cmd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_jsonl", type=Path)
    p.add_argument("input_csv", type=Path, nargs="?", default=None,
                   help="Input CSV with brand/ad_name/context columns. "
                        "Optional — without it, slugs come from the URL.")
    p.add_argument("--url-column", default="youtube_url",
                   help="CSV column holding the URL (default: youtube_url).")
    p.add_argument("--license", default="Fair use — research / educational",
                   help="License string baked into every manifest.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing.")
    args = p.parse_args()

    if not args.results_jsonl.exists():
        print(f"ERROR: {args.results_jsonl} not found", file=sys.stderr)
        return 1

    csv_index: dict[str, dict[str, str]] = {}
    if args.input_csv:
        if not args.input_csv.exists():
            print(f"ERROR: {args.input_csv} not found", file=sys.stderr)
            return 1
        csv_index = _load_csv_index(args.input_csv, args.url_column)

    # The JSONL stores predictions paths relative to the cwd it was written
    # from (cortyze_product/), so resolve relative to the JSONL's parents.
    repo_root = Path(__file__).resolve().parents[1]

    n_ok = 0
    n_skip = 0
    n_err = 0
    for line_no, line in enumerate(args.results_jsonl.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        result = json.loads(line)
        url = result.get("reel_url")
        if not url:
            n_skip += 1
            continue
        if result.get("error"):
            n_skip += 1
            print(f"  skip [{line_no}] {url} — failed/skipped: {result['error'][:80]}")
            continue
        npy_rel = result.get("local_predictions_path")
        request_id = result.get("request_id")
        if not (npy_rel or request_id):
            n_skip += 1
            print(f"  skip [{line_no}] {url} — no local_predictions_path or request_id")
            continue

        # Try several resolution strategies — the JSONL stores paths relative
        # to the original run cwd, but the run folder may have been moved.
        # The .npy always sits at <jsonl_parent>/predictions/<request_id>.npy
        # since the run script writes both into the same folder.
        jsonl_dir = args.results_jsonl.resolve().parent
        candidates: list[Path] = []
        if npy_rel:
            npy_basename = Path(npy_rel).name
            candidates += [
                Path(npy_rel),                      # absolute or relative to cwd
                repo_root / npy_rel,                # relative to repo root (legacy)
                jsonl_dir / "predictions" / npy_basename,  # next to the JSONL
            ]
        if request_id:
            candidates.append(jsonl_dir / "predictions" / f"{request_id}.npy")

        npy_path = next((c.resolve() for c in candidates if c.exists()), None)
        if npy_path is None:
            n_err += 1
            tried = "\n      ".join(str(c) for c in candidates)
            print(f"  ERROR [{line_no}] {url} — npy not found, tried:\n      {tried}")
            continue

        cmd = _build_args(
            npy_path=npy_path,
            result=result,
            csv_row=csv_index.get(url),
            license_default=args.license,
        )

        if args.dry_run:
            print("DRY:", " ".join(repr(a) if " " in a else a for a in cmd))
            n_ok += 1
            continue

        print(f"  → registering {cmd[cmd.index('--name') + 1]}  ({url})")
        proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
        if proc.returncode != 0:
            n_err += 1
            print(f"    FAILED:\n{proc.stderr[-500:]}")
        else:
            n_ok += 1

    print(f"\nDone. {n_ok} registered, {n_skip} skipped, {n_err} errored.")
    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
