"""Register a fixture as a Stage 2 reference ad.

Computes the BrainReport (region scores + per-goal overall scores) from
the (T, 20484) prediction array, then writes a manifest to
data/reference_ads/<name>.json. The Stage 2 suggestion engine queries
these manifests to surface high-scoring examples for whichever region
the user's content scored low in.

Usage:
    uv run python scripts/register_reference_ad.py \\
        tests/fixtures/golden_pred_sintel_T53.npy \\
        --name sintel_trailer \\
        --display-name "Sintel Trailer" \\
        --source-url https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4 \\
        --description "Blender Foundation open-movie trailer, fantasy animation" \\
        --license "CC BY 3.0"
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from core.atlas.mapper import aggregate
from core.scoring.goals import Goal, overall_score
from core.scoring.normalize import normalize


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_path", type=Path)
    parser.add_argument("--name", required=True, help="Slug used as filename and lookup key")
    parser.add_argument("--display-name", help="Human-readable title (default: name)")
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--license", default="")
    args = parser.parse_args()

    if not args.predictions_path.exists():
        print(f"ERROR: {args.predictions_path} not found", file=sys.stderr)
        return 1

    preds = np.load(args.predictions_path)
    if preds.ndim != 2 or preds.shape[1] != 20484:
        print(f"ERROR: unexpected shape {preds.shape}", file=sys.stderr)
        return 1

    region_scores = normalize(aggregate(preds))
    overall_by_goal = {g.value: overall_score(region_scores, g) for g in Goal}

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "data" / "reference_ads"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        rel_path = args.predictions_path.resolve().relative_to(repo_root)
    except ValueError:
        rel_path = args.predictions_path.resolve()

    manifest = {
        "name": args.name,
        "display_name": args.display_name or args.name,
        "source_url": args.source_url,
        "description": args.description,
        "license": args.license,
        "predictions_path": str(rel_path),
        "predictions_shape": list(preds.shape),
        "region_scores": region_scores,
        "overall_by_goal": overall_by_goal,
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = out_dir / f"{args.name}.json"
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")

    top_region = max(region_scores, key=region_scores.get)
    top_goal = max(overall_by_goal, key=overall_by_goal.get)
    print(f"Registered {args.name} → {out_path.relative_to(repo_root)}")
    print(f"  Top region: {top_region} ({region_scores[top_region]:.1f})")
    print(f"  Top goal:   {top_goal} ({overall_by_goal[top_goal]:.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
