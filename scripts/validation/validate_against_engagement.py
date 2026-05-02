"""Validate the Cortyze scoring algorithm against real engagement metrics.

Tests whether the brain-region scores (and the goal-weighted overalls
that the scoring layer computes from them) actually predict real-world
marketing outcomes — likes, views, comments, etc.

Three validation tests, run in order:

  1. PER-REGION CORRELATIONS
     For each (region, engagement_metric) pair, Spearman rank
     correlation. rho > 0.3 = real signal worth investigating.

  2. GOAL-WEIGHTING ADDS VALUE?
     Does `overall_engagement` correlate with engagement-like metrics
     better than the single best region does? If yes, the weighted sum
     is contributing information; if no, the weights are noise.

  3. GOAL-LABEL ALIGNMENT
     Does `overall_conversion` correlate with conversion-like metrics
     (saves, comments) more than with awareness-like metrics (views)?
     If all four overall_* columns correlate identically with
     everything, the goal labels are cosmetic — the weights aren't
     differentiating goals.

Usage:
    # Use whatever engagement data is already in the results CSV (from yt-dlp):
    python scripts/validate_against_engagement.py results/<run>/results.csv

    # Or merge in manually-collected engagement data:
    python scripts/validate_against_engagement.py results/<run>/results.csv \\
        --engagement-csv my_engagement_data.csv

    # External CSV columns: reel_url,view_count,like_count,comment_count,
    #                       save_count,share_count,follower_count
    # (Any subset of these works; missing metrics are skipped.)

Requires: numpy, scipy. Install in your venv if missing:
    pip install numpy scipy
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REGIONS = (
    "visual_cortex",
    "fusiform_face",
    "amygdala",
    "prefrontal",
    "temporal_language",
    "hippocampus",
    "motor",
    "reward",
)

GOALS = ("conversion", "awareness", "engagement", "brand_recall")

# Mapping of engagement metric name -> the goal it should align with.
# Used in test 3 to check label alignment.
METRIC_GOAL_AFFINITY = {
    "view_count": "awareness",                  # raw reach
    "like_count": "engagement",                 # baseline emotional vote
    "comment_count": "engagement",              # deeper engagement
    "save_count": "brand_recall",               # save = "I want to remember this"
    "share_count": "engagement",                # viral reach via emotion
    "follower_count": None,                     # control variable, not an outcome
}

# Derived metrics computed from the raw fields (normalized for reach/account size).
DERIVED_METRICS = (
    "engagement_per_view",
    "likes_per_view",
    "comments_per_view",
    "engagement_rate",          # /follower_count
    "total_engagement",
)


def fnum(s):
    if s is None or s == "":
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def load_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def maybe_merge_engagement(brain_rows: list[dict], engagement_csv: Path | None) -> list[dict]:
    """If a separate engagement CSV is supplied, join it onto brain rows by reel_url."""
    if engagement_csv is None:
        return brain_rows
    eng_by_url: dict[str, dict] = {}
    with open(engagement_csv) as f:
        for row in csv.DictReader(f):
            url = (row.get("reel_url") or "").strip()
            if url:
                eng_by_url[url] = row

    merged = 0
    for r in brain_rows:
        eng = eng_by_url.get((r.get("reel_url") or "").strip())
        if not eng:
            continue
        merged += 1
        for col, plat_col in [
            ("view_count",     "plat_view_count"),
            ("like_count",     "plat_like_count"),
            ("comment_count",  "plat_comment_count"),
            ("save_count",     "plat_save_count"),
            ("share_count",    "plat_repost_count"),
            ("follower_count", "plat_channel_follower_count"),
        ]:
            if eng.get(col):
                r[plat_col] = eng[col]
    print(f"  Merged engagement data from {engagement_csv}: {merged} rows matched.")
    return brain_rows


def compute_derived(rows: list[dict]) -> None:
    """Recompute derived_* metrics in-place from whatever raw fields are present."""
    for r in rows:
        likes = fnum(r.get("plat_like_count"))
        comments = fnum(r.get("plat_comment_count"))
        shares = fnum(r.get("plat_repost_count")) or fnum(r.get("plat_share_count"))
        views = fnum(r.get("plat_view_count"))
        followers = fnum(r.get("plat_channel_follower_count"))

        eng_parts = [x for x in (likes, comments, shares) if x is not None]
        total = sum(eng_parts) if eng_parts else None

        def safediv(a, b):
            if a is None or b is None or b == 0:
                return None
            return a / b

        r["derived_total_engagement"] = total
        r["derived_engagement_rate"] = safediv(total, followers)
        r["derived_likes_per_view"] = safediv(likes, views)
        r["derived_comments_per_view"] = safediv(comments, views)
        r["derived_engagement_per_view"] = safediv(total, views)


def get_predictor_arrays(rows: list[dict]) -> dict[str, list[float]]:
    """Return dict of predictor name -> list (regions + 4 overalls)."""
    preds = {}
    for region in REGIONS:
        preds[f"region_{region}"] = [fnum(r.get(f"region_{region}")) for r in rows]
    for goal in GOALS:
        preds[f"overall_{goal}"] = [fnum(r.get(f"overall_{goal}")) for r in rows]
    return preds


def get_outcome_arrays(rows: list[dict]) -> dict[str, list[float]]:
    """Return dict of outcome name -> list. Skips outcomes with <3 non-null values."""
    out = {}
    for raw_metric in ("view_count", "like_count", "comment_count",
                       "save_count", "share_count", "repost_count"):
        col = f"plat_{raw_metric}"
        vals = [fnum(r.get(col)) for r in rows]
        if sum(1 for v in vals if v is not None) >= 3:
            out[raw_metric] = vals
    for derived in DERIVED_METRICS:
        col = f"derived_{derived}"
        vals = [fnum(r.get(col)) for r in rows]
        if sum(1 for v in vals if v is not None) >= 3:
            out[derived] = vals
    return out


def spearman_pair(x: list[float], y: list[float]) -> tuple[float, float, int]:
    """Spearman rho + p-value over rows where BOTH x and y are non-None."""
    pairs = [(xv, yv) for xv, yv in zip(x, y) if xv is not None and yv is not None]
    n = len(pairs)
    if n < 5:
        return float("nan"), float("nan"), n
    xs, ys = zip(*pairs)
    rho, p = stats.spearmanr(xs, ys)
    return float(rho), float(p), n


def section(title: str) -> None:
    print(f"\n{'='*78}")
    print(f"  {title}")
    print(f"{'='*78}")


def test_1_per_region(predictors: dict, outcomes: dict) -> None:
    section("TEST 1. PER-REGION SPEARMAN CORRELATIONS  (rho, p-value, n)")
    print("\n  ** What's a meaningful rho? **")
    print("  rho > 0.5  strong signal       (rare)")
    print("  rho > 0.3  real signal         (this is what you want)")
    print("  rho < 0.2  noise / weak signal (probably no real relationship)")
    print("  p < 0.05   statistically significant")
    print()

    for outcome_name, outcome_vals in outcomes.items():
        print(f"\n  vs {outcome_name}:")
        print(f"    {'predictor':<28} {'rho':>7} {'p':>7} {'n':>4}")
        results = []
        for pred_name in REGIONS:
            rho, p, n = spearman_pair(predictors[f"region_{pred_name}"], outcome_vals)
            results.append((pred_name, rho, p, n))
        # Sort by abs(rho) descending so strongest signal is at top
        results.sort(key=lambda x: -abs(x[1]) if x[1] == x[1] else 0)
        for pred_name, rho, p, n in results:
            star = " *" if p < 0.05 and abs(rho) > 0.3 else ""
            rho_str = f"{rho:7.2f}" if rho == rho else "    nan"
            p_str = f"{p:7.3f}" if p == p else "    nan"
            print(f"    region_{pred_name:<22} {rho_str} {p_str} {n:4d}{star}")


def test_2_weighting_value(predictors: dict, outcomes: dict) -> None:
    section("TEST 2. DO THE GOAL-WEIGHTED OVERALLS BEAT BEST SINGLE REGION?")
    print()
    print("  Compares overall_<goal> vs the strongest individual region for each outcome.")
    print("  If overall_<goal> > best region, the weighting adds information.")
    print("  If overall_<goal> ~ best region, the weighting is just amplifying that region.")
    print("  If overall_<goal> < best region, the weighting is HURTING (noise).")

    for outcome_name, outcome_vals in outcomes.items():
        # Best individual region
        best_region, best_rho = None, 0
        for region in REGIONS:
            rho, p, n = spearman_pair(predictors[f"region_{region}"], outcome_vals)
            if rho == rho and abs(rho) > abs(best_rho):
                best_region, best_rho = region, rho

        if best_region is None:
            continue
        print(f"\n  vs {outcome_name}:")
        print(f"    Best single region: region_{best_region}  rho = {best_rho:+.2f}")
        print(f"    Goal-weighted overalls:")
        for goal in GOALS:
            rho, p, n = spearman_pair(predictors[f"overall_{goal}"], outcome_vals)
            if rho != rho:
                continue
            delta = abs(rho) - abs(best_rho)
            verdict = (
                "WORSE  " if delta < -0.05 else
                "tied   " if abs(delta) < 0.05 else
                "BETTER "
            )
            print(f"      overall_{goal:<13}  rho = {rho:+.2f}   ({verdict} vs best region by {delta:+.2f})")


def test_3_goal_alignment(predictors: dict, outcomes: dict) -> None:
    section("TEST 3. ARE THE GOAL LABELS ALIGNED WITH THEIR NAMED OUTCOMES?")
    print()
    print("  For each engagement metric, which goal-weighting correlates STRONGEST?")
    print("  If overall_conversion correlates strongest with save_count -> alignment GOOD")
    print("  If overall_engagement correlates strongest with views     -> alignment OFF")
    print("  If all four overall_* are equal -> labels are cosmetic, weights aren't differentiating.")

    print(f"\n  {'metric':<28} {'best goal':<18} {'rho':>7} {'expected':<14} {'verdict'}")
    print("  " + "-" * 78)
    for outcome_name, outcome_vals in outcomes.items():
        rhos = {}
        for goal in GOALS:
            rho, p, n = spearman_pair(predictors[f"overall_{goal}"], outcome_vals)
            if rho == rho:
                rhos[goal] = rho
        if not rhos:
            continue
        best_goal = max(rhos, key=lambda g: abs(rhos[g]))
        worst_goal = min(rhos, key=lambda g: abs(rhos[g]))
        spread = abs(rhos[best_goal]) - abs(rhos[worst_goal])

        # Cross-check vs metric->goal affinity table
        raw_metric_name = outcome_name.replace("derived_", "").replace("_per_view", "_count").replace("_rate", "_count")
        expected_goal = METRIC_GOAL_AFFINITY.get(raw_metric_name)

        if spread < 0.05:
            verdict = "labels = cosmetic"
        elif expected_goal and best_goal == expected_goal:
            verdict = "ALIGNED"
        elif expected_goal:
            verdict = f"misaligned (expected {expected_goal})"
        else:
            verdict = "no expectation"

        print(f"  {outcome_name:<28} {best_goal:<18} {rhos[best_goal]:+7.2f} "
              f"{(expected_goal or '-'):<14} {verdict}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path",
                        help="Path to results.csv produced by reels_to_csv.py")
    parser.add_argument("--engagement-csv", default=None,
                        help="Optional external CSV with reel_url + engagement columns "
                             "(view_count,like_count,comment_count,save_count,share_count,follower_count)")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        return 1

    rows = load_csv(csv_path)
    rows = [r for r in rows if not (r.get("error") or "").strip()]
    print(f"\nFile: {csv_path}")
    print(f"Successful rows loaded: {len(rows)}")

    if args.engagement_csv:
        rows = maybe_merge_engagement(rows, Path(args.engagement_csv))
        compute_derived(rows)

    predictors = get_predictor_arrays(rows)
    outcomes = get_outcome_arrays(rows)

    if not outcomes:
        print()
        print("=" * 78)
        print("  NO ENGAGEMENT METRICS in this dataset. Validation impossible.")
        print("=" * 78)
        print()
        print("To validate, you need engagement data per reel. Three options:")
        print()
        print("  A) Re-run the inference script with cookies-from-browser so yt-dlp")
        print("     authenticates as you and IG returns the metrics.")
        print()
        print("  B) Manually collect engagement metrics for each reel and put them")
        print("     in a CSV with columns: reel_url,view_count,like_count,comment_count,")
        print("     save_count,share_count,follower_count. Then re-run this script with:")
        print("        python scripts/validate_against_engagement.py results/.../results.csv \\")
        print("            --engagement-csv my_engagement_data.csv")
        print()
        print("  C) Use the Instagram Graph API for accounts you control. Requires")
        print("     Business/Creator accounts and OAuth. Out of scope for a quick test.")
        return 1

    print(f"Predictor variables: {len(predictors)} (8 regions + 4 goal-weighted overalls)")
    print(f"Outcome metrics:     {len(outcomes)}  ({', '.join(outcomes.keys())})")
    n_per_outcome = {k: sum(1 for v in vs if v is not None) for k, vs in outcomes.items()}
    print(f"Coverage per outcome: {n_per_outcome}")

    test_1_per_region(predictors, outcomes)
    test_2_weighting_value(predictors, outcomes)
    test_3_goal_alignment(predictors, outcomes)

    print()
    print("=" * 78)
    print("  HOW TO READ THE RESULTS")
    print("=" * 78)
    print()
    print("  TEST 1 tells you whether brain regions predict engagement at all.")
    print("    Lots of rho > 0.3 with p < 0.05 = brain model is doing something useful.")
    print("    All rho < 0.2 = brain scores are noise wrt your engagement metrics.")
    print()
    print("  TEST 2 tells you whether the goal weights are pulling weight or noise.")
    print("    If overall_<goal> beats best single region, weights add value.")
    print("    If overall_<goal> matches best region, weights are just emphasis.")
    print("    If overall_<goal> loses to best region, weights are HURTING.")
    print()
    print("  TEST 3 tells you whether the goal labels are correctly aligned.")
    print("    Lots of 'ALIGNED' = the conversion/awareness/engagement labels mean what they say.")
    print("    Lots of 'misaligned' = the goal weights need re-tuning.")
    print("    Lots of 'labels = cosmetic' = the four overall_* are basically the same number.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
