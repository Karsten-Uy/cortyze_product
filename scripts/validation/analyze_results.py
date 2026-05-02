"""Sanity-check analysis of a results.csv: is the brain model producing
meaningful, differentiated output, or is everything flat/random/broken?

Answers four questions:

  1. Do region scores VARY across reels? (if stdev < 1, model isn't
     differentiating — bug or content too uniform)
  2. Do different ACCOUNTS have different region profiles? (Marvel
     should look different from Harvard; if they look the same, the
     model isn't picking up on content type)
  3. Where are the OUTLIER reels per region? (manual gut check — do
     the high-visual-cortex reels actually look visually busy?)
  4. How big is the inter-account difference vs within-account variance?
     (real signal vs noise; reported as Cohen's d-style effect size)

Usage:
    python scripts/analyze_results.py results/<run_dir>/results.csv

Engagement-correlation analysis (likes/views vs region scores) requires
yt-dlp to have returned engagement metrics. Many IG posts won't surface
those without authentication; this script will note if metrics are
missing and skip that section.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

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


def fnum(s):
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def section(title: str) -> None:
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def load_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def per_region_distribution(ok_rows: list[dict]) -> None:
    section("1. PER-REGION SCORE DISTRIBUTION")
    print(f"{'Region':<22} {'mean':>6} {'stdev':>6} {'min':>6} {'max':>6} {'range':>6}")
    print("-" * 60)
    flat_regions = []
    for region in REGIONS:
        vals = [fnum(r.get(f"region_{region}")) for r in ok_rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0
        rng = max(vals) - min(vals)
        flag = " <- flat" if s < 1.0 else ""
        if s < 1.0:
            flat_regions.append(region)
        print(f"  {region:<20} {m:6.1f} {s:6.1f} {min(vals):6.1f} {max(vals):6.1f} {rng:6.1f}{flag}")

    print()
    print("INTERPRETATION:")
    print("  stdev > 5  -> scores meaningfully vary across reels (model differentiates) GOOD")
    print("  stdev 1-5  -> some variation, possibly real but small  OK")
    print("  stdev < 1  -> region is FLAT, model isn't seeing variation  BAD")
    if flat_regions:
        print(f"\n  WARNING: flat regions: {', '.join(flat_regions)}")
        print("           Either the calibration squashes everything to a constant,")
        print("           or your reels lack the variation that region detects.")


def per_account_profiles(ok_rows: list[dict]) -> dict[str, list[dict]]:
    section("2. PER-ACCOUNT REGION PROFILES")
    by_uploader: dict[str, list[dict]] = defaultdict(list)
    for r in ok_rows:
        u = r.get("plat_uploader") or "unknown"
        by_uploader[u].append(r)

    eligible = {u: rs for u, rs in by_uploader.items() if len(rs) >= 3}
    print(f"\nAccounts with >=3 reels: {len(eligible)}")
    if not eligible:
        print("  Not enough per-account data yet — let the batch finish more reels.")
        return {}

    for uploader, rs in sorted(eligible.items(), key=lambda kv: -len(kv[1])):
        print(f"\n  {uploader} (n={len(rs)})")
        for region in REGIONS:
            vals = [fnum(r.get(f"region_{region}")) for r in rs]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            m = statistics.mean(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else 0
            bar_len = max(0, min(40, int((m - 30) * 1.0)))
            bar = "#" * bar_len
            print(f"    {region:<22} {m:5.1f} +/- {s:4.1f}  {bar}")

    print()
    print("INTERPRETATION:")
    print("  - Different accounts should have visibly DIFFERENT profiles.")
    print("    e.g. Marvel/NBA -> high visual_cortex + amygdala (action, emotion)")
    print("    e.g. Harvard    -> high prefrontal + temporal_language (cognitive, verbal)")
    print("    e.g. Mr Beast   -> high reward + amygdala (excitement)")
    print("  - If all accounts look identical -> model isn't sensitive to content type.")
    return eligible


def outlier_reels(ok_rows: list[dict]) -> None:
    section("3. TOP / BOTTOM REELS PER REGION  (the manual gut check)")
    print()
    print("Click the URLs. If the high-scoring reels actually look the way you'd expect")
    print("for that brain region, the model is doing something real.")
    for region in REGIONS:
        scored = []
        for r in ok_rows:
            v = fnum(r.get(f"region_{region}"))
            if v is not None:
                scored.append((v, r))
        if len(scored) < 3:
            continue
        scored.sort(key=lambda x: -x[0])

        print(f"\n  {region.upper()}")
        print(f"    TOP (highest activation):")
        for sc, r in scored[:3]:
            cap = (r.get("caption") or "")[:55].replace("\n", " ")
            uploader = r.get("plat_uploader") or "?"
            print(f"      {sc:5.1f}  [{uploader[:14]:<14}]  {r['reel_url']}")
            if cap:
                print(f"             {cap}...")
        print(f"    BOTTOM (lowest activation):")
        for sc, r in scored[-3:]:
            cap = (r.get("caption") or "")[:55].replace("\n", " ")
            uploader = r.get("plat_uploader") or "?"
            print(f"      {sc:5.1f}  [{uploader[:14]:<14}]  {r['reel_url']}")
            if cap:
                print(f"             {cap}...")


def account_effect_sizes(eligible: dict[str, list[dict]]) -> None:
    if len(eligible) < 2:
        return
    section("4. INTER-ACCOUNT EFFECT SIZES (Cohen's d)")
    print()
    print("How distinguishable are accounts on each region?")
    print("  d > 0.8  large effect (model strongly separates these accounts)")
    print("  d > 0.5  medium")
    print("  d > 0.2  small")
    print("  d < 0.2  trivial (accounts indistinguishable on this region)")

    accounts = list(eligible.keys())
    for region in REGIONS:
        print(f"\n  {region}:")
        # Compute mean+std per account
        stats = {}
        for u in accounts:
            vals = [fnum(r.get(f"region_{region}")) for r in eligible[u]]
            vals = [v for v in vals if v is not None]
            if len(vals) < 2:
                continue
            stats[u] = (statistics.mean(vals), statistics.stdev(vals), len(vals))
        if len(stats) < 2:
            continue
        for u, (m, s, n) in sorted(stats.items(), key=lambda kv: -kv[1][0]):
            print(f"    {u:<22}  mean={m:5.1f}  sd={s:4.1f}  (n={n})")
        # Pairwise effect size between top and bottom
        sorted_by_mean = sorted(stats.items(), key=lambda kv: kv[1][0])
        bot, top = sorted_by_mean[0], sorted_by_mean[-1]
        m1, s1, n1 = bot[1]
        m2, s2, n2 = top[1]
        # pooled stdev
        pooled = ((s1 ** 2 * (n1 - 1) + s2 ** 2 * (n2 - 1)) / (n1 + n2 - 2)) ** 0.5
        d = abs(m2 - m1) / pooled if pooled > 0 else 0
        verdict = (
            "LARGE"  if d > 0.8 else
            "medium" if d > 0.5 else
            "small"  if d > 0.2 else
            "trivial"
        )
        print(f"    -> {top[0]} vs {bot[0]}: d={d:.2f}  ({verdict})")


def engagement_check(ok_rows: list[dict]) -> None:
    section("5. ENGAGEMENT METRICS COVERAGE (validation prerequisite)")
    keys = ["plat_like_count", "plat_view_count", "plat_comment_count", "plat_channel_follower_count"]
    coverage = {}
    for k in keys:
        n = sum(1 for r in ok_rows if fnum(r.get(k)) is not None)
        coverage[k] = (n, len(ok_rows))
        print(f"  {k:<32}: {n}/{len(ok_rows)} reels have data")
    if all(c[0] == 0 for c in coverage.values()):
        print()
        print("  No engagement metrics in this dataset (yt-dlp returned nothing).")
        print("  This means you cannot do the brain<->engagement correlation analysis")
        print("  on this batch — the brain side is fine, but the outcome variable is missing.")
        print("  Workaround: re-run with cookies-from-browser (authenticated yt-dlp) OR")
        print("  hand-collect engagement metrics for the same reels separately.")
    elif any(c[0] >= 3 for c in coverage.values()):
        print()
        print("  Some metrics present. Once batch finishes, do per-region Spearman vs each metric:")
        print("  python -c \"import pandas as pd; df = pd.read_csv('results/.../results.csv');")
        print("            print(df[[c for c in df.columns if c.startswith('region_')] +")
        print("                     ['derived_engagement_per_view']].corr(method='spearman'))\"")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path",
                        help="Path to results.csv produced by reels_to_csv.py")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        return 1

    rows = load_csv(csv_path)
    ok_rows = [r for r in rows if not (r.get("error") or "").strip()]
    print(f"\nFile: {csv_path}")
    print(f"Total rows:    {len(rows)}")
    print(f"Successful:    {len(ok_rows)}")
    print(f"Errored/skip:  {len(rows) - len(ok_rows)}")

    if not ok_rows:
        print("\nNo successful reels — nothing to analyze.")
        return 1

    per_region_distribution(ok_rows)
    eligible = per_account_profiles(ok_rows)
    outlier_reels(ok_rows)
    account_effect_sizes(eligible)
    engagement_check(ok_rows)

    print(f"\n{'='*72}")
    print("  Watch a few of the URLs above. Your eyes are the validation here.")
    print(f"{'='*72}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
