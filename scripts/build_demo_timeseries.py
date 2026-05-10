"""Generate `plan.region_timeseries` blocks for the 3 demo runs.

The Results screen sparkline expects per-second activation values per
region. We don't have real TRIBE timeseries for canned demos, so we
synthesize a deterministic curve from the existing static `score`
values plus the suggestion `peak_start_s` / `peak_end_s` windows so
the curve actually justifies what each suggestion claims.

Usage:
    uv run python scripts/build_demo_timeseries.py

Idempotent — re-running produces identical curves (seeded RNG).
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

REGION_KEYS = ("memory", "emotion", "attention", "language", "face", "reward")
DURATION_S = 60  # all 3 demos are roughly one-minute spots
DEMO_DIR = Path(__file__).resolve().parents[1] / "data" / "demo_runs"


def _smooth(values: list[float], passes: int = 2) -> list[float]:
    """Box-filter pass — soften jaggedness so curves read as natural."""
    out = list(values)
    for _ in range(passes):
        smoothed = []
        for i, v in enumerate(out):
            lo = max(0, i - 1)
            hi = min(len(out), i + 2)
            smoothed.append(sum(out[lo:hi]) / (hi - lo))
        out = smoothed
    return out


def _build_region_curve(
    score: float,
    peaks: list[tuple[float, float]],
    *,
    seed: int,
) -> list[float]:
    """Curve for one region: gentle wandering baseline with a lifted bump
    over each peak window. Numbers stay in 0..100."""
    rng = random.Random(seed)
    base = max(5.0, score - 12.0)
    peak_height = min(100.0, score + 15.0)

    # Wandering baseline — tiny per-second drift around `base`.
    curve = [base + rng.uniform(-3.5, 3.5) for _ in range(DURATION_S)]

    # Bump up across each suggestion's peak window. Use a half-cosine
    # rise/fall so the shoulders look natural rather than a square wave.
    for start_s, end_s in peaks:
        s = max(0, int(math.floor(start_s)))
        e = min(DURATION_S, int(math.ceil(end_s)) + 1)
        if e <= s:
            continue
        # Extend rise/fall by ~2s on each side for smoother shoulders.
        ramp = 2
        full_lo = max(0, s - ramp)
        full_hi = min(DURATION_S, e + ramp)
        width = full_hi - full_lo
        for i in range(full_lo, full_hi):
            t = (i - full_lo) / max(1, width - 1)  # 0..1
            envelope = 0.5 - 0.5 * math.cos(2 * math.pi * t)  # 0..1..0
            target = base + envelope * (peak_height - base)
            # Use the higher of baseline noise vs lifted target so we
            # don't accidentally drag a noisy spike DOWN.
            curve[i] = max(curve[i], target)

    curve = _smooth(curve, passes=2)
    return [round(max(0.0, min(100.0, v)), 1) for v in curve]


def _build_demo(demo: dict) -> dict[str, list[float]]:
    plan = demo["plan"]
    region_scores = {r["key"]: float(r["score"]) for r in plan["regions"]}

    # Group peak windows by region using the suggestion area.
    peaks_by_region: dict[str, list[tuple[float, float]]] = {
        k: [] for k in REGION_KEYS
    }
    for sug in plan["suggestions"]:
        ps, pe = sug.get("peak_start_s"), sug.get("peak_end_s")
        if ps is None or pe is None:
            continue
        peaks_by_region.setdefault(sug["area"], []).append((float(ps), float(pe)))

    # Stable seed per (demo, region) so re-runs produce identical output.
    demo_seed = sum(ord(c) for c in demo["demo_id"])
    return {
        region: _build_region_curve(
            region_scores.get(region, 50.0),
            peaks_by_region.get(region, []),
            seed=demo_seed * 100 + i,
        )
        for i, region in enumerate(REGION_KEYS)
    }


def main() -> None:
    files = sorted(DEMO_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"No demo files found in {DEMO_DIR}")
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "plan" not in data:
            continue
        data["plan"]["region_timeseries"] = _build_demo(data)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"updated {path.name}")


if __name__ == "__main__":
    main()
