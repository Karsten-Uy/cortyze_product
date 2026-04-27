"""Benchmark a deployed RunPod Pod or Serverless endpoint.

Runs a structured suite (warm latency distribution, content-length scaling,
parallel concurrency) and writes a markdown report to docs/runpod_benchmark.md.

Captures what's needed for Stage 1.2 capacity planning per the plan in
~/.claude/plans/can-you-show-me-unified-ladybug.md:
  - Warm-inference latency p50/p90/p95
  - Latency vs. content length (3 clips of varying duration)
  - Concurrency: 2 + 4 parallel /predict calls
  - First-inference time (proxy for cold start)

Cold-start time and peak VRAM are captured manually (see comments below):
the cold-start signal lives in the pod's container logs, and VRAM requires
nvidia-smi inside the pod. This script records placeholders for both so the
final report has slots ready to be filled in.

Usage:
    # Pod mode (against a Pod's HTTPS proxy URL):
    RUNPOD_POD_URL=https://abc123-8000.proxy.runpod.net \\
      uv run python scripts/benchmark_runpod.py pod

    # Serverless mode (uses RUNPOD_ENDPOINT_ID + RUNPOD_API_KEY from env):
    RUNPOD_ENDPOINT_ID=xyz789 RUNPOD_API_KEY=... \\
      uv run python scripts/benchmark_runpod.py serverless
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import io
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CLIPS = {
    # Approximate durations baked into the labels — actual processed event
    # count depends on tribev2's chunking.
    "sintel_~52s": "https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4",
    "bbb_~10min": "https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_480p_h264.mov",
    "tos_~12min": "https://download.blender.org/mango/download.blender.org/demo/movies/ToS/ToS-4k-1920.mov",
}
WARM_RUNS = 10
DEFAULT_TIMEOUT = 600


def _post_pod(pod_url: str, content_url: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
    body = json.dumps({"content_url": content_url, "content_type": "video"}).encode()
    req = urllib.request.Request(
        f"{pod_url.rstrip('/')}/predict",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_serverless(
    endpoint_id: str, api_key: str, content_url: str, timeout: int = DEFAULT_TIMEOUT
) -> dict:
    body = json.dumps(
        {"input": {"content_url": content_url, "content_type": "video"}}
    ).encode()
    req = urllib.request.Request(
        f"https://api.runpod.ai/v2/{endpoint_id}/runsync",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("status") in ("FAILED", "CANCELLED", "TIMED_OUT"):
        raise RuntimeError(f"Serverless inference {payload['status']}: {payload}")
    return payload["output"]


def _decode_shape(output: dict) -> tuple[int, int]:
    return tuple(output["shape"])  # type: ignore[return-value]


def _decode_array(output: dict) -> np.ndarray:
    npz_bytes = base64.b64decode(output["data_b64"])
    with np.load(io.BytesIO(npz_bytes)) as data:
        return data["preds"]


def _summarize(times: list[float]) -> dict[str, float]:
    if not times:
        return {}
    return {
        "n": len(times),
        "min": min(times),
        "p50": statistics.median(times),
        "mean": statistics.mean(times),
        "p90": float(np.percentile(times, 90)),
        "p95": float(np.percentile(times, 95)),
        "max": max(times),
    }


def _bench_one(post_fn, content_url: str) -> tuple[float, dict]:
    t0 = time.monotonic()
    output = post_fn(content_url)
    elapsed = time.monotonic() - t0
    return elapsed, output


def run(mode: str) -> dict:
    if mode == "pod":
        pod_url = os.environ["RUNPOD_POD_URL"]
        post = lambda url: _post_pod(pod_url, url)  # noqa: E731
        target = pod_url
    elif mode == "serverless":
        endpoint_id = os.environ["RUNPOD_ENDPOINT_ID"]
        api_key = os.environ["RUNPOD_API_KEY"]
        post = lambda url: _post_serverless(endpoint_id, api_key, url)  # noqa: E731
        target = f"serverless:{endpoint_id}"
    else:
        raise ValueError(f"unknown mode {mode!r}")

    sintel_url = CLIPS["sintel_~52s"]

    print(f"[{mode}] Target: {target}")
    print(f"[{mode}] First inference (cold-ish for serverless, warm for pod)...")
    first_elapsed, first_out = _bench_one(post, sintel_url)
    first_shape = _decode_shape(first_out)
    print(f"  → {first_elapsed:.1f}s, shape={first_shape}")

    print(f"[{mode}] Warm latency over {WARM_RUNS} sequential calls (sintel)...")
    warm_times: list[float] = []
    for i in range(WARM_RUNS):
        elapsed, _ = _bench_one(post, sintel_url)
        warm_times.append(elapsed)
        print(f"  call {i + 1:2d}: {elapsed:.1f}s")
    warm_stats = _summarize(warm_times)

    print(f"[{mode}] Latency vs. content length...")
    by_clip: dict[str, dict] = {}
    for name, url in CLIPS.items():
        elapsed, output = _bench_one(post, url)
        shape = _decode_shape(output)
        by_clip[name] = {"elapsed_s": elapsed, "T": shape[0]}
        print(f"  {name:20s} → {elapsed:.1f}s, T={shape[0]}")

    print(f"[{mode}] Concurrency: 2 parallel calls...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        t0 = time.monotonic()
        futures = [ex.submit(_bench_one, post, sintel_url) for _ in range(2)]
        elapsed_pairs = [f.result()[0] for f in concurrent.futures.as_completed(futures)]
        wall_2 = time.monotonic() - t0
    print(f"  individual: {elapsed_pairs}, wall: {wall_2:.1f}s")

    print(f"[{mode}] Concurrency: 4 parallel calls...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        t0 = time.monotonic()
        futures = [ex.submit(_bench_one, post, sintel_url) for _ in range(4)]
        elapsed_quads = [f.result()[0] for f in concurrent.futures.as_completed(futures)]
        wall_4 = time.monotonic() - t0
    print(f"  individual: {elapsed_quads}, wall: {wall_4:.1f}s")

    return {
        "mode": mode,
        "target": target,
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "first_inference_s": first_elapsed,
        "warm_latency": warm_stats,
        "by_clip": by_clip,
        "concurrency_2_parallel": {"wall_s": wall_2, "per_call_s": elapsed_pairs},
        "concurrency_4_parallel": {"wall_s": wall_4, "per_call_s": elapsed_quads},
    }


def render_markdown(pod_results: dict | None, serverless_results: dict | None) -> str:
    out: list[str] = []
    out.append("# RunPod benchmark — Stage 1.2 capacity data\n")
    out.append(
        f"Generated by `scripts/benchmark_runpod.py` on {datetime.now(timezone.utc).isoformat(timespec='seconds')}.\n"
    )
    out.append(
        "## Manually captured (fill in from pod logs / nvidia-smi)\n\n"
        "| Metric | Value | How |\n"
        "|---|---|---|\n"
        "| Cold start (container start → `Model ready.`) | `<TBD>` | Pod logs, look for the timestamped `Model ready.` line printed by `gpu_worker/inference.py` |\n"
        "| Peak VRAM during inference | `<TBD> / 48 GB` | SSH into pod, `watch -n 0.5 nvidia-smi`, capture peak `Memory-Usage` while issuing inferences |\n"
        "| Image pull time on cold node | `<TBD>` | Pod event timeline in RunPod dashboard |\n"
    )

    for label, results in (("Pod mode", pod_results), ("Serverless mode", serverless_results)):
        if not results:
            continue
        out.append(f"\n## {label}\n")
        out.append(f"- Target: `{results['target']}`\n")
        out.append(f"- Ran at: {results['ran_at']}\n")
        out.append(f"- First inference: **{results['first_inference_s']:.1f} s**\n")

        w = results["warm_latency"]
        out.append(
            "\n### Warm latency ({} sequential calls, sintel)\n\n"
            "| Stat | Seconds |\n|---|---|\n"
            "| min  | {min:.1f} |\n"
            "| p50  | {p50:.1f} |\n"
            "| mean | {mean:.1f} |\n"
            "| p90  | {p90:.1f} |\n"
            "| p95  | {p95:.1f} |\n"
            "| max  | {max:.1f} |\n".format(w["n"], **w)
        )

        out.append("\n### Latency vs. content length\n\n| Clip | T (timesteps) | Elapsed |\n|---|---|---|\n")
        for clip, d in results["by_clip"].items():
            out.append(f"| {clip} | {d['T']} | {d['elapsed_s']:.1f} s |\n")

        out.append("\n### Concurrency\n\n| Parallel | Wall time | Per-call (s) |\n|---|---|---|\n")
        for n, key in ((2, "concurrency_2_parallel"), (4, "concurrency_4_parallel")):
            d = results[key]
            per = ", ".join(f"{x:.1f}" for x in d["per_call_s"])
            out.append(f"| {n} | {d['wall_s']:.1f} s | {per} |\n")

    out.append(
        "\n## Scaling implications (fill in after both modes complete)\n\n"
        "- **min_workers / max_workers:** `<TBD — based on observed concurrency ceiling>`\n"
        "- **idle_timeout:** `<TBD — balance cold-start UX vs. cost>`\n"
        "- **GPU choice:** `<A40 sufficient? or move to A100 40GB?>` — depends on peak VRAM above.\n"
        "- **Per-analysis cost:** ~`<warm p50> sec * $<rate>/3600 = $<cost>` per analysis.\n"
        "- **Recommendation for IMPLEMENTATION_PLAN.md cost estimate update:** `<TBD>`\n"
    )
    return "".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["pod", "serverless"])
    parser.add_argument(
        "--report",
        default="docs/runpod_benchmark.md",
        help="Markdown output path (relative to cortyze_product/)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing report (e.g., add serverless results next to pod results)",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent.parent
    report_path = here / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)

    pod_prev: dict | None = None
    serverless_prev: dict | None = None
    sidecar = report_path.with_suffix(".json")
    if args.append and sidecar.exists():
        cached = json.loads(sidecar.read_text())
        pod_prev = cached.get("pod")
        serverless_prev = cached.get("serverless")

    try:
        results = run(args.mode)
    except Exception as e:
        print(f"\nBenchmark failed: {e}", file=sys.stderr)
        return 1

    if args.mode == "pod":
        pod_prev = results
    else:
        serverless_prev = results

    sidecar.write_text(
        json.dumps({"pod": pod_prev, "serverless": serverless_prev}, indent=2) + "\n"
    )
    report_path.write_text(render_markdown(pod_prev, serverless_prev))
    print(f"\nReport: {report_path.relative_to(here)}")
    print(f"Sidecar JSON (for re-render): {sidecar.relative_to(here)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
