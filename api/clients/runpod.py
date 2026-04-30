"""RunPod GPU worker client.

Stage 1 ships with a `MockRunPodClient` that returns deterministic
synthetic predictions. The real `RunPodPodClient` and `RunPodClient`
talk to a deployed RunPod Pod or Serverless endpoint respectively.
All three return a `PredictResponse` carrying both the `(T, 20484)`
cortical predictions AND the timestamped events used by Stage 2 for
moment-anchored suggestions.
"""

import base64
import io
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from services.suggestions.schemas import Event


@dataclass
class PredictResponse:
    """Output of a single inference call.

    `predictions` is the raw `(T, 20484)` float32 cortical activation array
    as TRIBE v2 produces. `events` is the timestamped event list (Words,
    Sentences, Audio, Video chunks) tribev2 builds during preprocessing —
    None when the client doesn't have access to events (Stage 2 will
    eventually require them, but Stage 1 doesn't break if absent).
    """

    predictions: np.ndarray
    events: list[Event] | None = None


class RunPodClientProtocol(Protocol):
    def predict(
        self,
        content_url: str | None = None,
        content_type: str = "video",
        *,
        image_urls: list[str] | None = None,
        audio_url: str | None = None,
        caption: str | None = None,
        seconds_per_image: float = 2.5,
    ) -> PredictResponse:
        """Return predictions + timestamped events for the given content.

        Two input shapes:
          - video: pass content_url + content_type='video' (default)
          - post:  pass content_type='post' + image_urls (1-20 entries),
                   optionally audio_url and/or caption. The same shape
                   covers single-image posts and N-image carousels.
        """
        ...


class MockRunPodClient:
    """Synthetic predictions for development.

    If a real fixture from `tests/fixtures/golden_pred_*.npy` exists
    (produced by scripts/build_fixture.py), it is returned as-is — that's
    real-data mode. Events are synthesized either way so the Stage 2
    pipeline has something to annotate.

    Without a fixture: a fresh random `(T, 20484)` array is generated each
    call with per-region biases drawn from N(0, 2). Without those biases,
    averaging hundreds of N(0, 1) vertices per region collapses to ~0
    (LLN) and sigmoid maps that to ~50 across the board.
    """

    def __init__(self, fixtures_dir: Path | None = None):
        self.fixtures_dir = fixtures_dir or (
            Path(__file__).resolve().parents[2] / "tests" / "fixtures"
        )

    def predict(
        self,
        content_url: str | None = None,
        content_type: str = "video",
        *,
        image_urls: list[str] | None = None,
        audio_url: str | None = None,
        caption: str | None = None,
        seconds_per_image: float = 2.5,
    ) -> PredictResponse:
        # The mock layer doesn't actually inspect the inputs — fixtures
        # are deterministic. The post-mode params are accepted so the
        # signature matches the protocol; tests cover both shapes.
        for fixture in sorted(self.fixtures_dir.glob("golden_pred_*.npy")):
            preds = np.load(fixture)
            if preds.ndim == 2 and preds.shape[1] == 20484:
                return PredictResponse(
                    predictions=preds.astype(np.float32),
                    events=_synthesize_events(preds.shape[0]),
                )

        from core.atlas.mapper import REGION_VERTICES

        rng = np.random.default_rng()
        T = 24
        preds = rng.normal(0.0, 0.3, size=(T, 20484)).astype(np.float32)
        for vertex_idx in REGION_VERTICES.values():
            preds[:, vertex_idx] += float(rng.normal(0.0, 2.0))
        return PredictResponse(
            predictions=preds, events=_synthesize_events(T, rng=rng)
        )


def _synthesize_events(T: int, rng: np.random.Generator | None = None) -> list[Event]:
    """Make up a plausible-looking event list for mock mode.

    Used so the Stage 2 moment-annotation path has something to bind to
    when no real events are available. Never used in production.
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)
    sample_words = [
        "this", "product", "changes", "everything", "available",
        "in", "three", "colors", "click", "to", "buy", "now",
    ]
    out: list[Event] = []
    out.append(Event(type="Audio", start_s=0.0, duration_s=float(T)))
    out.append(Event(type="Video", start_s=0.0, duration_s=float(T)))
    # Scatter words across the duration
    n_words = max(1, T // 3)
    starts = sorted(rng.uniform(0.5, max(0.5, T - 1), size=n_words).tolist())
    for s in starts:
        out.append(
            Event(
                type="Word",
                start_s=float(s),
                duration_s=0.4,
                text=str(rng.choice(sample_words)),
            )
        )
    return out


class RunPodPodClient:
    """Direct HTTP to a deployed Pod's /predict endpoint.

    Used during the bring-up phase (gpu_worker/README.md §5) when iterating
    against a Pod is faster than redeploying a Serverless endpoint. Decodes
    the same response shape that gpu_worker.handler._serialize() produces,
    including the optional `events` array.
    """

    def __init__(self):
        self.url = os.environ["RUNPOD_POD_URL"].rstrip("/")
        self.timeout = int(os.environ.get("RUNPOD_TIMEOUT_SECONDS", "300"))

    def predict(
        self,
        content_url: str | None = None,
        content_type: str = "video",
        *,
        image_urls: list[str] | None = None,
        audio_url: str | None = None,
        caption: str | None = None,
        seconds_per_image: float = 2.5,
    ) -> PredictResponse:
        body = json.dumps(
            _build_request_body(
                content_url=content_url,
                content_type=content_type,
                image_urls=image_urls,
                audio_url=audio_url,
                caption=caption,
                seconds_per_image=seconds_per_image,
            )
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.url}/predict",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                output = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"Pod HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}"
            ) from e
        return _decode_predict_payload(output)


class RunPodClient:
    """Calls a deployed RunPod serverless endpoint over HTTPS.

    Pairs with gpu_worker/handler.py running in serverless mode. Uses
    urllib (stdlib) so no extra runtime deps. The /runsync endpoint is
    synchronous-blocking with a hard cap of 5 min on RunPod's side; for
    longer jobs switch to /run + /status polling (Stage 4).
    """

    def __init__(self):
        self.api_key = os.environ["RUNPOD_API_KEY"]
        self.endpoint_id = os.environ["RUNPOD_ENDPOINT_ID"]
        self.timeout = int(os.environ.get("RUNPOD_TIMEOUT_SECONDS", "300"))

    def predict(
        self,
        content_url: str | None = None,
        content_type: str = "video",
        *,
        image_urls: list[str] | None = None,
        audio_url: str | None = None,
        caption: str | None = None,
        seconds_per_image: float = 2.5,
    ) -> PredictResponse:
        url = f"https://api.runpod.ai/v2/{self.endpoint_id}/runsync"
        body = json.dumps(
            {
                "input": _build_request_body(
                    content_url=content_url,
                    content_type=content_type,
                    image_urls=image_urls,
                    audio_url=audio_url,
                    caption=caption,
                    seconds_per_image=seconds_per_image,
                )
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"RunPod HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}"
            ) from e

        status = payload.get("status")
        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise RuntimeError(f"RunPod inference {status}: {payload.get('error')}")

        output = payload.get("output")
        if output is None:
            raise RuntimeError(f"RunPod returned no output (status={status}): {payload}")
        return _decode_predict_payload(output)


def _build_request_body(
    *,
    content_url: str | None,
    content_type: str,
    image_urls: list[str] | None,
    audio_url: str | None,
    caption: str | None,
    seconds_per_image: float,
) -> dict:
    """Build the worker's input dict, omitting None values for cleanliness."""
    body: dict = {"content_type": content_type}
    if content_url is not None:
        body["content_url"] = content_url
    if image_urls is not None:
        body["image_urls"] = list(image_urls)
        body["seconds_per_image"] = seconds_per_image
    if audio_url is not None:
        body["audio_url"] = audio_url
    if caption is not None:
        body["caption"] = caption
    return body


def _decode_predict_payload(output: dict) -> PredictResponse:
    """Decode the (predictions, events) payload from a worker response."""
    npz_bytes = base64.b64decode(output["data_b64"])
    with np.load(io.BytesIO(npz_bytes)) as data:
        predictions = data["preds"].astype(np.float32)

    raw_events = output.get("events") or []
    events = [Event(**e) for e in raw_events] if raw_events else None
    return PredictResponse(predictions=predictions, events=events)


def get_client() -> RunPodClientProtocol:
    """Pick the inference client based on INFERENCE_MODE.

    INFERENCE_MODE=mock (default)
        Returns MockRunPodClient — synthetic predictions or local fixture.
        Free, deterministic, ideal for frontend dev and unit tests. The
        real RunPod env vars can stay set in .env without taking effect.
    INFERENCE_MODE=runpod
        Real GPU inference. Picks RunPodPodClient if RUNPOD_POD_URL is
        set; otherwise RunPodClient (serverless) if RUNPOD_ENDPOINT_ID
        and RUNPOD_API_KEY are both set. Errors loudly if neither.
    """
    mode = os.environ.get("INFERENCE_MODE", "mock").strip().lower()

    if mode == "mock":
        return MockRunPodClient()

    if mode == "runpod":
        if os.environ.get("RUNPOD_POD_URL"):
            return RunPodPodClient()
        if os.environ.get("RUNPOD_ENDPOINT_ID") and os.environ.get("RUNPOD_API_KEY"):
            return RunPodClient()
        raise RuntimeError(
            "INFERENCE_MODE=runpod but neither RUNPOD_POD_URL nor "
            "(RUNPOD_ENDPOINT_ID + RUNPOD_API_KEY) is configured. "
            "Set RUNPOD_POD_URL for Pod mode or both serverless vars, "
            "or switch INFERENCE_MODE back to 'mock'."
        )

    raise ValueError(
        f"INFERENCE_MODE={mode!r} not recognized. Use 'mock' or 'runpod'."
    )
