"""Sample-run loader for the Lab-bench "Try a sample" cards.

Reads canned `SuggestionPlan` payloads + form defaults out of
`data/demo_runs/*.json`. Each file is one demo and gets one card on
the Lab bench. The orchestrator's `_demo_pipeline` consumes
`load_demo_run(demo_id)` to short-circuit the real Phase 1-4 pipeline.

JSON authored by hand (or copied from a real run) — no model code,
no live data sources, no LLM calls.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from core.goals_v2 import GoalKey
from core.schemas_v2 import SuggestionPlan

_log = logging.getLogger("cortyze.demo")

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "demo_runs"
_COMPARISON_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "demo_comparison.json"
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class DemoFormDefaults(BaseModel):
    """Field values pre-filled on the Lab-bench form when this demo
    card is clicked. Mirrors the relevant `LabBenchInput` subset."""

    name: str
    goal: GoalKey
    brief: str = ""
    caption: str = ""


class DemoRun(BaseModel):
    """One full demo entry. Loaded from `data/demo_runs/<demo_id>.json`."""

    demo_id: str
    label: str
    tagline: str = ""
    thumbnail_url: str
    form_defaults: DemoFormDefaults
    media_url: str | None = None
    media_object_key: str | None = None
    kind: Literal["Video", "Image"] = "Video"
    plan: SuggestionPlan


class ComparisonNarrative(BaseModel):
    """Hand-written copy explaining why one demo wins over the others.

    Loaded from `data/demo_comparison.json`. Lives outside `data/demo_runs/`
    so the demo glob doesn't try to parse it as a `DemoRun`.
    """

    headline: str
    winner_demo_id: str
    subhead: str
    # Region key (memory/emotion/attention/language/face/reward) -> winner demo_id.
    per_region_winners: dict[str, str]
    # demo_id -> list of bullet points displayed under that demo's column.
    demo_takeaways: dict[str, list[str]]


class DemoSummary(BaseModel):
    """Lightweight projection — shipped to the frontend by `GET /demos`.

    The full `plan` is not in the summary because the frontend only
    needs it after clicking; trimming the payload keeps the demo
    listing endpoint cheap. `form_defaults` IS included so the Lab
    bench can prefill the campaign-name / goal / brief / caption fields
    on click without a second round-trip. `media_url` is included so
    the Lab-bench sample-card thumbnails can link out to the source
    YouTube clip in a new tab.
    """

    demo_id: str
    label: str
    tagline: str
    thumbnail_url: str
    kind: Literal["Video", "Image"]
    form_defaults: DemoFormDefaults
    media_url: str | None = None


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_all() -> dict[str, DemoRun]:
    if not _DATA_DIR.exists():
        return {}
    out: dict[str, DemoRun] = {}
    for path in sorted(_DATA_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            run = DemoRun(**data)
        except Exception as exc:  # noqa: BLE001
            _log.warning("skipping demo %s: %s", path.name, exc)
            continue
        out[run.demo_id] = run
    if not out:
        _log.info("no demo runs found in %s", _DATA_DIR)
    return out


def reload() -> None:
    """Drop the cache so the next call re-reads from disk. Test-friendly."""
    _load_all.cache_clear()
    _load_comparison.cache_clear()


@lru_cache(maxsize=1)
def _load_comparison() -> dict[str, ComparisonNarrative]:
    """Read `data/demo_comparison.json` into `{pair_key: ComparisonNarrative}`.

    Pair key = the two demo_ids sorted alphabetically and joined with
    `_vs_` — e.g. `apple_1984_vs_lays`. Symmetric so the route handler
    can normalize either input order.
    """
    if not _COMPARISON_PATH.exists():
        return {}
    try:
        data = json.loads(_COMPARISON_PATH.read_text(encoding="utf-8"))
        raw_pairs = data.get("pairs", {})
        return {
            key: ComparisonNarrative(**value) for key, value in raw_pairs.items()
        }
    except Exception as exc:  # noqa: BLE001
        _log.warning("could not load demo comparison narrative: %s", exc)
        return {}


def _pair_key(a: str, b: str) -> str:
    return "_vs_".join(sorted((a, b)))


def load_comparison_narrative(a: str, b: str) -> ComparisonNarrative | None:
    """Hand-written narrative for the `(a, b)` pair, order-independent."""
    if a == b:
        return None
    return _load_comparison().get(_pair_key(a, b))


def list_demos() -> list[DemoSummary]:
    """Summaries for the Lab-bench cards."""
    return [
        DemoSummary(
            demo_id=r.demo_id,
            label=r.label,
            tagline=r.tagline,
            thumbnail_url=r.thumbnail_url,
            kind=r.kind,
            form_defaults=r.form_defaults,
            media_url=r.media_url,
        )
        for r in _load_all().values()
    ]


def load_demo_run(demo_id: str) -> DemoRun | None:
    """Full demo payload, or `None` if the id isn't registered.

    Callers (the orchestrator) treat `None` as a hard 404 — the demo
    card the user clicked must be the same set the `/demos` endpoint
    returned, so a None here is a programming error worth surfacing.
    """
    return _load_all().get(demo_id)
