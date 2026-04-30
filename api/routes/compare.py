"""POST /compare — side-by-side analysis of two existing reports.

The frontend's `/compare` page lets a creator pick two of their past
runs and see "which one wins and why". This endpoint does the lifting:

1. Loads both BrainReports from the store (auth-checked — refuses if
   either belongs to a different user).
2. Computes per-region deltas in pure Python.
3. Asks Claude (via the same Anthropic client used for suggestions) to
   write a short "why A wins" paragraph grounded in the deltas.

Returns ComparisonResult so the frontend can render both reports +
diff + summary in one round-trip.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from core.schemas import BrainReport, ComparisonResult
from services.persistence.reports import get_store

from ..auth import require_user

router = APIRouter()
_log = logging.getLogger(__name__)


class CompareRequest(BaseModel):
    request_id_a: str
    request_id_b: str


_COMPARE_SYSTEM_PROMPT = """You are a marketing-neuroscience expert. Two pieces of content have been analyzed and scored across eight brain regions on a 0-100 scale, weighted by a marketing goal (Conversion, Awareness, Engagement, or Brand Recall). Your job is to explain — in plain English — which one is more effective for the goal and why.

Rules:
- Speak in concrete terms: "Post B's amygdala score is 12 points higher, meaning it triggers more emotional engagement."
- Reference 2-3 specific regions where the gap is widest. Don't enumerate all eight.
- If the overall winner is close (delta < 5), say "they're roughly tied" rather than forcing a winner.
- 3-5 short paragraphs maximum. No bullet lists. No filler.
- If the captions or additional context differ meaningfully, factor that in.
"""


def _build_compare_user_prompt(
    a: BrainReport, b: BrainReport, deltas: dict[str, float], overall_delta: float
) -> str:
    lines: list[str] = []
    lines.append(f"Goal: {a.goal.value}")
    if a.goal != b.goal:
        lines.append(f"(NOTE: Post B was scored against {b.goal.value} — comparison is goal-mismatched.)")
    lines.append("")
    lines.append(f"Post A overall: {a.overall_score:.1f}")
    lines.append(f"Post B overall: {b.overall_score:.1f}")
    lines.append(f"Delta (B - A): {overall_delta:+.1f}")
    lines.append("")
    lines.append("Per-region scores (A → B, delta):")
    for region, delta in sorted(deltas.items(), key=lambda kv: -abs(kv[1])):
        a_score = a.region_scores.get(region, 0.0)
        b_score = b.region_scores.get(region, 0.0)
        lines.append(f"  {region:20s}  {a_score:5.1f} → {b_score:5.1f}  ({delta:+.1f})")

    if a.caption_text or b.caption_text:
        lines.append("")
        lines.append(f"Caption A: {a.caption_text or '(none)'}")
        lines.append(f"Caption B: {b.caption_text or '(none)'}")
    if a.additional_context or b.additional_context:
        lines.append("")
        lines.append(f"Brand context A: {a.additional_context or '(none)'}")
        lines.append(f"Brand context B: {b.additional_context or '(none)'}")

    lines.append("")
    lines.append("Write the comparison in the format described in the system prompt.")
    return "\n".join(lines)


def _llm_summary(a: BrainReport, b: BrainReport, deltas, overall_delta) -> str:
    """Return a Claude-generated comparison paragraph, or a deterministic
    fallback if the LLM client isn't available / errors out."""
    try:
        from services.suggestions.llm import get_llm_client
    except Exception as e:
        _log.warning("compare llm import failed: %s", e)
        return _fallback_summary(deltas, overall_delta)
    try:
        client = get_llm_client()
    except Exception as e:
        _log.warning("compare llm client init failed: %s", e)
        return _fallback_summary(deltas, overall_delta)
    try:
        # The mock + real clients both implement chat_json; for compare
        # we want prose, not JSON. Wrap the response to coerce.
        result = client.chat_json(
            system=_COMPARE_SYSTEM_PROMPT,
            user=_build_compare_user_prompt(a, b, deltas, overall_delta)
            + "\n\nReturn JSON: {\"summary\": \"...\"}",
        )
        if isinstance(result, dict) and "summary" in result:
            return str(result["summary"])
        # Mock client returns a list; flatten to string.
        return json.dumps(result) if not isinstance(result, str) else result
    except Exception as e:
        _log.warning("compare llm call failed: %s", e)
        return _fallback_summary(deltas, overall_delta)


def _fallback_summary(deltas: dict[str, float], overall_delta: float) -> str:
    biggest = sorted(deltas.items(), key=lambda kv: -abs(kv[1]))[:3]
    winner = "B" if overall_delta > 0 else "A"
    if abs(overall_delta) < 5:
        opener = "Posts A and B are roughly tied overall."
    else:
        opener = f"Post {winner} is the stronger performer ({abs(overall_delta):.1f} points overall)."
    deltas_str = ", ".join(
        f"{region} {'+' if delta > 0 else ''}{delta:.0f}" for region, delta in biggest
    )
    return (
        f"{opener} Biggest regional differences: {deltas_str}. "
        "(LLM-generated narrative unavailable; install the suggestions extra to enable.)"
    )


@router.post("/compare", response_model=ComparisonResult)
def compare(
    body: CompareRequest, user_id: str = Depends(require_user)
) -> ComparisonResult:
    store = get_store()
    if store is None:
        raise HTTPException(
            status_code=501,
            detail="Reports persistence not configured. Set DATABASE_URL env var.",
        )

    report_a = store.get(body.request_id_a)
    report_b = store.get(body.request_id_b)
    if report_a is None:
        raise HTTPException(status_code=404, detail=f"Report {body.request_id_a} not found")
    if report_b is None:
        raise HTTPException(status_code=404, detail=f"Report {body.request_id_b} not found")
    # Both must belong to the caller. RLS already prevents cross-user
    # reads via the anon key, but the API uses the service-role key so
    # we enforce ownership here.
    for r in (report_a, report_b):
        if r.user_id and r.user_id != user_id:
            raise HTTPException(status_code=404, detail="report not found")

    # Per-region deltas: B minus A. Positive = B wins that region.
    all_regions = set(report_a.region_scores) | set(report_b.region_scores)
    deltas = {
        region: float(
            report_b.region_scores.get(region, 0.0)
            - report_a.region_scores.get(region, 0.0)
        )
        for region in all_regions
    }
    overall_delta = float(report_b.overall_score - report_a.overall_score)
    if abs(overall_delta) < 0.5:
        winner = "tie"
    elif overall_delta > 0:
        winner = "b"
    else:
        winner = "a"

    return ComparisonResult(
        report_a=report_a,
        report_b=report_b,
        overall_delta=overall_delta,
        per_region_delta=deltas,
        winner=winner,  # type: ignore[arg-type]
        llm_summary=_llm_summary(report_a, report_b, deltas, overall_delta),
    )
