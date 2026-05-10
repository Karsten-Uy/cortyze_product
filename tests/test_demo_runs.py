"""Demo flow — `GET /demos` listing + demo-id short-circuit on `POST /runs`."""

from __future__ import annotations

import time
from typing import Any

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def _poll_until_complete(run_id: str, timeout_s: float = 5.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        r = client.get(f"/runs/{run_id}")
        assert r.status_code == 200, r.text
        last = r.json()
        if last["status"] in ("complete", "failed"):
            return last
        time.sleep(0.05)
    raise AssertionError(
        f"run {run_id} did not finish in {timeout_s}s; last status={last.get('status')}"
    )


def test_demos_endpoint_lists_authored_samples():
    r = client.get("/demos")
    assert r.status_code == 200
    body = r.json()
    # We authored 3 — lays / apple_1984 / coinbase.
    assert len(body) >= 3
    ids = {item["demo_id"] for item in body}
    assert {"lays", "apple_1984", "coinbase"} <= ids
    for item in body:
        # Required listing fields.
        assert item["label"]
        assert item["thumbnail_url"]
        assert item["kind"] in ("Video", "Image")


def test_lays_demo_run_completes_with_canned_plan():
    r = client.post(
        "/runs",
        json={
            "name": "demo-test",
            "goal": "brand_recall",
            "demo_id": "lays",
        },
    )
    assert r.status_code == 202
    run_id = r.json()["run_id"]
    final = _poll_until_complete(run_id)
    assert final["status"] == "complete"
    plan = final["result"]
    assert plan is not None
    # Lays canned plan: score 78, status "Strong", 3 suggestions.
    assert plan["score"] == 78
    assert plan["status"] == "Strong"
    assert len(plan["suggestions"]) == 3
    # The demo's source clip URL must be persisted onto the run so the
    # Results screen's hero player can render it. Without this the
    # video panel goes empty even though the demo JSON has the URL.
    assert final["media_url"], "demo run should expose its source clip URL"
    assert "youtube" in final["media_url"]


def test_three_demos_yield_three_distinct_score_profiles():
    """The whole point of having three samples — confirm they actually
    produce different scores so the demo shows differentiation."""
    scores: dict[str, float] = {}
    for demo_id in ("lays", "apple_1984", "coinbase"):
        r = client.post(
            "/runs",
            json={
                "name": f"demo-{demo_id}",
                "goal": "brand_recall",
                "demo_id": demo_id,
            },
        )
        run_id = r.json()["run_id"]
        final = _poll_until_complete(run_id)
        scores[demo_id] = final["result"]["score"]
    # Distinct scores — no two equal.
    assert len(set(scores.values())) == 3, scores


def test_unknown_demo_id_marks_run_failed():
    r = client.post(
        "/runs",
        json={
            "name": "bogus",
            "goal": "brand_recall",
            "demo_id": "this_demo_does_not_exist",
        },
    )
    assert r.status_code == 202
    run_id = r.json()["run_id"]
    final = _poll_until_complete(run_id)
    assert final["status"] == "failed"
    assert "this_demo_does_not_exist" in (final.get("error") or "")


def test_get_demo_run_returns_full_plan():
    """The Compare page fetches each demo via GET /demos/{id} so it can
    render the canned SuggestionPlan side-by-side without spinning up
    fake runs."""
    r = client.get("/demos/lays")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["demo_id"] == "lays"
    assert body["plan"]["score"] == 78
    assert body["plan"]["status"] == "Strong"
    assert len(body["plan"]["suggestions"]) == 3


def test_get_demo_run_unknown_returns_404():
    r = client.get("/demos/this_demo_does_not_exist")
    assert r.status_code == 404
    assert "this_demo_does_not_exist" in r.json()["detail"]


def test_comparison_narrative_endpoint_returns_pair():
    """Compare page picks 2 of the 3 demos; backend serves a hand-written
    pairwise narrative. Static-route precedence is also verified — without
    it `/demos/comparison` would fall through to `/demos/{demo_id}` and
    try to look up `id="comparison"`."""
    r = client.get("/demos/comparison", params={"a": "lays", "b": "apple_1984"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["winner_demo_id"] == "lays"
    assert body["headline"]
    assert body["subhead"]
    # Pairwise — only the 2 chosen demos in takeaways.
    assert set(body["demo_takeaways"].keys()) == {"lays", "apple_1984"}


def test_comparison_narrative_is_order_independent():
    """`?a=lays&b=apple_1984` must match `?a=apple_1984&b=lays`."""
    r1 = client.get(
        "/demos/comparison", params={"a": "lays", "b": "apple_1984"}
    )
    r2 = client.get(
        "/demos/comparison", params={"a": "apple_1984", "b": "lays"}
    )
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json() == r2.json()


def test_comparison_narrative_all_three_pairs_authored():
    """All C(3,2) = 3 pairs of the demo set have a hand-written narrative."""
    pairs = [
        ("lays", "apple_1984"),
        ("lays", "coinbase"),
        ("apple_1984", "coinbase"),
    ]
    for a, b in pairs:
        r = client.get("/demos/comparison", params={"a": a, "b": b})
        assert r.status_code == 200, f"{a} vs {b}: {r.text}"
        body = r.json()
        assert set(body["demo_takeaways"].keys()) == {a, b}
        assert body["winner_demo_id"] in {a, b}


def test_comparison_narrative_rejects_same_id():
    r = client.get("/demos/comparison", params={"a": "lays", "b": "lays"})
    assert r.status_code == 400


def test_comparison_narrative_unknown_pair_returns_404():
    r = client.get(
        "/demos/comparison",
        params={"a": "lays", "b": "this_demo_does_not_exist"},
    )
    assert r.status_code == 404


def test_demo_plans_include_region_timeseries():
    """Sparkline chart on the Results screen needs a per-second curve
    for each of the 6 v2 regions, on every demo."""
    for demo_id in ("lays", "apple_1984", "coinbase"):
        r = client.get(f"/demos/{demo_id}")
        assert r.status_code == 200
        ts = r.json()["plan"]["region_timeseries"]
        assert ts is not None, f"{demo_id} missing region_timeseries"
        assert set(ts.keys()) == {
            "memory",
            "emotion",
            "attention",
            "language",
            "face",
            "reward",
        }
        for key, series in ts.items():
            assert len(series) >= 30, f"{demo_id}/{key} too short ({len(series)})"
            assert all(0 <= v <= 100 for v in series), f"{demo_id}/{key} out of range"


def test_real_run_flow_unchanged_when_demo_id_unset():
    """Sanity — POST /runs without demo_id still goes through the real
    mock pipeline and completes successfully."""
    r = client.post(
        "/runs",
        json={"name": "real-mock-test", "goal": "brand_recall"},
    )
    assert r.status_code == 202
    run_id = r.json()["run_id"]
    final = _poll_until_complete(run_id)
    assert final["status"] == "complete"
    # Real mock plan has at least one suggestion and a numeric score.
    plan = final["result"]
    assert plan is not None
    assert isinstance(plan["score"], (int, float))
    assert len(plan["suggestions"]) >= 1
