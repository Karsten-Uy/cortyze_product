"""Smoke tests for the FastAPI app.

Hits each endpoint via TestClient and verifies the full pipeline works
end-to-end with the mock RunPod client.
"""

from fastapi.testclient import TestClient

from api.main import app
from core.atlas.regions import REGIONS

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_analyze_returns_brain_report():
    r = client.post(
        "/analyze",
        json={
            "content_url": "https://example.com/test.mp4",
            "content_type": "video",
            "goal": "engagement",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "request_id" in body
    assert set(body["region_scores"].keys()) == set(REGIONS.keys())
    for score in body["region_scores"].values():
        assert 0.0 <= score <= 100.0
    assert 0.0 <= body["overall_score"] <= 100.0
    assert body["goal"] == "engagement"
    assert body["model_version"]
    assert body["elapsed_ms"] >= 0
    assert body["raw_predictions_uri"] is None


def test_analyze_request_id_threads_through():
    r = client.post(
        "/analyze",
        json={
            "content_url": "https://example.com/x.mp4",
            "content_type": "video",
            "goal": "conversion",
            "request_id": "abc-123",
        },
    )
    assert r.status_code == 200
    assert r.json()["request_id"] == "abc-123"


def test_analyze_rejects_invalid_goal():
    r = client.post(
        "/analyze",
        json={
            "content_url": "https://example.com/x.mp4",
            "content_type": "video",
            "goal": "not_a_real_goal",
        },
    )
    assert r.status_code == 422


def test_report_endpoint_stubbed():
    r = client.get("/report/some-id")
    assert r.status_code == 501


def test_upload_url_stubbed():
    r = client.post("/upload-url")
    assert r.status_code == 501


def test_cors_allowed_origin():
    r = client.get("/health", headers={"origin": "http://localhost:3000"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"
