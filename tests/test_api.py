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


# --- Post analysis (1-20 images + optional audio + optional caption) -----


def test_analyze_single_image_post_with_caption():
    r = client.post(
        "/analyze",
        json={
            "content_type": "post",
            "image_urls": ["https://example.com/post.jpg"],
            "caption": "Check out our new product, available now",
            "goal": "awareness",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["region_scores"].keys()) == set(REGIONS.keys())
    assert body["goal"] == "awareness"


def test_analyze_single_image_post_with_audio():
    r = client.post(
        "/analyze",
        json={
            "content_type": "post",
            "image_urls": ["https://example.com/post.jpg"],
            "audio_url": "https://example.com/voiceover.mp3",
            "goal": "engagement",
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["goal"] == "engagement"


def test_analyze_carousel_post_with_caption():
    r = client.post(
        "/analyze",
        json={
            "content_type": "post",
            "image_urls": [
                "https://example.com/post1.jpg",
                "https://example.com/post2.jpg",
                "https://example.com/post3.jpg",
            ],
            "caption": "Three new colors, available now",
            "goal": "awareness",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["region_scores"].keys()) == set(REGIONS.keys())
    assert body["goal"] == "awareness"


def test_analyze_carousel_post_with_audio_and_caption():
    r = client.post(
        "/analyze",
        json={
            "content_type": "post",
            "image_urls": [
                "https://example.com/a.jpg",
                "https://example.com/b.jpg",
            ],
            "audio_url": "https://example.com/track.mp3",
            "caption": "Swipe to see all variants",
            "seconds_per_image": 3.0,
            "goal": "engagement",
        },
    )
    assert r.status_code == 200


def test_analyze_post_requires_image_urls():
    r = client.post(
        "/analyze",
        json={
            "content_type": "post",
            "caption": "no image here",
            "goal": "conversion",
        },
    )
    assert r.status_code == 422


def test_analyze_post_requires_at_least_one_image():
    r = client.post(
        "/analyze",
        json={
            "content_type": "post",
            "image_urls": [],
            "caption": "empty list",
            "goal": "conversion",
        },
    )
    assert r.status_code == 422


def test_analyze_post_rejects_too_many_images():
    r = client.post(
        "/analyze",
        json={
            "content_type": "post",
            "image_urls": [f"https://example.com/{i}.jpg" for i in range(21)],
            "caption": "too many",
            "goal": "conversion",
        },
    )
    assert r.status_code == 422


def test_analyze_post_requires_audio_or_caption():
    r = client.post(
        "/analyze",
        json={
            "content_type": "post",
            "image_urls": ["https://example.com/post.jpg"],
            "goal": "conversion",
        },
    )
    assert r.status_code == 422


def test_analyze_post_rejects_invalid_seconds_per_image():
    r = client.post(
        "/analyze",
        json={
            "content_type": "post",
            "image_urls": [
                "https://example.com/a.jpg",
                "https://example.com/b.jpg",
            ],
            "caption": "x",
            "seconds_per_image": 0.1,
            "goal": "conversion",
        },
    )
    assert r.status_code == 422


def test_analyze_video_still_requires_content_url():
    r = client.post(
        "/analyze",
        json={
            "content_type": "video",
            "goal": "conversion",
        },
    )
    assert r.status_code == 422
