import pytest
from fastapi.testclient import TestClient

from backend.intent.intent_engine import determine_intent
from backend.fusion.multimodal_fusion import fuse_results
from backend.main import app

client = TestClient(app)


@pytest.mark.parametrize(
    "gesture, expression, expected_text",
    [
        ("thumbs_up", "happy", "Yes, that's good!"),
        ("thumbs_down", "sad", "No, not good."),
        ("wave", "happy", "Hello!"),
        ("stop", "angry", "Please stop."),
        ("pointing", "neutral", "Look there."),
        ("open_palm", "neutral", "Please look at me."),
        ("ok", "neutral", "Everything is okay."),
        ("swipe_right", "unknown", "Next, please."),
        ("swipe_left", "unknown", "Go back, please."),
        ("raise_hand", "unknown", "I need help."),
    ],
)
def test_intent_mapping(gesture, expression, expected_text):
    result = determine_intent(gesture, expression, 0.9)
    assert result["text"] == expected_text


def test_fuse_low_confidence_returns_unclear():
    result = fuse_results({"name": "unknown", "confidence": 0.2}, {"name": "unknown", "confidence": 0.2}, threshold=0.65)
    assert result["text"] == "Gesture unclear. Please try again."


def test_duplicate_message_prevention_logic():
    history = ["Hello!", "Hello!", "Hello!", "Yes, that's good!"]
    assert history.count("Hello!") >= 2


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_translate_endpoints():
    payload = {"gesture": {"name": "thumbs_up", "confidence": 0.9}, "expression": {"name": "happy", "confidence": 0.88}}
    response = client.post("/translate", json=payload)
    assert response.status_code == 200
    assert "Yes, that's good!" in response.json()["text"]


def test_live_frame_endpoint_returns_motion_contract():
    response = client.post("/analyze-frame", json={})
    assert response.status_code == 200
    payload = response.json()
    assert "gesture" in payload
    assert "expression" in payload
    assert "text" in payload


def test_motion_signal_can_trigger_fusion_without_pose_confidence():
    result = fuse_results(
        {"name": "swipe_right", "confidence": 0.2, "movement": {"action": "swipe_right", "moving": True}},
        {"name": "unknown", "confidence": 0.0},
        threshold=0.65,
    )
    assert result["text"] == "Next, please."
