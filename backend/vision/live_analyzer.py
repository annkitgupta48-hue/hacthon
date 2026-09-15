import cv2
import numpy as np
from collections import deque

from backend.vision.hand_detector import HandDetector
from backend.vision.face_detector import FaceExpressionDetector
from backend.fusion.multimodal_fusion import fuse_results


hand_detector = HandDetector()
face_detector = FaceExpressionDetector()
recent_predictions = deque(maxlen=4)


def _stable_result(result):
    recent_predictions.append(result)
    if len(recent_predictions) < 2:
        return result

    stable = {}
    for key in ("gesture", "expression"):
        values = [item[key]["name"] for item in recent_predictions]
        name = max(set(values), key=values.count)
        matching = [item[key]["confidence"] for item in recent_predictions if item[key]["name"] == name]
        stable[key] = {"name": name, "confidence": round(float(sum(matching) / len(matching)), 3)}
        if key == "gesture":
            stable[key]["movement"] = result[key].get("movement", {"direction": "still", "speed": 0.0, "moving": False})
            actions = [item[key].get("movement", {}).get("action", "none") for item in recent_predictions]
            action = max(set(actions), key=actions.count)
            if action != "none":
                stable[key]["name"] = action
                stable[key]["confidence"] = max(stable[key]["confidence"], 0.9)
    return stable


def analyze_frame(frame_bgr):
    if frame_bgr is None:
        return {
            "gesture": {"name": "unknown", "confidence": 0.0},
            "expression": {"name": "unknown", "confidence": 0.0},
            "intent": {"name": "unknown", "label": "Unknown", "confidence": 0.0, "text": "Gesture unclear. Please try again."},
            "text": "Gesture unclear. Please try again.",
        }

    frame = np.asarray(frame_bgr)
    if frame.size == 0:
        return {
            "gesture": {"name": "unknown", "confidence": 0.0},
            "expression": {"name": "unknown", "confidence": 0.0},
            "intent": {"name": "unknown", "label": "Unknown", "confidence": 0.0, "text": "Gesture unclear. Please try again."},
            "text": "Gesture unclear. Please try again.",
        }

    hand_result = hand_detector.detect(frame)
    face_result = face_detector.detect(frame)
    stable = _stable_result({"gesture": hand_result, "expression": face_result})
    hand_result = stable["gesture"]
    face_result = stable["expression"]

    fused = fuse_results(hand_result, face_result, threshold=0.58)
    return {
        "gesture": hand_result,
        "expression": face_result,
        "intent": fused["intent"],
        "text": fused["text"],
    }


def decode_base64_image(data_url):
    if not data_url:
        return None
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    image_bytes = __import__("base64").b64decode(data_url)
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame
