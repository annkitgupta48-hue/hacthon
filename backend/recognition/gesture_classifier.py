from backend.config import GESTURE_LABELS


def classify_gesture(landmarks=None):
    if not landmarks:
        return {"name": "unknown", "label": GESTURE_LABELS["unknown"], "confidence": 0.0}

    # Lightweight prototype: simple deterministic mapping from provided landmark metadata.
    # This is intentionally rule-based and extendable.
    if isinstance(landmarks, dict):
        gesture_name = landmarks.get("gesture") or landmarks.get("name")
        if gesture_name:
            gesture_name = gesture_name.lower()
            if gesture_name in GESTURE_LABELS:
                return {
                    "name": gesture_name,
                    "label": GESTURE_LABELS[gesture_name],
                    "confidence": float(landmarks.get("confidence", 0.85)),
                }

    return {"name": "unknown", "label": GESTURE_LABELS["unknown"], "confidence": 0.0}
