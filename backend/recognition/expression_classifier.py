from backend.config import EXPRESSION_LABELS


def classify_expression(landmarks=None):
    if not landmarks:
        return {"name": "unknown", "label": EXPRESSION_LABELS["unknown"], "confidence": 0.0}

    if isinstance(landmarks, dict):
        expression_name = landmarks.get("expression") or landmarks.get("name")
        if expression_name:
            expression_name = expression_name.lower()
            if expression_name in EXPRESSION_LABELS:
                return {
                    "name": expression_name,
                    "label": EXPRESSION_LABELS[expression_name],
                    "confidence": float(landmarks.get("confidence", 0.85)),
                }

    return {"name": "unknown", "label": EXPRESSION_LABELS["unknown"], "confidence": 0.0}
