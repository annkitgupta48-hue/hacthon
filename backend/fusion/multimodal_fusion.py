from backend.intent.intent_engine import determine_intent


def fuse_results(gesture_result=None, expression_result=None, threshold=0.65):
    gesture_result = gesture_result or {"name": "unknown", "confidence": 0.0}
    expression_result = expression_result or {"name": "unknown", "confidence": 0.0}

    gesture_conf = float(gesture_result.get("confidence", 0.0))
    expression_conf = float(expression_result.get("confidence", 0.0))

    movement = gesture_result.get("movement") or {}
    has_motion_signal = movement.get("action") not in {None, "none"} or movement.get("moving") is True
    if gesture_conf < threshold and expression_conf < threshold and not has_motion_signal:
        return {
            "gesture": gesture_result,
            "expression": expression_result,
            "intent": {
                "name": "unknown",
                "label": "Unknown",
                "confidence": 0.0,
                "text": "Gesture unclear. Please try again.",
            },
            "text": "Gesture unclear. Please try again.",
        }

    intent = determine_intent(
        gesture_result.get("name"),
        expression_result.get("name"),
        max(gesture_conf, expression_conf),
    )

    return {
        "gesture": gesture_result,
        "expression": expression_result,
        "intent": intent,
        "text": intent["text"],
    }
