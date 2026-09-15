from backend.config import INTENT_LABELS


def determine_intent(gesture: str | None, expression: str | None, confidence: float | None = None):
    gesture = (gesture or "unknown").lower()
    expression = (expression or "unknown").lower()

    movement_intents = {
        "swipe_right": ("next", "Next", "Next, please."),
        "swipe_left": ("back", "Back", "Go back, please."),
        "raise_hand": ("request_attention", "Request Attention", "I need help."),
        "lower_hand": ("cancel_request", "Cancel Request", "Cancel that request."),
    }
    if gesture in movement_intents:
        name, label, text = movement_intents[gesture]
        return {
            "name": name,
            "label": label,
            "confidence": float(confidence or 0.9),
            "text": text,
        }

    if gesture == "thumbs_up":
        if expression in {"happy", "neutral"}:
            return {
                "name": "positive_confirmation",
                "label": INTENT_LABELS["positive_confirmation"],
                "confidence": float(confidence or 0.9),
                "text": "Yes, that's good!",
            }
    if gesture == "thumbs_down":
        return {
            "name": "negative_confirmation",
            "label": INTENT_LABELS["negative_confirmation"],
            "confidence": float(confidence or 0.88),
            "text": "No, not good.",
        }
    if gesture == "wave":
        return {
            "name": "hello",
            "label": INTENT_LABELS["hello"],
            "confidence": float(confidence or 0.86),
            "text": "Hello!",
        }
    if gesture == "open_palm":
        return {
            "name": "attention",
            "label": "Attention",
            "confidence": float(confidence or 0.8),
            "text": "Please look at me.",
        }
    if gesture == "ok":
        return {
            "name": "agreement",
            "label": "Agreement",
            "confidence": float(confidence or 0.82),
            "text": "Everything is okay.",
        }
    if gesture == "peace":
        return {
            "name": "peace_sign",
            "label": "Peace Sign",
            "confidence": float(confidence or 0.84),
            "text": "Peace and hello!",
        }
    if gesture == "stop":
        return {
            "name": "please_stop",
            "label": INTENT_LABELS["please_stop"],
            "confidence": float(confidence or 0.9),
            "text": "Please stop.",
        }
    if gesture == "pointing":
        return {
            "name": "look_there",
            "label": INTENT_LABELS["look_there"],
            "confidence": float(confidence or 0.82),
            "text": "Look there.",
        }
    if gesture == "peace":
        return {
            "name": "hello",
            "label": INTENT_LABELS["hello"],
            "confidence": float(confidence or 0.84),
            "text": "Hello!",
        }
    if expression == "happy":
        return {
            "name": "i_like_it",
            "label": INTENT_LABELS["i_like_it"],
            "confidence": float(confidence or 0.7),
            "text": "Yes, I am happy.",
        }
    if expression == "sad":
        return {
            "name": "negative_confirmation",
            "label": INTENT_LABELS["negative_confirmation"],
            "confidence": float(confidence or 0.72),
            "text": "No, I don't like it.",
        }

    return {
        "name": "unknown",
        "label": INTENT_LABELS["unknown"],
        "confidence": float(confidence or 0.4),
        "text": "Gesture unclear. Please try again.",
    }
