from pydantic import BaseModel

GESTURE_LABELS = {
    "open_palm": "Open Palm",
    "closed_fist": "Closed Fist",
    "thumbs_up": "Thumbs Up",
    "thumbs_down": "Thumbs Down",
    "peace": "Peace / V Sign",
    "pointing": "Pointing",
    "ok": "OK Gesture",
    "wave": "Wave",
    "stop": "Stop Gesture",
    "both_hands_raised": "Both Hands Raised",
    "unknown": "Unknown",
}

EXPRESSION_LABELS = {
    "neutral": "Neutral",
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Angry",
    "surprised": "Surprised",
    "confused": "Confused",
    "unknown": "Unknown",
}

INTENT_LABELS = {
    "positive_confirmation": "Positive Confirmation",
    "negative_confirmation": "Negative Confirmation",
    "hello": "Hello",
    "goodbye": "Goodbye",
    "please_stop": "Please Stop",
    "look_there": "Look There",
    "i_like_it": "I Like It",
    "neutral": "Neutral",
    "unknown": "Unknown",
}

DEFAULT_CONFIDENCE_THRESHOLD = 0.65


class DetectionRequest(BaseModel):
    gesture: str | None = None
    expression: str | None = None
    confidence: float | None = None
    hand_data: dict | None = None
    face_data: dict | None = None
