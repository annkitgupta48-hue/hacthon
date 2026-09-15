DEFAULT_THRESHOLD = 0.65


def is_confident(confidence: float | None, threshold: float = DEFAULT_THRESHOLD) -> bool:
    if confidence is None:
        return False
    return float(confidence) >= float(threshold)
