def extract_landmarks(frame=None):
    """Return a minimal landmark structure for prototype use."""
    return {"landmarks": {}, "frame_shape": (480, 640, 3) if frame is not None else (0, 0, 0)}
