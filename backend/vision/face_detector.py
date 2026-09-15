import cv2
import mediapipe as mp
import numpy as np


class FaceExpressionDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _expression_from_landmarks(self, landmarks):
        if landmarks is None:
            return "unknown", 0.0
        points = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])

        mouth_top = points[13]
        mouth_bottom = points[14]
        mouth_left = points[61]
        mouth_right = points[291]
        left_eye_top = points[159]
        left_eye_bottom = points[145]
        right_eye_top = points[386]
        right_eye_bottom = points[374]

        mouth_height = abs(mouth_top[1] - mouth_bottom[1])
        mouth_width = abs(mouth_left[0] - mouth_right[0])
        left_eye_height = abs(left_eye_top[1] - left_eye_bottom[1])
        right_eye_height = abs(right_eye_top[1] - right_eye_bottom[1])

        if mouth_height > 0.12 and mouth_width > 0.18:
            return "surprised", 0.82
        if mouth_height < 0.05 and left_eye_height < 0.03 and right_eye_height < 0.03:
            return "angry", 0.76
        if mouth_height < 0.06 and mouth_width < 0.1:
            return "sad", 0.75
        if mouth_height > 0.09 and mouth_width > 0.14:
            return "happy", 0.88
        if left_eye_height > 0.03 and right_eye_height > 0.03:
            return "neutral", 0.7
        return "unknown", 0.35

    def detect(self, frame):
        if frame is None:
            return {"name": "unknown", "confidence": 0.0}

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return {"name": "unknown", "confidence": 0.0}

        face = results.multi_face_landmarks[0]
        expression_name, confidence = self._expression_from_landmarks(face)
        return {"name": expression_name, "confidence": float(confidence)}


def detect_face(frame=None):
    detector = FaceExpressionDetector()
    return detector.detect(frame)
