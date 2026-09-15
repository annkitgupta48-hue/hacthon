import cv2
import mediapipe as mp
import numpy as np
from collections import deque


class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.wrist_history = deque(maxlen=12)
        self.movement_history = deque(maxlen=12)

    def reset(self):
        self.wrist_history.clear()
        self.movement_history.clear()

    def _track_movement(self, landmarks):
        wrist = landmarks.landmark[0]
        current = np.array([wrist.x, wrist.y], dtype=np.float32)
        movement = np.zeros(2, dtype=np.float32)
        if self.wrist_history:
            movement = current - self.wrist_history[-1]
        self.wrist_history.append(current)
        self.movement_history.append(movement)
        displacement = float(np.linalg.norm(movement))
        horizontal_direction = "still"
        if displacement > 0.012:
            horizontal_direction = "right" if movement[0] > 0 else "left"
        return {
            "direction": horizontal_direction,
            "speed": round(displacement, 4),
            "moving": displacement > 0.012,
        }

    def _wave_detected(self):
        if len(self.movement_history) < 6:
            return False
        horizontal = [float(item[0]) for item in self.movement_history if abs(float(item[0])) > 0.012]
        if len(horizontal) < 4:
            return False
        signs = [1 if value > 0 else -1 for value in horizontal]
        changes = sum(previous != current for previous, current in zip(signs, signs[1:]))
        return changes >= 2

    def _movement_action(self):
        if len(self.wrist_history) < 5:
            return None
        start = self.wrist_history[0]
        end = self.wrist_history[-1]
        displacement = end - start
        distance = float(np.linalg.norm(displacement))
        if distance < 0.09:
            return None
        horizontal = abs(float(displacement[0]))
        vertical = abs(float(displacement[1]))
        if horizontal > vertical * 1.35:
            return "swipe_right" if displacement[0] > 0 else "swipe_left"
        if vertical > horizontal * 1.35:
            return "raise_hand" if displacement[1] < 0 else "lower_hand"
        return None

    def _gesture_from_landmarks(self, landmarks):
        if landmarks is None:
            return "unknown", 0.0

        points = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
        wrist = points[0]
        index_tip = points[8]
        middle_tip = points[12]
        ring_tip = points[16]
        pinky_tip = points[20]
        thumb_tip = points[4]
        thumb_ip = points[2]

        finger_open = [
            index_tip[1] < wrist[1],
            middle_tip[1] < wrist[1],
            ring_tip[1] < wrist[1],
            pinky_tip[1] < wrist[1],
        ]

        open_count = sum(finger_open)
        thumb_dir = thumb_tip[0] - thumb_ip[0]
        if open_count >= 3 and abs(thumb_dir) < 0.15:
            return "open_palm", 0.85
        if open_count <= 1 and thumb_dir > 0.2:
            return "thumbs_up", 0.88
        if open_count <= 1 and thumb_dir < -0.2:
            return "thumbs_down", 0.86
        if open_count >= 2 and abs(index_tip[0] - middle_tip[0]) < 0.12:
            return "peace", 0.82
        if abs(index_tip[0] - wrist[0]) > 0.25 and open_count <= 1:
            return "pointing", 0.8
        if open_count <= 1 and abs(thumb_tip[1] - index_tip[1]) < 0.08:
            return "ok", 0.8
        if open_count >= 2 and abs(pinky_tip[1] - wrist[1]) < 0.12:
            return "wave", 0.75
        if open_count <= 1 and abs(index_tip[1] - wrist[1]) < 0.1:
            return "stop", 0.85
        return "unknown", 0.4

    def detect(self, frame):
        if frame is None:
            return {"name": "unknown", "confidence": 0.0, "movement": {"direction": "still", "speed": 0.0, "moving": False}}

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if not results.multi_hand_landmarks:
            self.reset()
            return {"name": "unknown", "confidence": 0.0, "movement": {"direction": "still", "speed": 0.0, "moving": False}}

        hand = results.multi_hand_landmarks[0]
        gesture_name, confidence = self._gesture_from_landmarks(hand)
        movement = self._track_movement(hand)
        if self._wave_detected() and movement["moving"]:
            gesture_name, confidence = "wave", 0.94
        else:
            movement_action = self._movement_action()
            if movement_action:
                gesture_name, confidence = movement_action, 0.93
                movement["action"] = movement_action
            else:
                movement["action"] = "none"
        return {
            "name": gesture_name,
            "confidence": float(confidence),
            "movement": movement,
        }


def detect_hand(frame=None):
    detector = HandDetector()
    return detector.detect(frame)
