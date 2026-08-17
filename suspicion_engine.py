import numpy as np


class SuspicionEngine:
    def __init__(self):
        self.current_score = 0.0

    def update(self, head_angles, hands_suspicious, phone_detected, face_detected):
        target_score = 0.0

        # 1. جمع نقاط المخالفات المكتشفة لحظياً
        if phone_detected:
            target_score += 50.0

        if not face_detected:
            target_score += 35.0
        else:
            pitch, yaw, roll = head_angles

            # التفات الرأس
            if abs(yaw) > 32:
                target_score += 25.0

            # النظر للأعلى/الأسفل
            if pitch < -22 or pitch > 28:
                target_score += 15.0

        if hands_suspicious:
            target_score += 20.0

        target_score = min(100.0, target_score)

     if target_score > self.current_score:
            self.current_score += (target_score - self.current_score) * 0.08
        else:
            self.current_score -= (self.current_score - target_score) * 0.05

        return int(np.clip(self.current_score, 0, 100))
