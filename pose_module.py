import cv2
import mediapipe as mp
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseDetectorTask:
    def __init__(self, model_path='pose_landmarker.task', buffer_size=6):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=1
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.hands_history = deque(maxlen=buffer_size)

    def process(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self.detector.detect(mp_image)

        if not detection_result.pose_landmarks:
            raw_hands_suspicious = False
        else:
            landmarks = detection_result.pose_landmarks[0]

            # مواضع الكوع والكتف
            l_shoulder_y = landmarks[11].y * h
            r_shoulder_y = landmarks[12].y * h
            l_elbow_y = landmarks[13].y * h
            r_elbow_y = landmarks[14].y * h

            # اليد مشبوهة فقط إذا انخفضت الذراعان كلياً أسفل مستوى الطاولة/الكادر
            left_arm_down = l_elbow_y > (l_shoulder_y + h * 0.35) or l_elbow_y > h * 0.88
            right_arm_down = r_elbow_y > (r_shoulder_y + h * 0.35) or r_elbow_y > h * 0.88

            raw_hands_suspicious = left_arm_down and right_arm_down

        self.hands_history.append(raw_hands_suspicious)
        return self.hands_history.count(True) >= 3
