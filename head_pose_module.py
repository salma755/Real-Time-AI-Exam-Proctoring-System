import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HeadPoseDetectorTask:
    def __init__(self, model_path='face_landmarker.task'):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def process(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = self.detector.detect(mp_image)

        if not detection_result.face_landmarks:
            return None, None

        face_landmarks = detection_result.face_landmarks[0]

        # نقاط مرجعية بالـ 2D
        image_points = np.array([
            (face_landmarks[1].x * w, face_landmarks[1].y * h),     # الأنف
            (face_landmarks[152].x * w, face_landmarks[152].y * h), # الذقن
            (face_landmarks[33].x * w, face_landmarks[33].y * h),   # عين يسرى
            (face_landmarks[263].x * w, face_landmarks[263].y * h), # عين يمنى
            (face_landmarks[61].x * w, face_landmarks[61].y * h),   # فم أيسر
            (face_landmarks[291].x * w, face_landmarks[291].y * h)  # فم أيمن
        ], dtype="double")

        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ])

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch = euler_angles[0][0]
        yaw = euler_angles[1][0]
        roll = euler_angles[2][0]

        x_coords = [int(lm.x * w) for lm in face_landmarks]
        y_coords = [int(lm.y * h) for lm in face_landmarks]
        bbox = (min(x_coords), min(y_coords), max(x_coords) - min(x_coords), max(y_coords) - min(y_coords))

        return (pitch, yaw, roll), bbox
