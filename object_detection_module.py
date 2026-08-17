import cv2
from collections import deque
from ultralytics import YOLO

class ObjectDetectorYOLO:
    def __init__(self, model_name="yolov8n.pt", buffer_size=6):
        self.model = YOLO(model_name)
        self.phone_class_id = 67
        self.detection_history = deque(maxlen=buffer_size)

    def process(self, frame):
        results = self.model(frame, verbose=False, conf=0.35)[0]
        raw_phone_detected = False

        for box in results.boxes:
            if int(box.cls[0]) == self.phone_class_id:
                raw_phone_detected = True
                break

        self.detection_history.append(raw_phone_detected)
        # تثبيت الاكتشاف لمنع الرمش والتقطيع
        return self.detection_history.count(True) >= 2
