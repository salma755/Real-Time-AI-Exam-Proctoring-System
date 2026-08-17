# Real-Time-AI-Exam-Proctoring-System
An automated, computer vision-based proctoring system designed to monitor candidate behavior during online or computer-based examinations. 
The system integrates MediaPipe, YOLOv8

---

## Technical Stack & Algorithms

1.Core Language: Python 3.9+
2.Computer Vision: OpenCV (`cv2`)
3.Landmark Detection: MediaPipe Tasks API (`Face Landmarker`, `Pose Landmarker`)
4.Object Detection: Ultralytics YOLOv8 (`yolov8n.pt`)
5.Algorithms:
   1.Perspective-n-Point (`solvePnP`): Maps 2D facial landmarks to a 3D generic facial model for pose estimation.
   2.Exponential Weighted Moving Average (EWMA): For smooth risk score transitions.
   3.Temporal Buffering: Frame-history queues for detection stabilization.

---
## Key Features

1.3D Head Pose Estimation: Calculates Pitch, Yaw, and Roll angles using MediaPipe Face Landmarker and `solvePnP` to detect when a user looks away from the screen.
2.Posture & Hand Visibility Tracking: Monitors upper-body landmarks to detect concealed arms/hands below desk level using a flexible "benefit-of-the-doubt" logic.
3.Prohibited Object Detection: Detects mobile phones in real time using Ultralytics YOLOv8.
4.Weighted Cumulative Risk Engine: Aggregates multi-modal violations into a dynamic risk score ($0\% - 100\%$) smoothed via Exponential Moving Average (EMA) to prevent false positives.
5.Temporal Buffer Smoothing: Uses frame buffers (`deque`) to eliminate detection flickering caused by temporary lighting or motion blur.
6.HUD Dashboard UI: Displays live tracking telemetry, system status, and target bounding boxes inside a dark-mode HUD overlay.

---

## Project Structure

```text
├── main_app.py                 # Main execution script & HUD GUI dashboard
├── head_pose_module.py         # 3D Head pose estimation module
├── pose_module.py              # Body pose & hand concealment detector
├── object_detection_module.py  # YOLOv8 object scanner wrapper
├── suspicion_engine.py         # Weighted scoring & risk smoothing engine
├── face_landmarker.task        # MediaPipe Face Landmarker asset
├── pose_landmarker.task        # MediaPipe Pose Landmarker asset
└── README.md                   # Project documentation, and **OpenCV** to track head orientation, posture, and unauthorized objects (e.g., mobile phones) in real time.
---


## Technical Stack & Algorithms

1.Core Language: Python 3.9+
2.Computer Vision: OpenCV (`cv2`)
3.Landmark Detection: MediaPipe Tasks API (`Face Landmarker`, `Pose Landmarker`)
4.Object Detection: Ultralytics YOLOv8 (`yolov8n.pt`)
5.Algorithms:
  1.Perspective-n-Point (`solvePnP`): Maps 2D facial landmarks to a 3D generic facial model for pose estimation.
  2.Exponential Weighted Moving Average (EWMA): For smooth risk score transitions.
  3.Temporal Buffering: Frame-history queues for detection stabilization.

---

## Project Structure

```text
├── main_app.py                 # Main execution script & HUD GUI dashboard
├── head_pose_module.py         # 3D Head pose estimation module
├── pose_module.py              # Body pose & hand concealment detector
├── object_detection_module.py  # YOLOv8 object scanner wrapper
├── suspicion_engine.py         # Weighted scoring & risk smoothing engine
├── face_landmarker.task        # MediaPipe Face Landmarker asset
├── pose_landmarker.task        # MediaPipe Pose Landmarker asset
└── README.md                   # Project documentation
