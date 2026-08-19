# 🎓 Real-Time AI Exam Proctoring System

<img width="762" height="552" alt="Screenshot 2026-08-18 010953" src="https://github.com/user-attachments/assets/ab692c68-5e02-46a2-bd6c-00f154e6a60b" />

An AI-powered, computer vision-based exam proctoring system designed to monitor candidate behavior during online or computer-based examinations.

The system combines **MediaPipe**, **YOLOv8**, and **OpenCV** to analyze facial orientation, body posture, hand visibility, and prohibited objects in real time. Detected behaviors are processed through a weighted risk engine to generate a smoothed **Cumulative Risk Index**.

> **Note:** The risk score is an automated monitoring indicator and should not be treated as a definitive judgment of academic misconduct. Unusual movements may occur for legitimate reasons and should be reviewed appropriately.

---

## 📌 Overview

Traditional online examination monitoring can require continuous human observation, which can be time-consuming and difficult to scale.

This project provides an automated computer vision pipeline capable of continuously analyzing the examination environment and identifying potentially suspicious events.

The system combines multiple visual signals rather than relying on a single detection method.

```text
      Camera Feed
           │
           ▼
 ┌─────────────────────┐
 │   OpenCV Capture    │
 └─────────┬───────────┘
           │
           ▼
┌─────────────────────────────┐
│   Computer Vision Modules   │
├─────────────────────────────┤
│ • Face Landmarker           │
│ • Pose Landmarker           │
│ • YOLOv8 Object Detection   │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│    Suspicion Engine         │
│                             │
│ Weighted Risk Calculation   │
│ + Temporal Buffering        │
│ + EMA Smoothing             │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│       HUD Dashboard         │
│                             │
│ • Risk Index                │
│ • Detection Status          │
│ • Tracking Information      │
│ • Bounding Boxes            │
└─────────────────────────────┘
```

---

#  Key Features

### 👤 3D Head Pose Estimation

The system estimates the candidate's head orientation using:

* Pitch
* Yaw
* Roll

Facial landmarks detected using the **MediaPipe Face Landmarker** are combined with OpenCV's `solvePnP` algorithm to estimate the 3D orientation of the head.

This allows the system to identify significant head movements away from the screen.

---

###  Posture & Hand Visibility Tracking

The **MediaPipe Pose Landmarker** is used to track upper-body landmarks.

The system analyzes the visibility and position of the candidate's arms and hands to identify situations where hands may be concealed below the desk.

The detection logic uses a flexible approach to reduce unnecessary alerts caused by temporary landmark loss or natural movement.

---

###  Prohibited Object Detection

The system uses **Ultralytics YOLOv8** to detect prohibited objects such as mobile phones.

The detected objects are processed in real time and incorporated into the overall risk calculation.

---

### 📊 Cumulative Risk Index

Multiple detection events are combined into a weighted risk score ranging from:

```text
0% ───────────────────────────── 100%
Normal                         High Risk
```

The score is smoothed using an **Exponential Moving Average (EMA)** to prevent sudden fluctuations caused by temporary detections.

---

### ⏱️ Temporal Buffering

Detection results are stored using frame-history buffers implemented with Python's `deque`.

This helps reduce flickering caused by:

* Temporary detection failures
* Motion blur
* Lighting changes
* Short-term landmark loss

Instead of reacting to a single frame, the system considers recent detection history.

---

###  HUD Dashboard

The application provides a real-time dark-mode HUD displaying:

* Current risk index
* Detection status
* Head pose information
* Object detection results
* Tracking information
* Bounding boxes
* System status

---

# 🧠 System Methodology

The system consists of four main computer vision components.

```text
                Camera Frame
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
   Face Landmarker        Pose Landmarker
          │                     │
          ▼                     ▼
     Head Pose              Body Pose
          │                     │
          └──────────┬──────────┘
                     │
                     ▼
              YOLOv8 Detector
                     │
                     ▼
              Detection Events
                     │
                     ▼
             Suspicion Engine
                     │
                     ▼
            Risk Calculation
                     │
                     ▼
                EMA Smoothing
                     │
                     ▼
              HUD Dashboard
```

---

#  Detection Pipeline

## 1. Face Detection & Head Pose

MediaPipe Face Landmarker extracts facial landmarks from the camera frame.

Selected 2D landmarks are mapped to a generic 3D facial model.

OpenCV's Perspective-n-Point algorithm estimates the camera-relative head orientation:

```text
2D Facial Landmarks
        +
3D Facial Model
        │
        ▼
     solvePnP
        │
        ▼
Rotation / Pose
        │
        ▼
Pitch / Yaw / Roll
```

The system uses configurable thresholds to identify significant head movements.

---

## 2. Body Pose Detection

MediaPipe Pose Landmarker tracks body landmarks such as:

* Shoulders
* Elbows
* Wrists
* Upper-body landmarks

These landmarks are used to determine whether hands are visible within the expected examination area.

---

## 3. Object Detection

YOLOv8 processes camera frames to detect objects of interest.

The primary prohibited object considered by the current system is:

```text
Mobile Phone
```

The YOLO detection results are converted into events that can contribute to the risk score.

---

# 📊 Risk Scoring

The system assigns different weights to detected events.

| Detection Event                      | Weight Contribution |
| ------------------------------------ | ------------------: |
| 📱 Mobile Phone Detected             |              +50%  |
| 👤 Face Missing / Out of Frame       |               +35% |
| ↔️ Significant Head Turn (Yaw > 32°) |               +25% |
| 👐 Hands Concealed Below Desk        |               +20% |
| ↕️ Sharp Pitch Angle (Up/Down > 22°) |               +15% |

The final risk score is constrained to the range:

```text
0% ≤ Risk Index ≤ 100%
```

Multiple simultaneous events can contribute to the score.

---

# 📉 Risk Smoothing

Raw detection results can fluctuate between consecutive frames.

To reduce this problem, the system uses:

### Temporal Buffering

Recent detection states are stored in frame-history queues.

```text
Frame 1 → Detected
Frame 2 → Detected
Frame 3 → Not Detected
Frame 4 → Detected
Frame 5 → Detected
```

Instead of immediately changing the system state, the recent history is considered.

### Exponential Moving Average

The risk value is smoothed using an EMA-based approach.

Conceptually:

```text
Smoothed Risk =
α × Current Risk
+
(1 − α) × Previous Risk
```

This produces more stable risk transitions and reduces false alarms caused by individual frames.

---

# 🛠️ Technical Stack

| Technology  | Purpose                            |
| ----------- | ---------------------------------- |
| Python      | Core programming language          |
| OpenCV      | Camera capture and computer vision |
| MediaPipe   | Face and body landmark detection   |
| YOLOv8      | Prohibited object detection        |
| Ultralytics | YOLO model implementation          |
| NumPy       | Numerical operations               |
| `deque`     | Temporal detection buffering       |

---

# 📁 Project Structure

```text
exam-proctoring-system/
│
├── main_app.py
│   └── Main application and HUD dashboard
│
├── head_pose_module.py
│   └── 3D head pose estimation
│
├── pose_module.py
│   └── Body pose and hand visibility detection
│
├── object_detection_module.py
│   └── YOLOv8 object detection wrapper
│
├── suspicion_engine.py
│   └── Risk calculation and smoothing
│
├── face_landmarker.task
│   └── MediaPipe Face Landmarker model
│
├── pose_landmarker.task
│   └── MediaPipe Pose Landmarker model
│
├── requirements.txt
│   └── Python dependencies
│
├── .gitignore
│   └── Git ignored files
│
└── README.md
    └── Project documentation
```

---

# ⚙️ Requirements

* Python **3.9+**
* Webcam
* MediaPipe
* OpenCV
* Ultralytics
* NumPy

A GPU is **not required** for basic operation, although hardware acceleration may improve real-time performance depending on the environment.

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone <repository_url>
cd exam-proctoring-system
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
pip install opencv-python mediapipe ultralytics numpy
```

---

#  Model Assets

The following MediaPipe model files are required:

```text
face_landmarker.task
pose_landmarker.task
```

Place both files in the root directory:

```text
exam-proctoring-system/
├── face_landmarker.task
├── pose_landmarker.task
└── ...
```

The YOLOv8 model is configured through Ultralytics.

If `yolov8n.pt` is not available locally, Ultralytics can download the required weights when the model is initialized, depending on the implementation.

---

#  Usage

Run the main application:

```bash
python main_app.py
```

The system will initialize the webcam and begin real-time monitoring.

### Exit

Press:

```text
q
```

or close the application window using the window's **X** button.

---

#  Real-Time Monitoring

During execution, the dashboard provides visual feedback about the current monitoring state.

The interface can display:

```text
┌─────────────────────────────────┐
│       AI EXAM PROCTORING        │
├─────────────────────────────────┤
│ Risk Index:        XX%          │
│ Face Status:       NORMAL       │
│ Head Pose:         NORMAL       │
│ Hands:             VISIBLE      │
│ Mobile Phone:     NOT DETECTED  │
│ System Status:     ACTIVE       │
└─────────────────────────────────┘
```

---

# ⚠️ Limitations

The current system operates using computer vision and therefore may be affected by environmental conditions.

Potential limitations include:

* Poor lighting
* Camera occlusion
* Low camera resolution
* Extreme head angles
* Temporary landmark detection failures
* Motion blur
* Partial body visibility
* False positives from natural candidate movements
* YOLO detection confidence variations

The system should therefore be considered an **automated assistance and monitoring tool**, rather than a standalone decision-making system.

---

#  Future Improvements

Possible future extensions include:

* Multiple-person detection
* Additional prohibited-object classes
* Improved hand tracking
* Adaptive risk thresholds
* Configurable scoring profiles
* Event logging and reporting
* Examination session reports
* Database integration
* Remote monitoring dashboard
* Model optimization for edge devices
* Improved temporal anomaly detection
* Candidate-specific calibration

---

# 🔐 Privacy Considerations

This system processes camera data for examination monitoring.

A production deployment should consider:

* User consent
* Secure data handling
* Minimal data retention
* Access control
* Appropriate privacy policies
* Local regulations regarding biometric and video data

The current project is primarily intended for **educational, research, and prototype purposes**.

---

# 📌 Project Status

**Status:** Prototype / Academic Project

The system demonstrates the integration of multiple computer vision techniques into a real-time AI-assisted examination monitoring pipeline.

---

#  Technologies & Concepts Demonstrated

This project demonstrates practical implementation of:

* Computer Vision
* Real-Time Object Detection
* Pose Estimation
* Facial Landmark Detection
* 3D Head Pose Estimation
* Perspective-n-Point (`solvePnP`)
* Deep Learning
* YOLOv8
* MediaPipe
* Temporal Smoothing
* Exponential Moving Average
* Multi-modal Risk Assessment

---

## 📄 License

This project is intended for educational and research purposes. Add an appropriate open-source license if you intend to distribute the code under specific reuse terms.
