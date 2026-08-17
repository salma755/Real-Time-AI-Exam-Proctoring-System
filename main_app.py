import cv2
import numpy as np

from head_pose_module import HeadPoseDetectorTask
from pose_module import PoseDetectorTask
from object_detection_module import ObjectDetectorYOLO
from suspicion_engine import SuspicionEngine

WINDOW_NAME = "Smart Exam Proctoring System"


def draw_proctoring_dashboard(frame, bbox, score, head_angles, hands_suspicious, phone_detected):
    h_frame, w_frame, _ = frame.shape
    overlay = frame.copy()

    # تحديد الألوان حسب مستوى المخاطرة
    if score < 30:
        accent_color = (0, 204, 102)  # أخضر
        status_text = "NORMAL"
    elif score < 65:
        accent_color = (0, 191, 255)  # أصفر/عنبري
        status_text = "SUSPICIOUS"
    else:
        accent_color = (60, 60, 240)  # أحمر
        status_text = "HIGH RISK"

    panel_bg = (20, 22, 28)
    text_white = (245, 245, 245)
    text_muted = (150, 155, 165)


    panel_w = 320
    cv2.rectangle(overlay, (w_frame - panel_w, 0), (w_frame, h_frame), panel_bg, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)


    cv2.line(frame, (w_frame - panel_w, 0), (w_frame - panel_w, h_frame), (45, 50, 60), 1)

    px = w_frame - panel_w + 25
    py = 40


    cv2.putText(frame, "PROCTORING ANALYTICS", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.55, accent_color, 2, cv2.LINE_AA)
    py += 15
    cv2.line(frame, (px, py), (w_frame - 25, py), (45, 50, 65), 1)

    # 1. كشف الرأس
    py += 35
    cv2.putText(frame, "HEAD POSE TRACKING", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_muted, 1, cv2.LINE_AA)
    py += 22
    if head_angles:
        pitch, yaw, roll = head_angles
        yaw_status = "Center" if abs(yaw) <= 32 else ("Left" if yaw > 32 else "Right")
        cv2.putText(frame, f"Orientation : {yaw_status}", (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.46, text_white, 1,
                    cv2.LINE_AA)
    else:
        cv2.putText(frame, "Face : Out of Frame", (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (70, 70, 240), 1,
                    cv2.LINE_AA)

    py += 18
    cv2.line(frame, (px, py), (w_frame - 25, py), (35, 40, 50), 1)

    # 2. كشف اليدين والجسم
    py += 30
    cv2.putText(frame, "POSE & HANDS DETECTION", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_muted, 1, cv2.LINE_AA)
    py += 22
    hand_str = "Concealed / Hidden" if hands_suspicious else "Visible"
    hand_col = (70, 70, 240) if hands_suspicious else (0, 204, 102)
    cv2.putText(frame, f"Hands State : {hand_str}", (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.46, hand_col, 1,
                cv2.LINE_AA)

    py += 18
    cv2.line(frame, (px, py), (w_frame - 25, py), (35, 40, 50), 1)

    # 3. كشف الهاتف (YOLO)
    py += 30
    cv2.putText(frame, "OBJECT DETECTION (YOLO)", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_muted, 1, cv2.LINE_AA)
    py += 22
    phone_str = "Mobile Phone Detected!" if phone_detected else "No Prohibited Objects"
    phone_col = (70, 70, 240) if phone_detected else (0, 204, 102)
    cv2.putText(frame, phone_str, (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.46, phone_col, 1, cv2.LINE_AA)

    py += 18
    cv2.line(frame, (px, py), (w_frame - 25, py), (35, 40, 50), 1)

    # 4. مؤشر السكور التراكمي
    py += 35
    cv2.putText(frame, "CUMULATIVE RISK INDEX", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_muted, 1, cv2.LINE_AA)
    py += 38

    cv2.putText(frame, f"{score}%", (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 1.3, accent_color, 3, cv2.LINE_AA)
    cv2.putText(frame, f"[{status_text}]", (px + 110, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, accent_color, 2,
                cv2.LINE_AA)

    # شريط التقدم
    py += 20
    bar_x, bar_y, bar_w, bar_h = px + 10, py, panel_w - 70, 10
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 45, 55), -1)
    fill_w = int((score / 100.0) * bar_w)
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), accent_color, -1)

    # إطار تحديد الوجه
    if bbox is not None:
        bx, by, bw, bh = bbox
        bx, by = max(0, bx - 15), max(0, by - 15)
        bw, bh = bw + 30, bh + 30

        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), accent_color, 2)
        cv2.putText(frame, f"Student #01 | Risk: {score}%", (bx, max(20, by - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, accent_color, 1, cv2.LINE_AA)

    # الشريط العلوي
    cv2.rectangle(frame, (0, 0), (w_frame - panel_w, 30), (15, 18, 22), -1)
    cv2.putText(frame, "LIVE SESSION PROCTORING", (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 185, 195), 1,
                cv2.LINE_AA)

    return frame


def main():
    cap = cv2.VideoCapture(0)

    head_detector = HeadPoseDetectorTask('face_landmarker.task')
    pose_detector = PoseDetectorTask('pose_landmarker.task')
    object_detector = ObjectDetectorYOLO('yolov8n.pt')
    suspicion_engine = SuspicionEngine()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    print("done!!")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        head_angles, bbox = head_detector.process(frame)
        face_detected = head_angles is not None
        hands_suspicious = pose_detector.process(frame)
        phone_detected = object_detector.process(frame)

        score = suspicion_engine.update(head_angles, hands_suspicious, phone_detected, face_detected)

        frame = draw_proctoring_dashboard(frame, bbox, score, head_angles, hands_suspicious, phone_detected)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
