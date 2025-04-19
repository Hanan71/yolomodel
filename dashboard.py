import cv2
import streamlit as st
import tempfile
from ultralytics import YOLO
from collections import deque
from PIL import Image
import numpy as np

# YOLO model
model = YOLO("yolov5s.pt")

# Tracker class
class Tracker:
    def __init__(self):
        self.id_count = 0
        self.tracks = {}

    def update(self, detections):
        updated_tracks = []
        for det in detections:
            matched = False
            for track_id, track in self.tracks.items():
                if self._iou(det, track) > 0.3:
                    updated_tracks.append([*det, track_id])
                    self.tracks[track_id] = det
                    matched = True
                    break
            if not matched:
                self.id_count += 1
                self.tracks[self.id_count] = det
                updated_tracks.append([*det, self.id_count])

        self.tracks = {
            track_id: track for track_id, track in self.tracks.items()
            if track_id in [t[-1] for t in updated_tracks]
        }
        return updated_tracks

    def _iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        inter_x1 = max(x1, x3)
        inter_y1 = max(y1, y3)
        inter_x2 = min(x2, x4)
        inter_y2 = min(y2, y4)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        return inter_area / (box1_area + box2_area - inter_area + 1e-5)

# تحميل أسماء الكلاسات
with open("coco.txt", "r") as f:
    class_list = f.read().strip().split("\n")

# Streamlit واجهة
st.set_page_config(layout="wide")
st.title("👀 People Detection and Counting with YOLO + Streamlit")
st.markdown("---")

video_file = st.file_uploader("ارفع فيديو للكشف عن الأشخاص", type=["mp4", "mov", "avi", "mkv"])
line_position = st.slider("حدد مكان خط العد (من الأعلى إلى الأسفل)", 100, 480, 383)
show_warning = st.checkbox("أظهر تنبيه إذا تعدى العدد 60")

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)
    tracker = Tracker()
    counter = deque(maxlen=1000)

    stframe = st.empty()
    people_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame, verbose=False)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = box.cls[0].cpu().numpy()
            if class_list[int(cls)] == "person":
                detections.append([int(x1), int(y1), int(x2), int(y2)])

        tracked_objects = tracker.update(detections)

        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if line_position - 4 < cy < line_position + 4 and obj_id not in counter:
                counter.append(obj_id)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        cv2.line(frame, (0, line_position), (1020, line_position), (0, 255, 0), 2)
        people_count = len(counter)

        cv2.putText(frame, f"People Count: {people_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if show_warning and people_count > 60:
            cv2.putText(frame, "Warning!!", (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    cap.release()
