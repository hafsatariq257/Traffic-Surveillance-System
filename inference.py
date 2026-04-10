import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import time
from utils.excel_logger import log_violation
 
reader = easyocr.Reader(['en'], gpu=False)
 
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
    1: "Bicycle",
}
 
PIXELS_PER_METER = 8.0
FPS = 30
 
class VehicleTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.vehicle_count = 0
        self.counted_ids = set()
 
    def update(self, detections):
        updated = {}
        for det in detections:
            cx, cy, cls, conf = det
            matched_id = None
            min_dist = 60
            for tid, tdata in self.tracks.items():
                dx = cx - tdata['cx']
                dy = cy - tdata['cy']
                dist = (dx**2 + dy**2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    matched_id = tid
            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1
                self.tracks[matched_id] = {'cx': cx, 'cy': cy, 'cls': cls, 'history': [], 'conf': conf}
            self.tracks[matched_id]['history'].append((cx, cy, time.time()))
            self.tracks[matched_id]['cx'] = cx
            self.tracks[matched_id]['cy'] = cy
            if len(self.tracks[matched_id]['history']) > 30:
                self.tracks[matched_id]['history'].pop(0)
            updated[matched_id] = self.tracks[matched_id]
        self.tracks = updated
        return self.tracks
 
    def get_speed(self, track_id):
        if track_id not in self.tracks:
            return 0
        history = self.tracks[track_id]['history']
        if len(history) < 2:
            return 0
        x1, y1, t1 = history[0]
        x2, y2, t2 = history[-1]
        pixel_dist = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
        time_diff = t2 - t1
        if time_diff == 0:
            return 0
        meters = pixel_dist / PIXELS_PER_METER
        speed_ms = meters / time_diff
        speed_kmh = speed_ms * 3.6
        return min(speed_kmh, 180)
 
    def count_vehicle(self, track_id, line_y, cy):
        if track_id not in self.counted_ids and cy > line_y:
            self.counted_ids.add(track_id)
            self.vehicle_count += 1
 
def detect_plate_text(frame, box):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    results = reader.readtext(gray)
    texts = [r[1] for r in results if r[2] > 0.2]
    return " ".join(texts).strip()
 
def process_frame(frame, model, tracker, stop_line_y, signal_red, frame_count):
    results = model(frame, verbose=False)[0]
    detections = []
    annotated = frame.copy()
    h, w = frame.shape[:2]
 
    cv2.line(annotated, (0, stop_line_y), (w, stop_line_y), (0, 0, 255), 2)
    cv2.putText(annotated, "STOP LINE", (10, stop_line_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
 
    signal_color = (0, 0, 255) if signal_red else (0, 255, 0)
    signal_text = "RED" if signal_red else "GREEN"
    cv2.circle(annotated, (w - 50, 50), 20, signal_color, -1)
    cv2.putText(annotated, signal_text, (w - 80, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, signal_color, 2)
 
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in VEHICLE_CLASSES:
            continue
        conf = float(box.conf[0])
        if conf < 0.35:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        detections.append((cx, cy, cls_id, conf))
 
    tracks = tracker.update(detections)
    violations_this_frame = []
 
    for tid, tdata in tracks.items():
        cx, cy = tdata['cx'], tdata['cy']
        cls_id = tdata['cls']
        cls_name = VEHICLE_CLASSES.get(cls_id, "Vehicle")
        speed = tracker.get_speed(tid)
        tracker.count_vehicle(tid, stop_line_y, cy)
 
        color = (0, 255, 0)
        violation_type = None
 
        if signal_red and cy > stop_line_y:
            color = (0, 0, 255)
            violation_type = "Signal Violation"
 
        for x1, y1, x2, y2 in [(tdata['cx']-40, tdata['cy']-40,
                                  tdata['cx']+40, tdata['cy']+40)]:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
 
        label = f"{cls_name} {speed:.0f}km/h"
        cv2.putText(annotated, label, (cx - 40, cy - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
 
        if violation_type and frame_count % 30 == 0:
            plate = detect_plate_text(frame, (cx-60, cy-60, cx+60, cy+60))
            log_violation(violation_type, plate, cls_name, speed)
            violations_this_frame.append({
                "type": violation_type,
                "plate": plate,
                "class": cls_name,
                "speed": round(speed, 1)
            })
            cv2.putText(annotated, "VIOLATION!", (cx - 40, cy - 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
    cv2.putText(annotated, f"Count: {tracker.vehicle_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
 
    return annotated, tracker.vehicle_count, violations_this_frame
 