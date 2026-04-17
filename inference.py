"""
Traffic Monitoring System — Main Inference Script
Usage:
    python run_inference.py --source video.mp4 --model best.pt
    python run_inference.py --source 0          # webcam
    python run_inference.py --source video.mp4 --model best.pt --signal-red
"""

import argparse
import cv2
import time
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

from detector import (
    TrackedVehicle, ViolationRecord,
    SpeedEstimator, SignalViolationDetector,
    NumberPlateReader, HelmetDetector,
    VehicleCounter, ViolationLogger,
    draw_vehicle, draw_ui,
)

# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Traffic Monitoring Inference")
    p.add_argument("--source",      default="0",            help="Video path or 0 for webcam")
    p.add_argument("--model",       default="best.pt",      help="YOLO model weights")
    p.add_argument("--plate-model", default=None,           help="Number plate YOLO model (optional)")
    p.add_argument("--helmet-model",default=None,           help="Helmet YOLO model (optional)")
    p.add_argument("--conf",        type=float, default=0.35, help="Detection confidence threshold")
    p.add_argument("--iou",         type=float, default=0.5,  help="NMS IOU threshold")
    p.add_argument("--pixels-per-m",type=float, default=8.0,  help="Calibration: pixels per metre")
    p.add_argument("--stop-line",   type=int,   default=None, help="Stop-line Y pixel (auto = 40% height)")
    p.add_argument("--count-line",  type=int,   default=None, help="Count-line Y pixel (auto = 60% height)")
    p.add_argument("--signal-red",  action="store_true",    help="Start with signal RED")
    p.add_argument("--save-video",  action="store_true",    help="Save annotated output video")
    p.add_argument("--output-dir",  default="outputs",      help="Directory for outputs")
    return p.parse_args()


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load models ──────────────────────────────────────────────────────────
    print("Loading YOLOv8 model …")
    model = YOLO(args.model)
    model.fuse()

    plate_reader  = NumberPlateReader(args.plate_model)
    helmet_det    = HelmetDetector(args.helmet_model)

    # ── Open video ───────────────────────────────────────────────────────────
    source = int(args.source) if args.source.isdigit() else args.source
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open source '{args.source}'")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height} @ {fps:.1f} fps")

    # ── Auto-set line positions ───────────────────────────────────────────────
    stop_line_y  = args.stop_line  or int(height * 0.40)
    count_line_y = args.count_line or int(height * 0.65)

    # ── Init components ───────────────────────────────────────────────────────
    speed_est   = SpeedEstimator(fps, pixels_per_meter=args.pixels_per_m)
    sig_det     = SignalViolationDetector(stop_line_y, signal_red=args.signal_red)
    counter     = VehicleCounter(count_line_y)

    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger      = ViolationLogger(str(output_dir / f"violations_{ts}.xlsx"))

    # ── Video writer ──────────────────────────────────────────────────────────
    writer = None
    if args.save_video:
        out_path = str(output_dir / f"annotated_{ts}.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # ── Track state ──────────────────────────────────────────────────────────
    vehicles: dict[int, TrackedVehicle] = {}
    frame_num   = 0
    signal_red  = args.signal_red

    print("\nRunning inference … Press 'q' to quit, 'r' to toggle signal RED/GREEN\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        t_start = time.time()

        # ── YOLOv8 tracking ──────────────────────────────────────────────────
        results = model.track(
            frame,
            imgsz       = 640,
            conf        = args.conf,
            iou         = args.iou,
            persist     = True,
            tracker     = "bytetrack.yaml",
            verbose     = False,
        )

        active_ids = set()

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            ids     = results[0].boxes.id.cpu().numpy().astype(int)

            for bbox, cls_id, track_id in zip(boxes, cls_ids, ids):
                cls_name = model.names[cls_id]
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)

                # Create or update tracked vehicle
                if track_id not in vehicles:
                    vehicles[track_id] = TrackedVehicle(
                        track_id   = track_id,
                        class_name = cls_name,
                        bbox       = bbox,
                        centroid   = (cx, cy),
                    )
                else:
                    vehicles[track_id].bbox     = bbox
                    vehicles[track_id].centroid = (cx, cy)
                    vehicles[track_id].last_seen = time.time()

                v = vehicles[track_id]
                active_ids.add(track_id)

                # Speed
                speed_est.update(v)

                # Counting
                counter.update(v)

                # Signal violation
                if sig_det.check(v):
                    ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Read plate for violator
                    v.plate_text = plate_reader.read(frame, v.bbox)

                    record = ViolationRecord(
                        timestamp   = ts_now,
                        track_id    = track_id,
                        vehicle_cls = cls_name,
                        plate_text  = v.plate_text,
                        speed_kmh   = v.speed_kmh,
                        violation   = "Signal Violation",
                    )
                    logger.log(record)
                    print(f"  ⚠ VIOLATION | ID:{track_id} | {cls_name} | "
                          f"Plate:{v.plate_text or 'N/A'} | {v.speed_kmh:.0f} km/h")

                # Helmet (every 10 frames to save CPU)
                if frame_num % 10 == 0:
                    v.has_helmet = helmet_det.detect(frame, v.bbox, cls_name)
                    if not v.has_helmet and not v.violated:
                        ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        v.plate_text = v.plate_text or plate_reader.read(frame, v.bbox)
                        record = ViolationRecord(
                            timestamp   = ts_now,
                            track_id    = track_id,
                            vehicle_cls = cls_name,
                            plate_text  = v.plate_text,
                            speed_kmh   = v.speed_kmh,
                            violation   = "No Helmet",
                        )
                        logger.log(record)

                draw_vehicle(frame, v, signal_red)

        # ── Draw UI overlay ───────────────────────────────────────────────────
        sig_det.set_signal(signal_red)
        draw_ui(frame, counter, signal_red, stop_line_y, count_line_y)

        # FPS display
        elapsed = time.time() - t_start
        fps_disp = 1.0 / (elapsed + 1e-6)
        cv2.putText(frame, f"FPS: {fps_disp:.1f}", (width - 100, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if writer:
            writer.write(frame)

        cv2.imshow("Traffic Monitoring System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            signal_red = not signal_red
            print(f"  Signal toggled → {'RED' if signal_red else 'GREEN'}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    logger.save()

    print(f"\nDone. Violations saved to: {logger.path}")
    print(f"Total vehicles counted: {counter.total}")
    for cls, cnt in counter.counts.items():
        print(f"  {cls}: {cnt}")


if __name__ == "__main__":
    main()
