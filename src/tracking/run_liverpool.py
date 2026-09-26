import sys
import time
import csv
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

VIDEO_PATH = Path("data/videos/liverpool_highlights.mp4")
OUTPUT_DIR = Path("outputs/tracking")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = "runs/detect/outputs/training/yolov8s_soccernet_baseline/weights/best.pt"
TRACKER_YAML = "src/tracking/botsort_conf35_orb_match85.yaml"
CSV_PATH = OUTPUT_DIR / "liverpool_tracking_champion.csv"
CONF = 0.35

def run_liverpool_tracking():
    print("=" * 70)
    print("RUNNING CHAMPION TRACKER ON LIVERPOOL HIGHLIGHTS")
    print(f"Video: {VIDEO_PATH}")
    print(f"Model: {MODEL_PATH}")
    print(f"Tracker: {TRACKER_YAML} (conf={CONF}, BoT-SORT, ORB, match=0.85, buffer=30, ReID=False)")
    print(f"Output CSV: {CSV_PATH}")
    print("=" * 70)

    model = YOLO(MODEL_PATH)
    t0 = time.time()
    
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "class", "confidence", "x1", "y1", "x2", "y2"])
        
        results = model.track(
            source=str(VIDEO_PATH),
            tracker=TRACKER_YAML,
            persist=True,
            conf=CONF,
            imgsz=640,
            device=0,
            stream=True,
            verbose=False
        )
        
        frame_idx = 0
        total_records = 0
        
        for res in results:
            frame_idx += 1
            if frame_idx % 500 == 0:
                print(f"  Processed {frame_idx} frames... ({time.time() - t0:.1f}s)")
                
            if res.boxes is None or res.boxes.id is None:
                continue
                
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.int().cpu().tolist()
            classes = res.boxes.cls.int().cpu().tolist()
            confs = res.boxes.conf.cpu().tolist()
            
            for b, tid, cls, c in zip(boxes, ids, classes, confs):
                x1, y1, x2, y2 = b
                writer.writerow([
                    frame_idx, tid, cls, round(float(c), 4),
                    round(float(x1), 2), round(float(y1), 2),
                    round(float(x2), 2), round(float(y2), 2)
                ])
                total_records += 1
                
    elapsed = time.time() - t0
    fps = frame_idx / elapsed if elapsed > 0 else 0
    print("\n" + "=" * 70)
    print("TRACKING COMPLETED")
    print("=" * 70)
    print(f"Total Frames Processed: {frame_idx}")
    print(f"Total Detections Logged: {total_records}")
    print(f"Elapsed Time: {elapsed:.1f}s ({fps:.1f} FPS)")
    print(f"CSV saved to: {CSV_PATH}")
    
    # Analysis
    analyze_tracking(CSV_PATH, frame_idx)

def analyze_tracking(csv_path, total_frames):
    df = pd.read_csv(csv_path)
    dur = df.groupby("track_id")["frame"].nunique()
    per_frame = df.groupby("frame")["track_id"].nunique()
    
    print("\n" + "=" * 70)
    print("LIVERPOOL TRACKING SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Unique Track IDs: {df['track_id'].nunique()}")
    print(f"Average Active Tracks / Frame: {per_frame.mean():.2f}")
    print(f"Median Active Tracks / Frame: {per_frame.median():.1f}")
    print(f"Max Active Tracks in a Single Frame: {per_frame.max()}")
    print(f"Mean Track Duration: {dur.mean():.1f} frames ({dur.mean()/29.97:.2f}s)")
    print(f"Median Track Duration: {dur.median():.1f} frames ({dur.median()/29.97:.2f}s)")
    print(f"Max Track Duration: {dur.max()} frames ({dur.max()/29.97:.2f}s)")
    print(f"Tracks <= 1s (<=30 frames): {(dur <= 30).sum()} ({(dur <= 30).sum() / len(dur) * 100:.1f}%)")
    print(f"Tracks 1-5s (31-150 frames): {((dur > 30) & (dur <= 150)).sum()} ({((dur > 30) & (dur <= 150)).sum() / len(dur) * 100:.1f}%)")
    print(f"Tracks > 5s (>150 frames): {(dur > 150).sum()} ({(dur > 150).sum() / len(dur) * 100:.1f}%)")

if __name__ == "__main__":
    run_liverpool_tracking()
