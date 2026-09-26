import sys
import time
import csv
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

VIDEO_PATH = Path("data/videos/barcelona_highlight.mp4")
OUTPUT_DIR = Path("outputs/tracking")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = "runs/detect/outputs/training/yolov8s_soccernet_baseline/weights/best.pt"

FINALISTS = [
    {
        "name": "Finalist_A_match80",
        "yaml": "src/tracking/botsort_conf35_orb.yaml",
        "conf": 0.35,
        "csv": OUTPUT_DIR / "barcelona_finalist_a_match80.csv",
        "desc": "conf=0.35, ORB GMC, match=0.80, ReID=False"
    },
    {
        "name": "Finalist_B_match85",
        "yaml": "src/tracking/botsort_conf35_orb_match85.yaml",
        "conf": 0.35,
        "csv": OUTPUT_DIR / "barcelona_finalist_b_match85.csv",
        "desc": "conf=0.35, ORB GMC, match=0.85, ReID=False"
    }
]

def run_tracking_on_video(cfg):
    csv_path = cfg["csv"]
    if csv_path.exists() and csv_path.stat().st_size > 5000:
        print(f"[{cfg['name']}] CSV already exists ({csv_path.stat().st_size} bytes). Skipping inference.")
        return

    print(f"\n{'='*70}\n[RUNNING TRACKER] {cfg['name']} ({cfg['desc']})\n{'='*70}")
    model = YOLO(MODEL_PATH)
    t0 = time.time()
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "class", "confidence", "x1", "y1", "x2", "y2"])
        
        results = model.track(
            source=str(VIDEO_PATH),
            tracker=cfg["yaml"],
            persist=True,
            conf=cfg["conf"],
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
                writer.writerow([frame_idx, tid, cls, round(float(c), 4),
                                 round(float(x1), 2), round(float(y1), 2),
                                 round(float(x2), 2), round(float(y2), 2)])
                total_records += 1
                
    elapsed = time.time() - t0
    fps = frame_idx / elapsed if elapsed > 0 else 0
    print(f"[{cfg['name']}] Finished {frame_idx} frames in {elapsed:.1f}s ({fps:.1f} FPS, {total_records} detections)")

def detect_camera_cuts_and_motion(video_path, threshold=0.45):
    """Detects broadcast camera cuts using frame-to-frame color histogram correlation."""
    print("\n[SCENE ANALYSIS] Detecting camera cuts and motion dynamics...")
    cap = cv2.VideoCapture(str(video_path))
    cuts = []
    motions = []
    
    prev_hsv = None
    prev_gray = None
    prev_hist = None
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        small = cv2.resize(frame, (320, 180))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        if prev_hist is not None:
            sim = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            diff = cv2.absdiff(gray, prev_gray)
            mean_diff = float(np.mean(diff))
            motions.append({"frame": frame_idx, "sim": sim, "diff": mean_diff})
            
            if sim < threshold:
                cuts.append(frame_idx)
                
        prev_hist = hist
        prev_gray = gray
        prev_hsv = hsv
        
    cap.release()
    print(f"  Detected {len(cuts)} camera cuts across {frame_idx} frames.")
    return cuts, pd.DataFrame(motions)

def analyze_tracker_behavior(csv_path, cuts, motions_df, total_frames=4491, fps=25.0):
    df = pd.read_csv(csv_path)
    df["box_w"] = df["x2"] - df["x1"]
    df["box_h"] = df["y2"] - df["y1"]
    df["area_ratio"] = (df["box_w"] * df["box_h"]) / (1280.0 * 720.0)
    
    # 1. Identity persistence
    track_stats = df.groupby("track_id").agg(
        first_frame=("frame", "min"),
        last_frame=("frame", "max"),
        duration=("frame", "nunique"),
        mean_conf=("confidence", "mean"),
        max_area=("area_ratio", "max"),
        mean_area=("area_ratio", "mean")
    ).reset_index()
    track_stats["span"] = track_stats["last_frame"] - track_stats["first_frame"] + 1
    track_stats["continuity"] = track_stats["duration"] / track_stats["span"]
    
    num_tracks = len(track_stats)
    short_tracks = (track_stats["duration"] <= 25).sum() # <= 1 sec
    medium_tracks = ((track_stats["duration"] > 25) & (track_stats["duration"] <= 125)).sum() # 1-5 sec
    long_tracks = (track_stats["duration"] > 125).sum() # > 5 sec
    ultra_long = (track_stats["duration"] > 250).sum() # > 10 sec
    
    # Active tracks per frame
    per_frame_counts = df.groupby("frame")["track_id"].nunique()
    mean_active = per_frame_counts.mean()
    median_active = per_frame_counts.median()
    
    # 2. Camera cuts behavior: how many tracks straddle a camera cut?
    straddling_cuts = 0
    tracks_spanning_cuts = []
    cut_set = set(cuts)
    for _, row in track_stats.iterrows():
        f_start = row["first_frame"]
        f_end = row["last_frame"]
        crossed = [c for c in cuts if f_start < c <= f_end]
        if crossed:
            straddling_cuts += 1
            tracks_spanning_cuts.append(row["track_id"])
            
    # 3. Close-up vs wide pitch behavior
    close_up_records = df[df["area_ratio"] > 0.035]
    close_up_tracks = close_up_records["track_id"].nunique()
    
    # 4. Occlusion analysis (high bbox overlap within same frame)
    occlusion_frames = 0
    sample_frames = sorted(df["frame"].unique())[::5]
    for f_idx in sample_frames:
        f_df = df[df["frame"] == f_idx]
        if len(f_df) < 2:
            continue
        boxes = f_df[["x1", "y1", "x2", "y2"]].values
        x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
        y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
        x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
        y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area[:, None] + area[None, :] - inter
        iou = inter / np.maximum(union, 1e-6)
        np.fill_diagonal(iou, 0)
        if np.any(iou > 0.25):
            occlusion_frames += 1
            
    # 5. Boundary entries and exits
    edge_margin = 30
    first_dets = df.sort_values("frame").groupby("track_id").first()
    last_dets = df.sort_values("frame").groupby("track_id").last()
    
    edge_births = (
        (first_dets["x1"] < edge_margin) | 
        (first_dets["x2"] > 1280 - edge_margin) | 
        (first_dets["y1"] < edge_margin) | 
        (first_dets["y2"] > 720 - edge_margin)
    ).sum()
    
    edge_deaths = (
        (last_dets["x1"] < edge_margin) | 
        (last_dets["x2"] > 1280 - edge_margin) | 
        (last_dets["y1"] < edge_margin) | 
        (last_dets["y2"] > 720 - edge_margin)
    ).sum()

    return {
        "total_detections": len(df),
        "unique_track_ids": num_tracks,
        "mean_active_per_frame": round(mean_active, 2),
        "median_active_per_frame": round(median_active, 1),
        "mean_duration_frames": round(track_stats["duration"].mean(), 1),
        "median_duration_frames": round(track_stats["duration"].median(), 1),
        "max_duration_frames": int(track_stats["duration"].max()),
        "short_tracks_pct": round(short_tracks / num_tracks * 100, 1),
        "medium_tracks_pct": round(medium_tracks / num_tracks * 100, 1),
        "long_tracks_pct": round(long_tracks / num_tracks * 100, 1),
        "ultra_long_tracks_pct": round(ultra_long / num_tracks * 100, 1),
        "straddling_cuts_count": straddling_cuts,
        "straddling_cuts_pct": round(straddling_cuts / num_tracks * 100, 1),
        "close_up_tracks": close_up_tracks,
        "edge_births_pct": round(edge_births / num_tracks * 100, 1),
        "edge_deaths_pct": round(edge_deaths / num_tracks * 100, 1),
        "occlusion_sampled_frames": occlusion_frames
    }

def main():
    cuts, motions_df = detect_camera_cuts_and_motion(VIDEO_PATH)
    
    results = []
    for cfg in FINALISTS:
        run_tracking_on_video(cfg)
        analysis = analyze_tracker_behavior(cfg["csv"], cuts, motions_df)
        analysis["tracker"] = cfg["name"]
        analysis["description"] = cfg["desc"]
        results.append(analysis)
        
    df_res = pd.DataFrame(results)
    print("\n" + "="*80)
    print("FINALISTS COMPARATIVE ANALYSIS ON BARCELONA HIGHLIGHT")
    print("="*80)
    display_cols = [
        "tracker", "unique_track_ids", "total_detections",
        "mean_duration_frames", "max_duration_frames",
        "short_tracks_pct", "long_tracks_pct", "ultra_long_tracks_pct",
        "straddling_cuts_count", "edge_births_pct"
    ]
    print(df_res[display_cols].to_markdown(index=False))
    
    df_res.to_csv("outputs/tracking/barcelona_finalists_comparison.csv", index=False)
    print("\nSaved detailed comparison to outputs/tracking/barcelona_finalists_comparison.csv")

if __name__ == "__main__":
    main()
