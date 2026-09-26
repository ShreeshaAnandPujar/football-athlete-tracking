import sys
import time
import argparse
import json
import csv
from pathlib import Path
from collections import defaultdict, deque
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Add parent directory to sys.path for internal imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_tracked_video import render_video

MODEL_PATH = "runs/detect/outputs/training/yolov8s_soccernet_baseline/weights/best.pt"
TRACKER_YAML = "src/tracking/botsort_conf35_orb_match85.yaml"
DEFAULT_CONF = 0.35

def evaluate_video(video_path, output_dir=None, conf=DEFAULT_CONF, skip_if_done=True):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_name = video_path.stem
    if output_dir is None:
        out_dir = Path("outputs/evaluation/videos") / video_name
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracking_csv = out_dir / "tracking.csv"
    annotated_mp4 = out_dir / "annotated.mp4"
    track_stats_csv = out_dir / "track_statistics.csv"
    video_metrics_json = out_dir / "video_metrics.json"

    print("=" * 80)
    print(f"EVALUATING VIDEO: {video_name}")
    print(f"Path: {video_path}")
    print(f"Output Directory: {out_dir}")
    print("=" * 80)

    # 1. Run Tracking Inference
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps): fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0
    cap.release()

    if not (skip_if_done and tracking_csv.exists() and tracking_csv.stat().st_size > 5000):
        print("\n[1/3] Running Tracking Inference with Champion BoT-SORT...")
        model = YOLO(MODEL_PATH)
        t0 = time.time()
        
        with open(tracking_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "track_id", "class", "confidence", "x1", "y1", "x2", "y2"])
            
            results = model.track(
                source=str(video_path),
                tracker=TRACKER_YAML,
                persist=True,
                conf=conf,
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
                    print(f"  Processed {frame_idx}/{total_frames} frames ({time.time() - t0:.1f}s)...")
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
        print(f"Tracking complete in {elapsed:.1f}s ({total_frames/elapsed if elapsed > 0 else 0:.1f} FPS, {total_records} detections).")
    else:
        print("\n[1/3] Tracking CSV already exists. Skipping inference.")

    # 2. Render Annotated Video
    if not (skip_if_done and annotated_mp4.exists() and annotated_mp4.stat().st_size > 50000):
        print("\n[2/3] Rendering Annotated Video...")
        render_video(video_path, tracking_csv, annotated_mp4, trail_len=25)
    else:
        print("\n[2/3] Annotated video already exists. Skipping rendering.")

    # 3. Compute Comprehensive Analytics, track_statistics.csv, & video_metrics.json
    print("\n[3/3] Computing Tracking & Video Metrics...")
    df = pd.read_csv(tracking_csv)
    df["box_w"] = df["x2"] - df["x1"]
    df["box_h"] = df["y2"] - df["y1"]
    df["area_pct"] = (df["box_w"] * df["box_h"]) / (width * height) * 100.0

    # Per-track statistics
    track_stats = df.groupby("track_id").agg(
        first_frame=("frame", "min"),
        last_frame=("frame", "max"),
        detected_frames=("frame", "nunique"),
        mean_confidence=("confidence", "mean"),
        max_confidence=("confidence", "max"),
        mean_box_width=("box_w", "mean"),
        mean_box_height=("box_h", "mean"),
        max_area_pct=("area_pct", "max")
    ).reset_index()
    track_stats["span_frames"] = track_stats["last_frame"] - track_stats["first_frame"] + 1
    track_stats["continuity"] = round(track_stats["detected_frames"] / track_stats["span_frames"], 4)
    track_stats["duration_seconds"] = round(track_stats["detected_frames"] / fps, 2)
    track_stats.to_csv(track_stats_csv, index=False)
    print(f"  Saved track statistics to {track_stats_csv}")

    # Camera Cuts Detection
    cap = cv2.VideoCapture(str(video_path))
    cuts = []
    prev_hist = None
    f_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        f_idx += 1
        small = cv2.resize(frame, (320, 180))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        if prev_hist is not None:
            sim = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if sim < 0.45:
                cuts.append(f_idx)
        prev_hist = hist
    cap.release()

    # Cross-cut bleeds
    cut_bleeds = 0
    for c in cuts:
        before = set(df[df["frame"] == c - 1]["track_id"])
        after = set(df[df["frame"] == c]["track_id"])
        cut_bleeds += len(before.intersection(after))

    # Occlusions (IoU > 0.20)
    occlusion_frames = 0
    total_overlaps = 0
    for _, f_df in df.groupby("frame"):
        if len(f_df) < 2: continue
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
        if np.any(iou > 0.20):
            occlusion_frames += 1
            total_overlaps += np.sum(iou > 0.20) // 2

    # Edge births & deaths
    edge_margin = 35
    first_dets = df.sort_values("frame").groupby("track_id").first()
    last_dets = df.sort_values("frame").groupby("track_id").last()
    edge_births = int(((first_dets["x1"] <= edge_margin) | (first_dets["x2"] >= width - edge_margin) |
                       (first_dets["y1"] <= edge_margin) | (first_dets["y2"] >= height - edge_margin)).sum())
    edge_deaths = int(((last_dets["x1"] <= edge_margin) | (last_dets["x2"] >= width - edge_margin) |
                       (last_dets["y1"] <= edge_margin) | (last_dets["y2"] >= height - edge_margin)).sum())

    # Temporary gap recoveries
    gap_recoveries = 0
    for _, grp in df.groupby("track_id"):
        diffs = np.diff(grp["frame"].values)
        gap_recoveries += int(np.sum((diffs > 1) & (diffs <= 30)))

    per_frame_counts = df.groupby("frame")["track_id"].nunique()

    video_metrics = {
        "video_name": video_name,
        "video_path": str(video_path),
        "resolution": f"{width}x{height}",
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_seconds": round(duration_sec, 2),
        "total_detections": len(df),
        "unique_track_ids": int(df["track_id"].nunique()),
        "average_active_tracks_per_frame": round(float(per_frame_counts.mean()), 2),
        "median_active_tracks_per_frame": round(float(per_frame_counts.median()), 1),
        "max_active_tracks_per_frame": int(per_frame_counts.max()),
        "mean_track_duration_frames": round(float(track_stats["detected_frames"].mean()), 1),
        "mean_track_duration_seconds": round(float(track_stats["duration_seconds"].mean()), 2),
        "max_track_duration_frames": int(track_stats["detected_frames"].max()),
        "max_track_duration_seconds": round(float(track_stats["duration_seconds"].max()), 2),
        "short_tracks_le_1s": int((track_stats["duration_seconds"] <= 1.0).sum()),
        "sustained_tracks_gt_5s": int((track_stats["duration_seconds"] > 5.0).sum()),
        "mean_continuity": round(float(track_stats["continuity"].mean()), 4),
        "median_continuity": round(float(track_stats["continuity"].median()), 4),
        "camera_cuts_detected": len(cuts),
        "cross_cut_identity_bleeds": cut_bleeds,
        "occlusion_frames": occlusion_frames,
        "occlusion_frame_ratio": round(occlusion_frames / max(1, df["frame"].nunique()), 4),
        "pairwise_overlap_events": total_overlaps,
        "occlusion_gaps_bridged": gap_recoveries,
        "edge_births": edge_births,
        "edge_deaths": edge_deaths
    }

    def json_converter(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(video_metrics_json, "w") as f:
        json.dump(video_metrics, f, indent=2, default=json_converter)
    print(f"  Saved video metrics to {video_metrics_json}")

    print("\n" + "-" * 60)
    print(f"SUMMARY FOR {video_name}:")
    print(f"  Resolution: {video_metrics['resolution']} @ {video_metrics['fps']} FPS ({video_metrics['duration_seconds']}s)")
    print(f"  Tracks: {video_metrics['unique_track_ids']} unique IDs ({video_metrics['total_detections']} detections)")
    print(f"  Active Tracks/Frame: {video_metrics['average_active_tracks_per_frame']} (Peak: {video_metrics['max_active_tracks_per_frame']})")
    print(f"  Mean Longevity: {video_metrics['mean_track_duration_seconds']}s (Max: {video_metrics['max_track_duration_seconds']}s)")
    print(f"  Camera Cuts: {video_metrics['camera_cuts_detected']} (Cross-Cut Bleeds: {video_metrics['cross_cut_identity_bleeds']})")
    print(f"  Occlusion Gaps Bridged: {video_metrics['occlusion_gaps_bridged']}")
    print("-" * 60)

    return video_metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate video with champion tracker, generating standard 4 artifacts.")
    parser.add_argument("--video", required=True, help="Input video file path")
    parser.add_argument("--output_dir", default=None, help="Directory to store the 4 output artifacts")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Detector confidence threshold")
    parser.add_argument("--force", action="store_true", help="Force re-running even if artifacts exist")
    args = parser.parse_args()

    evaluate_video(args.video, output_dir=args.output_dir, conf=args.conf, skip_if_done=not args.force)

if __name__ == "__main__":
    main()
