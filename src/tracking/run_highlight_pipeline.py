import sys
import time
import csv
from pathlib import Path
from collections import defaultdict, deque
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_tracked_video import render_video

VIDEO_PATH = Path("data/videos/highlight.mp4")
OUTPUT_TRACKING_DIR = Path("outputs/tracking")
OUTPUT_VIDEO_DIR = Path("outputs/videos")
OUTPUT_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "runs/detect/outputs/training/yolov8s_soccernet_baseline/weights/best.pt"
TRACKER_YAML = "src/tracking/botsort_conf35_orb_match85.yaml"
CSV_PATH = OUTPUT_TRACKING_DIR / "highlight_tracking_champion.csv"
OUTPUT_VIDEO_PATH = OUTPUT_VIDEO_DIR / "highlight_tracked_champion.mp4"
CONF = 0.35

# ==============================================================================
# STAGE 1: TRACKING INFERENCE
# ==============================================================================
def run_tracking():
    print("=" * 80)
    print("STAGE 1: RUNNING CHAMPION TRACKER ON HIGHLIGHT.MP4")
    print(f"Video: {VIDEO_PATH}")
    print(f"Model: {MODEL_PATH}")
    print(f"Tracker: {TRACKER_YAML} (conf={CONF}, BoT-SORT, ORB, match=0.85, buffer=30, ReID=False)")
    print(f"Output CSV: {CSV_PATH}")
    print("=" * 80)

    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 5000:
        print(f"Tracking CSV already exists ({CSV_PATH.stat().st_size} bytes). Skipping inference.")
        return

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
    print(f"\nTracking complete: {frame_idx} frames in {elapsed:.1f}s ({fps:.1f} FPS, {total_records} detections).")

# ==============================================================================
# STAGE 2: VIDEO RENDERING
# ==============================================================================
def render_output_video():
    print("\n" + "=" * 80)
    print("STAGE 2: RENDERING OUTPUT TRACKED VIDEO")
    print("=" * 80)
    render_video(VIDEO_PATH, CSV_PATH, OUTPUT_VIDEO_PATH, trail_len=25)

# ==============================================================================
# STAGE 3: SCENE DYNAMICS & CAMERA CUTS ANALYSIS
# ==============================================================================
def analyze_scene_and_tracking():
    print("\n" + "=" * 80)
    print("STAGE 3: COMPREHENSIVE SCENE & TRACKING DYNAMICS ANALYSIS")
    print("=" * 80)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps): fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video Info: {width}x{height} @ {fps:.2f} FPS ({total_frames} frames)")
    print("Detecting camera cuts via HSV correlation...")

    cuts = []
    prev_hist = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        small = cv2.resize(frame, (320, 180))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        if prev_hist is not None:
            sim = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if sim < 0.45:
                cuts.append((frame_idx, round(frame_idx / fps, 2), sim))

        prev_hist = hist
    cap.release()

    cut_frames = [c[0] for c in cuts]
    print(f"  Detected {len(cuts)} camera cuts across {frame_idx} frames.")

    df = pd.read_csv(CSV_PATH)
    df["box_w"] = df["x2"] - df["x1"]
    df["box_h"] = df["y2"] - df["y1"]
    df["area_pct"] = (df["box_w"] * df["box_h"]) / (width * height) * 100.0

    total_tracks = df["track_id"].nunique()
    total_dets = len(df)
    active_frames = df["frame"].nunique()

    # 1. Cuts and shots
    shots = []
    prev_c = 1
    for c in cut_frames:
        shots.append((prev_c, c - 1, (c - prev_c) / fps))
        prev_c = c
    shots.append((prev_c, total_frames, (total_frames - prev_c + 1) / fps))
    shot_durations = [s[2] for s in shots]

    # 2. Tracks crossing camera cuts (Immediate bleed)
    direct_cut_crosses = []
    for c in cut_frames:
        before = set(df[df["frame"] == c - 1]["track_id"])
        after = set(df[df["frame"] == c]["track_id"])
        common = before.intersection(after)
        if len(common) > 0:
            direct_cut_crosses.append((c, common))

    # 3. Buffer bleed (30 frames)
    buffer_bleeds = []
    for c in cut_frames:
        before_win = set(df[(df["frame"] >= c - 30) & (df["frame"] < c)]["track_id"])
        after_win = set(df[(df["frame"] >= c) & (df["frame"] <= min(total_frames, c + 30))]["track_id"])
        common_buf = before_win.intersection(after_win)
        if len(common_buf) > 0:
            buffer_bleeds.append((c, common_buf))

    span_cut_tracks = set()
    for tid, grp in df.groupby("track_id"):
        fmin = grp["frame"].min()
        fmax = grp["frame"].max()
        for c in cut_frames:
            if fmin < c <= fmax:
                span_cut_tracks.add(tid)

    # 4. Edge births & deaths (margins scaled for 1080p -> 45px)
    edge_margin = 45
    first_dets = df.sort_values("frame").groupby("track_id").first()
    last_dets = df.sort_values("frame").groupby("track_id").last()

    edge_births = (
        (first_dets["x1"] <= edge_margin) |
        (first_dets["x2"] >= width - edge_margin) |
        (first_dets["y1"] <= edge_margin) |
        (first_dets["y2"] >= height - edge_margin)
    ).sum()

    edge_deaths = (
        (last_dets["x1"] <= edge_margin) |
        (last_dets["x2"] >= width - edge_margin) |
        (last_dets["y1"] <= edge_margin) |
        (last_dets["y2"] >= height - edge_margin)
    ).sum()

    # 5. Occlusion & overlaps
    occlusion_20 = 0
    occlusion_35 = 0
    total_overlap_events = 0

    for f_idx, f_df in df.groupby("frame"):
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
            occlusion_20 += 1
            total_overlap_events += np.sum(iou > 0.20) // 2
        if np.any(iou > 0.35):
            occlusion_35 += 1

    # 6. Close-ups
    med_close_up_dets = df[df["area_pct"] > 1.5]
    tight_close_up_dets = df[df["area_pct"] > 3.0]
    med_close_tracks = med_close_up_dets["track_id"].nunique()
    tight_close_tracks = tight_close_up_dets["track_id"].nunique()
    close_up_frames = med_close_up_dets["frame"].nunique()

    # 7. Track continuity & gap recovery
    track_stats = df.groupby("track_id").agg(
        first_frame=("frame", "min"),
        last_frame=("frame", "max"),
        detected_frames=("frame", "nunique")
    )
    track_stats["span"] = track_stats["last_frame"] - track_stats["first_frame"] + 1
    track_stats["continuity"] = track_stats["detected_frames"] / track_stats["span"]

    gap_1 = 0
    gap_2_5 = 0
    gap_6_15 = 0
    gap_16_30 = 0
    total_gaps = 0

    for tid, grp in df.groupby("track_id"):
        frames = grp["frame"].values
        diffs = np.diff(frames)
        gaps = diffs[diffs > 1]
        for g in gaps:
            total_gaps += 1
            if g == 2: gap_1 += 1
            elif 3 <= g <= 6: gap_2_5 += 1
            elif 7 <= g <= 16: gap_6_15 += 1
            elif 17 <= g <= 31: gap_16_30 += 1

    # 8. Cut dynamics table
    cut_dynamics = []
    for c, c_time, sim in cuts:
        pre_counts = [len(df[df["frame"] == f]) for f in range(max(1, c - 5), c)]
        post_counts = [len(df[df["frame"] == f]) for f in range(c, min(total_frames + 1, c + 6))]
        pre_mean = np.mean(pre_counts) if pre_counts else 0
        post_mean = np.mean(post_counts) if post_counts else 0
        post_ids = df[df["frame"].between(c, min(total_frames, c + 5))]["track_id"].unique()
        born_after = sum(first_dets.loc[tid, "frame"] >= c for tid in post_ids)
        cut_dynamics.append({
            "cut_frame": c,
            "timestamp": f"{int(c_time//60):02d}:{c_time%60:04.1f}",
            "pre_cut_players": round(pre_mean, 1),
            "post_cut_players": round(post_mean, 1),
            "new_ids_initialized": born_after
        })
    df_cuts_dyn = pd.DataFrame(cut_dynamics)
    df_cuts_dyn.to_csv("outputs/tracking/highlight_cuts_dynamics.csv", index=False)

    # General Track Durations
    dur = track_stats["detected_frames"]
    per_frame = df.groupby("frame")["track_id"].nunique()

    print("\n" + "=" * 80)
    print("HIGHLIGHT.MP4 COMPLETE TRACKING & DYNAMICS REPORT")
    print("=" * 80)

    print("\n[A] OVERALL TRACKING METRICS")
    print(f"  Total Detections Logged: {total_dets}")
    print(f"  Unique Track IDs: {total_tracks}")
    print(f"  Average Active Tracks / Frame: {per_frame.mean():.2f}")
    print(f"  Median Active Tracks / Frame: {per_frame.median():.1f}")
    print(f"  Max Active Tracks in a Single Frame: {per_frame.max()}")
    print(f"  Mean Track Duration: {dur.mean():.1f} frames ({dur.mean()/fps:.2f}s)")
    print(f"  Median Track Duration: {dur.median():.1f} frames ({dur.median()/fps:.2f}s)")
    print(f"  Max Track Duration: {dur.max()} frames ({dur.max()/fps:.2f}s)")
    print(f"  Tracks <= 1s (<=30 frames): {(dur <= 30).sum()} ({(dur <= 30).sum() / len(dur) * 100:.1f}%)")
    print(f"  Tracks 1-5s (31-150 frames): {((dur > 30) & (dur <= 150)).sum()} ({((dur > 30) & (dur <= 150)).sum() / len(dur) * 100:.1f}%)")
    print(f"  Tracks > 5s (>150 frames): {(dur > 150).sum()} ({(dur > 150).sum() / len(dur) * 100:.1f}%)")

    print("\n[B] CAMERA CUTS & SHOT METRICS")
    print(f"  Total Camera Cuts Detected: {len(cuts)}")
    print(f"  Total Distinct Broadcast Shots: {len(shots)}")
    print(f"  Mean Shot Duration: {np.mean(shot_durations):.2f}s (Min: {np.min(shot_durations):.2f}s, Max: {np.max(shot_durations):.2f}s)")

    print("\n[C] TRACKS CROSSING CAMERA CUTS & FALSE IDENTITY PERSISTENCE")
    print(f"  Direct Cut Frame Bleed: {len(direct_cut_crosses)} ({len(direct_cut_crosses)/max(1, len(cuts))*100:.1f}%)")
    print(f"  Buffer Re-appearance Bleed (within 30f): {len(buffer_bleeds)} ({len(buffer_bleeds)/max(1, len(cuts))*100:.1f}%)")
    print(f"  Total Tracks Spanning Any Cut: {len(span_cut_tracks)} out of {total_tracks} ({len(span_cut_tracks)/total_tracks*100:.2f}%)")

    print("\n[D] EDGE BIRTHS & DEATHS")
    print(f"  Edge Births: {edge_births} / {total_tracks} ({edge_births/total_tracks*100:.1f}%)")
    print(f"  In-Pitch Births: {total_tracks - edge_births} / {total_tracks} ({(total_tracks - edge_births)/total_tracks*100:.1f}%)")
    print(f"  Edge Deaths: {edge_deaths} / {total_tracks} ({edge_deaths/total_tracks*100:.1f}%)")
    print(f"  In-Pitch Deaths: {total_tracks - edge_deaths} / {total_tracks} ({(total_tracks - edge_deaths)/total_tracks*100:.1f}%)")

    print("\n[E] OCCLUSION DYNAMICS")
    print(f"  Frames with Player Overlaps (IoU > 0.20): {occlusion_20} / {active_frames} ({occlusion_20/active_frames*100:.1f}%)")
    print(f"  Frames with Severe Overlaps (IoU > 0.35): {occlusion_35} / {active_frames} ({occlusion_35/active_frames*100:.1f}%)")
    print(f"  Total Pairwise Overlap Events: {total_overlap_events}")

    print("\n[F] CLOSE-UP TRACKS & BROADCAST SCALE")
    print(f"  Close-Up Video Frames (>1.5% frame area): {close_up_frames} ({close_up_frames/active_frames*100:.1f}%)")
    print(f"  Players in Medium Close-Ups (>1.5% area): {med_close_tracks} tracks")
    print(f"  Players in Tight Close-Ups (>3.0% area): {tight_close_tracks} tracks")

    print("\n[G] TRACK CONTINUITY & OCCLUSION RECOVERY")
    print(f"  Mean Track Continuity: {track_stats['continuity'].mean()*100:.1f}%")
    print(f"  Median Track Continuity: {track_stats['continuity'].median()*100:.1f}%")
    print(f"  Total Occlusion Gaps Bridged by Tracker: {total_gaps}")
    if total_gaps > 0:
        print(f"    - 1-Frame Drop: {gap_1} ({gap_1/total_gaps*100:.1f}%)")
        print(f"    - 2-5 Frames Drop: {gap_2_5} ({gap_2_5/total_gaps*100:.1f}%)")
        print(f"    - 6-15 Frames Drop: {gap_6_15} ({gap_6_15/total_gaps*100:.1f}%)")
        print(f"    - 16-30 Frames Drop: {gap_16_30} ({gap_16_30/total_gaps*100:.1f}%)")

    print("\n[H] ID BEHAVIOR AROUND FIRST 10 CAMERA CUTS")
    print(df_cuts_dyn.head(10).to_markdown(index=False))

def main():
    run_tracking()
    render_output_video()
    analyze_scene_and_tracking()

if __name__ == "__main__":
    main()
