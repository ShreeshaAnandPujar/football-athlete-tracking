import sys
import time
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd

VIDEO_PATH = Path("data/videos/liverpool_highlights.mp4")
CSV_PATH = Path("outputs/tracking/liverpool_tracking_champion.csv")
WIDTH = 1280
HEIGHT = 720
FPS = 29.97

def detect_camera_cuts(video_path, threshold=0.45):
    print("\n[1] DETECTING CAMERA CUTS...")
    cap = cv2.VideoCapture(str(video_path))
    cuts = []
    prev_hist = None
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        small = cv2.resize(frame, (320, 180))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        if prev_hist is not None:
            sim = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if sim < threshold:
                cuts.append((frame_idx, round(frame_idx / FPS, 2), sim))
                
        prev_hist = hist
        
    cap.release()
    print(f"  Detected {len(cuts)} camera cuts across {frame_idx} frames.")
    return cuts

def run_deep_analysis():
    cuts_data = detect_camera_cuts(VIDEO_PATH)
    cut_frames = [c[0] for c in cuts_data]
    
    print("\n[2] LOADING TRACKING PREDICTIONS...")
    df = pd.read_csv(CSV_PATH)
    df["box_w"] = df["x2"] - df["x1"]
    df["box_h"] = df["y2"] - df["y1"]
    df["area_pct"] = (df["box_w"] * df["box_h"]) / (WIDTH * HEIGHT) * 100.0
    
    total_tracks = df["track_id"].nunique()
    total_frames = int(df["frame"].max())
    total_detections = len(df)
    
    # -------------------------------------------------------------
    # 1. Camera Cuts & Shot Breakdown
    # -------------------------------------------------------------
    shots = []
    prev_c = 1
    for c in cut_frames:
        shots.append((prev_c, c - 1, (c - prev_c) / FPS))
        prev_c = c
    shots.append((prev_c, total_frames, (total_frames - prev_c + 1) / FPS))
    
    # -------------------------------------------------------------
    # 2. Tracks Crossing Camera Cuts (Immediate bleed)
    # -------------------------------------------------------------
    direct_cut_crosses = []
    for c in cut_frames:
        before = set(df[df["frame"] == c - 1]["track_id"])
        after = set(df[df["frame"] == c]["track_id"])
        common = before.intersection(after)
        if len(common) > 0:
            direct_cut_crosses.append((c, common))
            
    # -------------------------------------------------------------
    # 3. False Identity Persistence After Cuts (Buffer Bleed within 30 frames)
    # -------------------------------------------------------------
    buffer_bleeds = []
    for c in cut_frames:
        # Tracks active within last 30 frames before cut
        before_window = set(df[(df["frame"] >= c - 30) & (df["frame"] < c)]["track_id"])
        # Tracks active within first 30 frames after cut
        after_window = set(df[(df["frame"] >= c) & (df["frame"] <= min(total_frames, c + 30))]["track_id"])
        common_buffer = before_window.intersection(after_window)
        if len(common_buffer) > 0:
            buffer_bleeds.append((c, common_buffer))
            
    # Total tracks spanning cut frames anytime during their span
    span_cut_tracks = set()
    for tid, grp in df.groupby("track_id"):
        fmin = grp["frame"].min()
        fmax = grp["frame"].max()
        for c in cut_frames:
            if fmin < c <= fmax:
                span_cut_tracks.add(tid)
                
    # -------------------------------------------------------------
    # 4. Edge Births & Deaths (Frame Boundary Interactions)
    # -------------------------------------------------------------
    edge_margin = 35
    first_dets = df.sort_values("frame").groupby("track_id").first()
    last_dets = df.sort_values("frame").groupby("track_id").last()
    
    edge_birth_mask = (
        (first_dets["x1"] <= edge_margin) |
        (first_dets["x2"] >= WIDTH - edge_margin) |
        (first_dets["y1"] <= edge_margin) |
        (first_dets["y2"] >= HEIGHT - edge_margin)
    )
    edge_births = edge_birth_mask.sum()
    
    edge_death_mask = (
        (last_dets["x1"] <= edge_margin) |
        (last_dets["x2"] >= WIDTH - edge_margin) |
        (last_dets["y1"] <= edge_margin) |
        (last_dets["y2"] >= HEIGHT - edge_margin)
    )
    edge_deaths = edge_death_mask.sum()
    
    # -------------------------------------------------------------
    # 5. Occlusion Frames & Overlap Pairs (IoU > 0.20 and IoU > 0.35)
    # -------------------------------------------------------------
    occlusion_frames_20 = 0
    occlusion_frames_35 = 0
    total_overlap_events = 0
    
    frames_grouped = df.groupby("frame")
    for f_idx, f_df in frames_grouped:
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
        
        has_20 = np.any(iou > 0.20)
        has_35 = np.any(iou > 0.35)
        if has_20:
            occlusion_frames_20 += 1
            total_overlap_events += np.sum(iou > 0.20) // 2
        if has_35:
            occlusion_frames_35 += 1

    # -------------------------------------------------------------
    # 6. Close-Up Tracks & Scaled Detections
    # -------------------------------------------------------------
    # Tactical pitch view player: ~0.15% to 0.6% of frame area
    # Medium close-up: > 1.5% area
    # Tight close-up: > 3.0% area
    med_close_up_dets = df[df["area_pct"] > 1.5]
    tight_close_up_dets = df[df["area_pct"] > 3.0]
    
    med_close_up_tracks = med_close_up_dets["track_id"].nunique()
    tight_close_up_tracks = tight_close_up_dets["track_id"].nunique()
    close_up_frames = med_close_up_dets["frame"].nunique()
    
    # -------------------------------------------------------------
    # 7. Track Continuity & Gap Analysis
    # -------------------------------------------------------------
    track_stats = df.groupby("track_id").agg(
        first_frame=("frame", "min"),
        last_frame=("frame", "max"),
        detected_frames=("frame", "nunique")
    )
    track_stats["span"] = track_stats["last_frame"] - track_stats["first_frame"] + 1
    track_stats["continuity"] = track_stats["detected_frames"] / track_stats["span"]
    
    # Re-association gap counts
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
            
    # -------------------------------------------------------------
    # 8. ID Behavior Around Cuts (Settle Time & Re-initialization)
    # -------------------------------------------------------------
    cut_dynamics = []
    for c, c_time, sim in cuts_data:
        pre_counts = [len(df[df["frame"] == f]) for f in range(max(1, c - 5), c)]
        post_counts = [len(df[df["frame"] == f]) for f in range(c, min(total_frames + 1, c + 6))]
        pre_mean = np.mean(pre_counts) if pre_counts else 0
        post_mean = np.mean(post_counts) if post_counts else 0
        
        # New IDs born in the 5 frames after cut
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
    
    # -------------------------------------------------------------
    # PRINT RESULTS
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("LIVERPOOL HIGHLIGHTS: COMPREHENSIVE SCENE & TRACKING DYNAMICS")
    print("=" * 80)
    
    print("\n[A] CAMERA CUTS & SHOT METRICS")
    print(f"  Total Camera Cuts Detected: {len(cuts_data)}")
    print(f"  Total Distinct Broadcast Shots: {len(shots)}")
    shot_durations = [s[2] for s in shots]
    print(f"  Mean Shot Duration: {np.mean(shot_durations):.2f}s (Min: {np.min(shot_durations):.2f}s, Max: {np.max(shot_durations):.2f}s)")
    
    print("\n[B] TRACKS CROSSING CAMERA CUTS & FALSE IDENTITY PERSISTENCE")
    print(f"  Direct Cut Frame Bleed (Tracks on frame c-1 AND frame c): {len(direct_cut_crosses)} ({len(direct_cut_crosses)/len(cuts_data)*100:.1f}%)")
    print(f"  Buffer Re-appearance Bleed (Track within 30f before cut reappearing within 30f after cut): {len(buffer_bleeds)} ({len(buffer_bleeds)/len(cuts_data)*100:.1f}%)")
    print(f"  Total Tracks Spanning Across Any Cut: {len(span_cut_tracks)} out of {total_tracks} ({len(span_cut_tracks)/total_tracks*100:.2f}%)")
    if len(direct_cut_crosses) == 0 and len(buffer_bleeds) == 0:
        print("  -> PERFECT CUT HYGIENE: 0.0% false identity bleed across camera cuts!")

    print("\n[C] EDGE BIRTHS & DEATHS (CANVAS BOUNDARIES)")
    print(f"  Edge Births (Track initiated within 35px of boundary): {edge_births} / {total_tracks} ({edge_births/total_tracks*100:.1f}%)")
    print(f"  In-Pitch Births (Track initiated inside pitch): {total_tracks - edge_births} / {total_tracks} ({(total_tracks - edge_births)/total_tracks*100:.1f}%)")
    print(f"  Edge Deaths (Track terminated within 35px of boundary): {edge_deaths} / {total_tracks} ({edge_deaths/total_tracks*100:.1f}%)")
    print(f"  In-Pitch Deaths (Terminated in field of play - occlusions/cuts): {total_tracks - edge_deaths} / {total_tracks} ({(total_tracks - edge_deaths)/total_tracks*100:.1f}%)")

    print("\n[D] OCCLUSION & OVERLAP DYNAMICS")
    print(f"  Frames with Player Overlaps (IoU > 0.20): {occlusion_frames_20} / {total_frames} ({occlusion_frames_20/total_frames*100:.1f}% of video)")
    print(f"  Frames with Severe Overlaps (IoU > 0.35): {occlusion_frames_35} / {total_frames} ({occlusion_frames_35/total_frames*100:.1f}% of video)")
    print(f"  Total Pairwise Overlap Events: {total_overlap_events}")

    print("\n[E] CLOSE-UP TRACKS & BROADCAST SCALE")
    print(f"  Close-Up Video Frames (>1.5% frame area): {close_up_frames} ({close_up_frames/total_frames*100:.1f}%)")
    print(f"  Players Tracked in Medium Close-Ups (>1.5% area): {med_close_up_tracks} tracks")
    print(f"  Players Tracked in Tight Close-Ups (>3.0% area): {tight_close_up_tracks} tracks")

    print("\n[F] TRACK CONTINUITY & OCCLUSION RECOVERY")
    print(f"  Mean Track Continuity (detected / span): {track_stats['continuity'].mean()*100:.1f}%")
    print(f"  Median Track Continuity: {track_stats['continuity'].median()*100:.1f}%")
    print(f"  Total Temporary Occlusion Gaps Bridged by Tracker: {total_gaps}")
    print(f"    - 1-Frame Drop (gap=2f): {gap_1} ({gap_1/total_gaps*100:.1f}%)")
    print(f"    - 2-5 Frames Drop: {gap_2_5} ({gap_2_5/total_gaps*100:.1f}%)")
    print(f"    - 6-15 Frames Drop: {gap_6_15} ({gap_6_15/total_gaps*100:.1f}%)")
    print(f"    - 16-30 Frames Drop: {gap_16_30} ({gap_16_30/total_gaps*100:.1f}%)")

    print("\n[G] ID BEHAVIOR AROUND FIRST 10 CAMERA CUTS SAMPLE")
    print(df_cuts_dyn.head(10).to_markdown(index=False))

    # Save detailed cuts table
    df_cuts_dyn.to_csv("outputs/tracking/liverpool_cuts_dynamics.csv", index=False)
    print("\nSaved camera cut dynamics to outputs/tracking/liverpool_cuts_dynamics.csv")

if __name__ == "__main__":
    run_deep_analysis()
