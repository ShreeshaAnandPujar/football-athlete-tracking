"""
Dynamic Camera-Compensated Trajectory & Speed Generator.

Eliminates camera panning/zooming artifacts from player speed calculations
by composing homographies with ORB camera motion compensation:
    H_t = H_ref * M_{t -> ref}
and applying low-pass temporal filtering to remove bounding box jitter.
"""

import sys
import time
from pathlib import Path
from collections import defaultdict, deque
import cv2
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analytics.dynamic_pitch_mapper import DynamicPitchMapper
from src.analytics.kinematics import KinematicsAnalyzer


def generate_camera_compensated_trajectories(
    video_path: str,
    tracking_csv: str,
    output_csv: str,
    fps: float = None,
    smooth_window: int = 11,
):
    video_path = Path(video_path)
    tracking_csv = Path(tracking_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COMPUTING DYNAMIC CAMERA-COMPENSATED PITCH TRAJECTORIES & TRUE SPEEDS")
    print(f"Video: {video_path}")
    print(f"Tracking CSV: {tracking_csv}")
    print(f"Output: {output_csv}")
    print("=" * 80)

    # 1. Load tracking data
    df_track = pd.read_csv(tracking_csv)
    grouped = df_track.groupby("frame")

    # 2. Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    if vid_fps <= 0 or np.isnan(vid_fps):
        vid_fps = 29.97
    actual_fps = fps or vid_fps
    dt = 1.0 / actual_fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize dynamic pitch mapper
    box_homo = DynamicPitchMapper.get_liverpool_attacking_box_homography()
    dyn_mapper = DynamicPitchMapper(anchor_homography=box_homo, smoothing_alpha=0.35)

    records = []
    frame_idx = 0
    t0 = time.time()

    print(f"Streaming video frames and computing camera motion at {actual_fps:.2f} FPS...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Track camera motion across frames
        dyn_mapper.update_camera_motion(frame)

        if frame_idx in grouped.groups:
            frame_dets = grouped.get_group(frame_idx)
            for _, row in frame_dets.iterrows():
                tid = int(row["track_id"])
                x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
                conf = float(row["confidence"])
                x_pixel = (x1 + x2) / 2.0
                y_pixel = y2

                # Project player foot position with dynamic camera compensation
                x_pitch, y_pitch, in_bounds = dyn_mapper.project_player(tid, x_pixel, y_pixel)

                record = {
                    "frame": frame_idx,
                    "track_id": tid,
                    "confidence": conf,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "x_pixel": round(x_pixel, 2),
                    "y_pixel": round(y_pixel, 2),
                    "x_pitch": round(float(x_pitch), 2) if x_pitch is not None else np.nan,
                    "y_pitch": round(float(y_pitch), 2) if y_pitch is not None else np.nan,
                    "in_bounds": in_bounds,
                }
                if "team_id" in row:
                    record["team_id"] = row["team_id"]
                if "team_name" in row:
                    record["team_name"] = row["team_name"]

                records.append(record)

        if frame_idx % 500 == 0 or frame_idx == total_frames:
            elapsed = time.time() - t0
            cur_fps = frame_idx / elapsed if elapsed > 0 else 0
            print(f"  Processed {frame_idx}/{total_frames} frames ({cur_fps:.1f} FPS)...")

    cap.release()

    res_df = pd.DataFrame(records)
    print(f"\nExtracted {len(res_df)} dynamic camera-compensated records.")

    # 3. Compute true physical kinematics on pitch plane
    print(f"Calculating true physical speed with {smooth_window}-frame temporal filter...")
    res_df = res_df.sort_values(["track_id", "frame"]).reset_index(drop=True)

    # Smooth pitch coordinates to remove remaining detector bounding box jitter
    res_df["x_pitch_smooth"] = (
        res_df.groupby("track_id")["x_pitch"]
        .transform(lambda s: s.rolling(smooth_window, min_periods=1, center=True).mean())
        .round(2)
    )
    res_df["y_pitch_smooth"] = (
        res_df.groupby("track_id")["y_pitch"]
        .transform(lambda s: s.rolling(smooth_window, min_periods=1, center=True).mean())
        .round(2)
    )

    # Metric step distance
    res_df["frame_diff"] = res_df.groupby("track_id")["frame"].diff().fillna(1)
    res_df["dt_step"] = res_df["frame_diff"] * dt

    res_df["dx"] = res_df.groupby("track_id")["x_pitch_smooth"].diff().fillna(0.0)
    res_df["dy"] = res_df.groupby("track_id")["y_pitch_smooth"].diff().fillna(0.0)
    res_df["step_dist_m"] = np.sqrt(res_df["dx"] ** 2 + res_df["dy"] ** 2)

    # Biologically plausible speed threshold (Usain Bolt top sprint is 11.5 m/s = 41.4 km/h)
    max_dist_step = 11.5 * res_df["dt_step"]
    res_df["is_teleport"] = res_df["step_dist_m"] > (max_dist_step * 1.5)
    res_df["step_dist_m"] = np.where(res_df["is_teleport"], 0.0, res_df["step_dist_m"])

    # Instantaneous speed
    res_df["speed_mps_raw"] = np.where(res_df["is_teleport"], 0.0, res_df["step_dist_m"] / res_df["dt_step"])
    res_df["speed_mps"] = (
        res_df.groupby("track_id")["speed_mps_raw"]
        .transform(lambda s: s.rolling(smooth_window, min_periods=1).mean())
        .clip(0.0, 11.5)
        .round(2)
    )
    res_df["speed_kmh"] = (res_df["speed_mps"] * 3.6).round(1)

    # Cleanup temp columns
    res_df = res_df.drop(columns=["frame_diff", "dt_step", "dx", "dy", "is_teleport", "speed_mps_raw"])

    # Re-sort by frame
    res_df = res_df.sort_values(["frame", "track_id"]).reset_index(drop=True)
    res_df.to_csv(output_csv, index=False)
    print(f"Saved true camera-compensated trajectory dataset to {output_csv}")
    return res_df


if __name__ == "__main__":
    generate_camera_compensated_trajectories(
        video_path="data/videos/liverpool_highlights.mp4",
        tracking_csv="outputs/tracking/liverpool_tracking_with_analytics.csv",
        output_csv="outputs/tracking/liverpool_tracking_compensated.csv",
    )
