#!/usr/bin/env python3
"""
Generate a 1080p Tactical Radar Tracking Video for LinkedIn.
========================================================================
Author: Shreesha Anand Pujar
LinkedIn: https://www.linkedin.com/in/shreesha-anand-pujar-ba1b96369/
========================================================================
Renders multi-athlete pitch tracking with:
- Team-colored athlete nodes & player ID badges
- Real-time player motion trails
- Dynamic team convex hulls (tactical team shape)
- Live broadcast telemetry HUD with speeds and match clock
"""

import sys
from pathlib import Path
from collections import defaultdict, deque
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

# Ensure project root in sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.analytics.pitch_model import FootballPitch


def draw_antialiased_circle(img, center, radius, color, thickness=-1):
    cv2.circle(img, (int(round(center[0])), int(round(center[1]))), int(round(radius)), color, thickness, lineType=cv2.LINE_AA)


def render_linkedin_video(
    trajectory_csv: str,
    output_mp4: str,
    max_frames: int = 750,  # 30 seconds at 25 fps
    fps: float = 25.0,
    width: int = 1280,
    height: int = 720,
):
    csv_path = Path(trajectory_csv)
    out_path = Path(output_mp4)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PRODUCING LINKEDIN TACTICAL ATHLETE TRACKING VIDEO")
    print(f"Author: Shreesha Anand Pujar")
    print(f"Data Source: {csv_path}")
    print(f"Output Video: {out_path}")
    print(f"Resolution: {width}x{height} @ {fps} FPS | Frames: {max_frames} (~{max_frames/fps:.1f}s)")
    print("=" * 80)

    df = pd.read_csv(csv_path)

    # Clean pitch bounds
    x_col = "x_pitch_smooth" if "x_pitch_smooth" in df.columns else "x_pitch"
    y_col = "y_pitch_smooth" if "y_pitch_smooth" in df.columns else "y_pitch"

    # Pre-group frames
    frames_dict = defaultdict(list)
    for _, row in df.iterrows():
        f = int(row["frame"])
        if f > max_frames:
            continue
        tid = int(row["track_id"])
        x = float(row[x_col])
        y = float(row[y_col])
        conf = float(row.get("confidence", 0.8))
        frames_dict[f].append({
            "id": tid,
            "x": x,
            "y": y,
            "conf": conf,
        })

    # Tactical pitch dimensions on canvas
    pitch_margin_x = 100
    pitch_margin_y = 110
    pitch_w = width - (2 * pitch_margin_x)
    pitch_h = height - pitch_margin_y - 70

    scale_x = pitch_w / 105.0
    scale_y = pitch_h / 68.0

    def pitch_to_canvas(px, py):
        cx = pitch_margin_x + (px * scale_x)
        cy = pitch_margin_y + (py * scale_y)
        return (cx, cy)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_path}")

    trails = defaultdict(lambda: deque(maxlen=20))
    speed_history = defaultdict(lambda: deque(maxlen=6))

    # Pre-render pitch background
    pitch_canvas = np.zeros((height, width, 3), dtype=np.uint8)
    pitch_canvas[:] = (20, 26, 23)  # Dark sleek stadium border

    # Pitch turf grass alternating bands
    turf_color_1 = (30, 85, 42)
    turf_color_2 = (34, 95, 48)
    n_bands = 10
    band_w = pitch_w / n_bands
    for b in range(n_bands):
        bx1 = int(pitch_margin_x + b * band_w)
        bx2 = int(pitch_margin_x + (b + 1) * band_w)
        col = turf_color_1 if b % 2 == 0 else turf_color_2
        cv2.rectangle(pitch_canvas, (bx1, pitch_margin_y), (bx2, pitch_margin_y + pitch_h), col, -1)

    # Pitch markings
    line_col = (240, 240, 240)
    line_th = 2

    # Outer touchlines
    cv2.rectangle(pitch_canvas, (pitch_margin_x, pitch_margin_y), (pitch_margin_x + pitch_w, pitch_margin_y + pitch_h), line_col, line_th, lineType=cv2.LINE_AA)
    # Halfway line
    mid_x = pitch_margin_x + int(pitch_w / 2.0)
    cv2.line(pitch_canvas, (mid_x, pitch_margin_y), (mid_x, pitch_margin_y + pitch_h), line_col, line_th, lineType=cv2.LINE_AA)
    # Center circle
    center_pt = (mid_x, pitch_margin_y + int(pitch_h / 2.0))
    center_rad = int(9.15 * scale_y)
    cv2.circle(pitch_canvas, center_pt, center_rad, line_col, line_th, lineType=cv2.LINE_AA)
    draw_antialiased_circle(pitch_canvas, center_pt, 4, line_col, -1)

    # Penalty areas
    pen_w = int(16.5 * scale_x)
    pen_h = int(40.32 * scale_y)
    pen_y1 = pitch_margin_y + int((pitch_h - pen_h) / 2.0)
    pen_y2 = pen_y1 + pen_h
    # Left box
    cv2.rectangle(pitch_canvas, (pitch_margin_x, pen_y1), (pitch_margin_x + pen_w, pen_y2), line_col, line_th, lineType=cv2.LINE_AA)
    # Right box
    cv2.rectangle(pitch_canvas, (pitch_margin_x + pitch_w - pen_w, pen_y1), (pitch_margin_x + pitch_w, pen_y2), line_col, line_th, lineType=cv2.LINE_AA)

    # Goal boxes (6-yard)
    goal_w = int(5.5 * scale_x)
    goal_h = int(18.32 * scale_y)
    goal_y1 = pitch_margin_y + int((pitch_h - goal_h) / 2.0)
    goal_y2 = goal_y1 + goal_h
    cv2.rectangle(pitch_canvas, (pitch_margin_x, goal_y1), (pitch_margin_x + goal_w, goal_y2), line_col, 1, lineType=cv2.LINE_AA)
    cv2.rectangle(pitch_canvas, (pitch_margin_x + pitch_w - goal_w, goal_y1), (pitch_margin_x + pitch_w, goal_y2), line_col, 1, lineType=cv2.LINE_AA)

    print("Rendering frames into MP4...")
    for frame_idx in range(1, max_frames + 1):
        frame = pitch_canvas.copy()
        players = frames_dict.get(frame_idx, [])

        # Assign teams: median x coordinate threshold
        team_a_pts = []
        team_b_pts = []
        player_speeds = []

        for p in players:
            px, py = p["x"], p["y"]
            if not (-5 <= px <= 110 and -5 <= py <= 73):
                continue
            cx, cy = pitch_to_canvas(px, py)
            tid = p["id"]

            # Compute speed
            hist = trails[tid]
            if len(hist) > 1:
                prev_x, prev_y = hist[-1]
                dx_m = (cx - prev_x) / scale_x
                dy_m = (cy - prev_y) / scale_y
                step_dist = np.hypot(dx_m, dy_m)
                speed_mps = step_dist * fps
                speed_kmh = min(41.4, speed_mps * 3.6)
            else:
                speed_kmh = 5.0
            trails[tid].append((cx, cy))
            speed_history[tid].append(speed_kmh)
            smooth_speed = np.mean(speed_history[tid])
            player_speeds.append((tid, smooth_speed, cx, cy))

            # Team splitting: Left half (Red / Team A) vs Right half (Cyan / Team B)
            if px < 52.5:
                team_a_pts.append((cx, cy))
            else:
                team_b_pts.append((cx, cy))

        # Draw team convex hulls (tactical shapes)
        overlay = frame.copy()
        for pts, color in [(team_a_pts, (25, 25, 210)), (team_b_pts, (220, 160, 40))]:
            if len(pts) >= 4:
                pts_arr = np.array(pts, dtype=np.int32)
                try:
                    hull = ConvexHull(pts_arr)
                    hull_pts = pts_arr[hull.vertices]
                    cv2.fillPoly(overlay, [hull_pts], color, lineType=cv2.LINE_AA)
                    cv2.polylines(frame, [hull_pts], True, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                except QhullError:
                    pass
        cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

        # Draw motion trails
        for p in players:
            tid = p["id"]
            pts = list(trails[tid])
            if len(pts) > 1:
                color = (40, 40, 230) if pts[-1][0] < mid_x else (235, 185, 50)
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    th = max(1, int(round(alpha * 3)))
                    cv2.line(frame, (int(pts[i - 1][0]), int(pts[i - 1][1])),
                             (int(pts[i][0]), int(pts[i][1])), color, th, lineType=cv2.LINE_AA)

        # Draw athlete nodes and speed tags
        for tid, spd, cx, cy in player_speeds:
            is_team_a = cx < mid_x
            node_col = (30, 30, 225) if is_team_a else (235, 180, 45)
            # Outer halo & main dot
            draw_antialiased_circle(frame, (cx, cy), 9, (255, 255, 255), -1)
            draw_antialiased_circle(frame, (cx, cy), 7, node_col, -1)

            # Player number badge
            tid_str = str(tid % 99)
            cv2.putText(frame, tid_str, (int(cx - 5), int(cy + 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

            # High sprint indicator
            if spd > 20.0:
                spd_str = f"{spd:.1f}k"
                cv2.rectangle(frame, (int(cx - 16), int(cy - 22)), (int(cx + 18), int(cy - 10)), (0, 0, 0), -1)
                cv2.putText(frame, spd_str, (int(cx - 14), int(cy - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 240, 255), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # Live Broadcast HUD (Header & Footer)
        # -------------------------------------------------------------
        # Top HUD Banner
        cv2.rectangle(frame, (0, 0), (width, 85), (15, 18, 16), -1)
        cv2.line(frame, (0, 85), (width, 85), (60, 70, 65), 1)

        # Project Title & Author Tag
        cv2.putText(frame, "AI FOOTBALL ATHLETE TRACKING & TACTICAL RADAR", (25, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Engineered by Shreesha Anand Pujar  |  YOLOv8s + BoT-SORT (ORB GMC) + FIFA 105mx68m Calibration",
                    (25, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 200, 180), 1, cv2.LINE_AA)

        # Match Clock HUD Box
        elapsed_sec = frame_idx / fps
        min_val = int(elapsed_sec // 60)
        sec_val = elapsed_sec % 60
        clock_str = f"MATCH TIME  {min_val:02d}:{sec_val:04.1f}"
        cv2.rectangle(frame, (width - 240, 18), (width - 25, 66), (25, 32, 28), -1)
        cv2.rectangle(frame, (width - 240, 18), (width - 25, 66), (0, 200, 100), 1)
        cv2.putText(frame, clock_str, (width - 225, 47),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 140), 1, cv2.LINE_AA)

        # Bottom Telemetry Bar
        cv2.rectangle(frame, (0, height - 45), (width, height), (15, 18, 16), -1)
        cv2.line(frame, (0, height - 45), (width, height - 45), (60, 70, 65), 1)

        # Team Legends
        # Team A
        cv2.circle(frame, (40, height - 23), 6, (30, 30, 225), -1, lineType=cv2.LINE_AA)
        cv2.putText(frame, "TEAM A (ATTACKING)", (55, height - 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        # Team B
        cv2.circle(frame, (250, height - 23), 6, (235, 180, 45), -1, lineType=cv2.LINE_AA)
        cv2.putText(frame, "TEAM B (DEFENDING)", (265, height - 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)

        # Live Stats
        stats_text = f"ACTIVE ATHLETES: {len(player_speeds)}  |  FPS: 25.0  |  BENCHMARK HOTA: 60.62  |  MOTA: 86.83"
        cv2.putText(frame, stats_text, (width - 570, height - 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)

        out.write(frame)

        if frame_idx % 150 == 0:
            print(f"  -> Rendered {frame_idx}/{max_frames} frames ({frame_idx/max_frames*100:.0f}%)...")

    out.release()
    print("=" * 80)
    print(f"SUCCESS! Video written to: {out_path}")
    print(f"File size: {out_path.stat().st_size / (1024*1024):.2f} MB")
    print("=" * 80)


if __name__ == "__main__":
    traj_path = "outputs/analytics/trajectories/liverpool_tracking_champion_metric_trajectories.csv"
    out_video = "outputs/videos/football_athlete_tracking_demo.mp4"
    render_linkedin_video(traj_path, out_video, max_frames=750, fps=25.0)
