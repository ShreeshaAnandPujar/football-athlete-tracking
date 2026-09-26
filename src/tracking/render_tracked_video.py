import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict, deque
import cv2
import numpy as np
import pandas as pd

def get_color(track_id):
    """Generate a consistent, vibrant BGR color for each track ID."""
    np.random.seed(int(track_id) * 31 + 17)
    hue = np.random.randint(0, 180)
    sat = np.random.randint(180, 255)
    val = np.random.randint(200, 255)
    hsv_pixel = np.uint8([[[hue, sat, val]]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2]))

def render_video(video_path, csv_path, output_path, trail_len=30):
    print("=" * 70)
    print("RENDERING TRACKED FOOTBALL VIDEO")
    print(f"Source Video: {video_path}")
    print(f"Tracking CSV: {csv_path}")
    print(f"Output Video: {output_path}")
    print("=" * 70)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Tracking Data
    print("Loading tracking CSV...", end="", flush=True)
    df = pd.read_csv(csv_path)
    print(f" loaded {len(df)} records across {df['frame'].nunique()} frames.")

    # Group tracks by frame
    frames_dict = defaultdict(list)
    for _, row in df.iterrows():
        frames_dict[int(row["frame"])].append({
            "id": int(row["track_id"]),
            "conf": float(row["confidence"]),
            "x1": int(round(row["x1"])),
            "y1": int(round(row["y1"])),
            "x2": int(round(row["x2"])),
            "y2": int(round(row["y2"]))
        })

    # 2. Open Video Capture & Writer
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 29.97
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    print(f"Video specs: {width}x{height} @ {fps:.2f} FPS ({total_frames} frames)")
    print("Rendering frames with bounding boxes, IDs, and trajectory trails...")

    trajectories = defaultdict(lambda: deque(maxlen=trail_len))
    t0 = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        detections = frames_dict.get(frame_idx, [])
        active_ids = set()

        # Update and draw trajectory trails first (behind bounding boxes)
        for det in detections:
            tid = det["id"]
            active_ids.add(tid)
            # Bottom center position (feet on pitch)
            feet_x = int((det["x1"] + det["x2"]) / 2)
            feet_y = det["y2"]
            trajectories[tid].append((feet_x, feet_y))

        # Draw trails
        for tid, points in list(trajectories.items()):
            if tid not in active_ids and len(points) > 0:
                # Fade out or pop oldest point if player not detected
                points.popleft()
                if len(points) == 0:
                    del trajectories[tid]
                    continue

            color = get_color(tid)
            pts_list = list(points)
            for i in range(1, len(pts_list)):
                alpha = i / len(pts_list)
                thickness = max(1, int(round(1 + 2 * alpha)))
                cv2.line(frame, pts_list[i - 1], pts_list[i], color, thickness)

        # Draw bounding boxes and labels
        for det in detections:
            tid = det["id"]
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            conf = det["conf"]
            color = get_color(tid)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Small bottom-center anchor point (foot location)
            feet_x = int((x1 + x2) / 2)
            cv2.circle(frame, (feet_x, y2), 3, color, -1)

            # Label badge
            label = f"#{tid}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y1 = max(0, y1 - th - 6)
            label_y2 = y1
            label_x2 = x1 + tw + 8

            # Background for label
            cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), color, -1)
            # Label text
            cv2.putText(
                frame,
                label,
                (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        # Top-left HUD / Dashboard
        hud_bg_x2 = 360
        hud_bg_y2 = 75
        sub_img = frame[10:hud_bg_y2, 10:hud_bg_x2]
        black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        res = cv2.addWeighted(sub_img, 0.35, black_rect, 0.65, 1.0)
        frame[10:hud_bg_y2, 10:hud_bg_x2] = res
        cv2.rectangle(frame, (10, 10), (hud_bg_x2, hud_bg_y2), (255, 255, 255), 1)

        sec = frame_idx / fps
        tot_sec = total_frames / fps
        hud_time = f"Frame: {frame_idx:04d}/{total_frames}  [{int(sec//60):02d}:{sec%60:04.1f} / {int(tot_sec//60):02d}:{tot_sec%60:04.1f}]"
        hud_stats = f"Active Tracks: {len(detections)}  |  Tracker: BoT-SORT"

        cv2.putText(frame, hud_time, (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, hud_stats, (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        out.write(frame)

        if frame_idx % 500 == 0 or frame_idx == total_frames:
            elapsed = time.time() - t0
            cur_fps = frame_idx / elapsed if elapsed > 0 else 0
            print(f"  Rendered {frame_idx}/{total_frames} frames ({cur_fps:.1f} FPS, elapsed: {elapsed:.1f}s)")

    cap.release()
    out.release()

    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print("VIDEO RENDERING COMPLETE")
    print(f"Output saved to: {output_path}")
    print(f"Size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"Total time: {total_time:.1f}s ({frame_idx/total_time:.1f} FPS)")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Render tracked football video with boxes and motion trails.")
    parser.add_argument("--video", default="data/videos/liverpool_highlights.mp4", help="Input video path")
    parser.add_argument("--csv", default="outputs/tracking/liverpool_tracking_champion.csv", help="Tracking CSV path")
    parser.add_argument("--output", default="outputs/videos/liverpool_tracked_champion.mp4", help="Output MP4 path")
    parser.add_argument("--trail", type=int, default=25, help="Length of trajectory trail")
    args = parser.parse_args()

    render_video(Path(args.video), Path(args.csv), Path(args.output), trail_len=args.trail)

if __name__ == "__main__":
    main()
