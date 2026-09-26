import sys
from pathlib import Path
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_video import evaluate_video

VIDEOS = [
    "data/videos/penalty_box_crowded.mp4",
    "data/videos/camera_motion_counterattack.mp4",
    "data/videos/liverpool_highlights.mp4",
    "data/videos/barcelona_highlight.mp4",
    "data/videos/highlight.mp4",
]

def main():
    print("=" * 80)
    print("BATCH EVALUATION OF REAL-WORLD FOOTBALL VIDEOS")
    print("=" * 80)
    
    all_metrics = []
    for v_path in VIDEOS:
        p = Path(v_path)
        if not p.exists():
            print(f"Skipping {v_path} (file not found)")
            continue
        m = evaluate_video(p, skip_if_done=True)
        all_metrics.append(m)

    summary_df = pd.DataFrame(all_metrics)
    summary_cols = [
        "video_name", "resolution", "fps", "duration_seconds",
        "unique_track_ids", "average_active_tracks_per_frame",
        "mean_track_duration_seconds", "camera_cuts_detected",
        "cross_cut_identity_bleeds", "occlusion_gaps_bridged", "mean_continuity"
    ]
    summary_cols = [c for c in summary_cols if c in summary_df.columns]
    
    out_summary_csv = Path("outputs/evaluation/videos/all_videos_summary.csv")
    out_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_summary_csv, index=False)
    
    print("\n" + "=" * 95)
    print("ALL REAL-WORLD VIDEOS EVALUATION SUMMARY")
    print("=" * 95)
    print(summary_df[summary_cols].to_markdown(index=False))
    print("=" * 95)
    print(f"Saved master summary to {out_summary_csv}")

if __name__ == "__main__":
    main()
