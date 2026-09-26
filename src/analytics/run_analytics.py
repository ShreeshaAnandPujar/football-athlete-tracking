"""
End-to-End Football Analytics Runner CLI.

Integrates:
1. Team Classification (Jersey clustering + majority voting)
2. Player Kinematics (Distance, speed, acceleration, sprint zones)
3. Pitch Occupancy Heatmaps & Dominance Maps
4. Team Shape & Formation Estimation
5. Structured JSON Match Analytics Report
"""

import argparse
import json
from pathlib import Path
import sys
import cv2
import pandas as pd

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analytics.dynamic_pitch_mapper import DynamicPitchMapper
from src.analytics.heatmaps import HeatmapGenerator
from src.analytics.kinematics import KinematicsAnalyzer
from src.analytics.pitch_mapper import PitchMapper
from src.analytics.team_classifier import TeamClassifier
from src.analytics.team_shape import TeamShapeAnalyzer


def run_full_analytics(
    video_path: str,
    tracking_csv: str,
    trajectories_csv: str = None,
    output_dir: str = "outputs/analytics",
    fps: float = 25.0,
    sample_stride: int = 5,
    match_name: str = None,
):
    video_path = Path(video_path)
    tracking_csv = Path(tracking_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_dir = out_dir / "heatmaps"
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    name = match_name or video_path.stem
    print("=" * 80)
    print(f"RUNNING COMPLETE FOOTBALL-AWARE TRACKING ANALYTICS: {name}")
    print(f"Video: {video_path}")
    print(f"Tracking Detections: {tracking_csv}")
    print(f"Output Directory: {out_dir}")
    print("=" * 80)

    # 1. Load Tracking Detections
    df_track = pd.read_csv(tracking_csv)
    print(f"[1/5] Loaded {len(df_track)} detections for {df_track['track_id'].nunique()} tracks.")

    # 2. Team Classification
    print("[2/5] Running Unsupervised Jersey Color Clustering & Team Classification...")
    classifier = TeamClassifier(n_teams=2, include_referee=True)
    features, track_ids, _ = classifier.extract_dataset_features(
        video_path, df_track, sample_stride=sample_stride, max_samples_per_track=40
    )
    classifier.fit(features, track_ids)
    print(f"  -> Classified {len(classifier.track_teams)} tracks into teams:")
    team_counts = pd.Series(classifier.track_teams).value_counts().to_dict()
    for tid, count in sorted(team_counts.items()):
        name_str = classifier.team_names.get(tid, f"Team {tid}")
        print(f"     * {name_str}: {count} tracks")

    # 3. Trajectory Data (Load or Compute)
    if trajectories_csv is not None and Path(trajectories_csv).exists():
        print(f"[3/5] Loading precomputed metric trajectories from {trajectories_csv}...")
        df_traj = pd.read_csv(trajectories_csv)
    else:
        print("[3/5] Computing metric pitch coordinates via pitch mapper...")
        # Check if video is Liverpool or generic
        if "liverpool" in name.lower():
            mapper = DynamicPitchMapper.get_liverpool_attacking_box_homography()
        else:
            mapper = PitchMapper.get_tactical_default(1920, 1080)

        df_traj = df_track.copy()
        df_traj["x_pixel"] = (df_traj["x1"] + df_traj["x2"]) / 2.0
        df_traj["y_pixel"] = df_traj["y2"]
        coords = mapper.batch_pixel_to_pitch(df_traj[["x_pixel", "y_pixel"]].values)
        df_traj["x_pitch"] = coords[:, 0]
        df_traj["y_pitch"] = coords[:, 1]
        df_traj["x_pitch_smooth"] = (
            df_traj.groupby("track_id")["x_pitch"]
            .transform(lambda s: s.rolling(5, min_periods=1).mean())
            .round(2)
        )
        df_traj["y_pitch_smooth"] = (
            df_traj.groupby("track_id")["y_pitch"]
            .transform(lambda s: s.rolling(5, min_periods=1).mean())
            .round(2)
        )

    # Annotate trajectory with team assignments
    df_traj = classifier.annotate_dataframe(df_traj)

    # 4. Player Movement & Kinematics
    print("[4/5] Computing Player Kinematics & Workload Metrics...")
    kin_analyzer = KinematicsAnalyzer(fps=fps)
    annotated_df = kin_analyzer.compute_frame_kinematics(df_traj)
    player_summary = kin_analyzer.summarize_player_kinematics(annotated_df)

    kin_csv = out_dir / f"{name}_player_kinematics.csv"
    player_summary.to_csv(kin_csv, index=False)
    print(f"  -> Saved player kinematics to {kin_csv}")

    annotated_traj_csv = out_dir / f"{name}_annotated_trajectories.csv"
    annotated_df.to_csv(annotated_traj_csv, index=False)
    print(f"  -> Saved annotated trajectory dataset to {annotated_traj_csv}")

    # 5. Team Shape & Formations
    print("[5/5] Analyzing Team Shape & Tactical Formations...")
    shape_analyzer = TeamShapeAnalyzer()
    shape_df = shape_analyzer.analyze_sequence_team_shapes(annotated_df)
    shape_csv = out_dir / f"{name}_team_shapes.csv"
    shape_df.to_csv(shape_csv, index=False)
    print(f"  -> Saved team shape metrics to {shape_csv}")

    formation_a = shape_analyzer.estimate_formation(annotated_df, team_id=0)
    formation_b = shape_analyzer.estimate_formation(annotated_df, team_id=1)
    print(f"  -> Estimated Formation Team A: {formation_a.get('formation', 'Unknown')} ({formation_a.get('line_str', '')})")
    print(f"  -> Estimated Formation Team B: {formation_b.get('formation', 'Unknown')} ({formation_b.get('line_str', '')})")

    # 6. Generate Pitch Occupancy Heatmaps
    print("Generating High-Resolution Tactical Pitch Heatmaps...")
    heatmap_gen = HeatmapGenerator()

    # Team A & B Occupancy
    map_a_path = heatmaps_dir / f"{name}_team_a_heatmap.png"
    map_b_path = heatmaps_dir / f"{name}_team_b_heatmap.png"
    dom_path = heatmaps_dir / f"{name}_dominance_map.png"

    try:
        heatmap_gen.generate_team_heatmap(annotated_df, team_id=0, team_name="Team A", output_path=map_a_path)
        print(f"  -> Saved Team A Heatmap: {map_a_path}")
    except Exception as e:
        print(f"  -> Warning Team A Heatmap: {e}")

    try:
        heatmap_gen.generate_team_heatmap(annotated_df, team_id=1, team_name="Team B", output_path=map_b_path)
        print(f"  -> Saved Team B Heatmap: {map_b_path}")
    except Exception as e:
        print(f"  -> Warning Team B Heatmap: {e}")

    try:
        heatmap_gen.generate_dominance_map(annotated_df, team_a_id=0, team_b_id=1, output_path=dom_path)
        print(f"  -> Saved Territorial Dominance Map: {dom_path}")
    except Exception as e:
        print(f"  -> Warning Dominance Map: {e}")

    # Top individual player heatmaps
    top_tracks = player_summary["track_id"].head(2).tolist()
    for tid in top_tracks:
        p_path = heatmaps_dir / f"{name}_player_{tid}_heatmap.png"
        try:
            heatmap_gen.generate_player_heatmap(annotated_df, track_id=tid, output_path=p_path)
            print(f"  -> Saved Player #{tid} Heatmap: {p_path}")
        except Exception as e:
            print(f"  -> Warning Player #{tid} Heatmap: {e}")

    # 7. Summary JSON Match Report
    team_a_players = player_summary[player_summary["team_id"] == 0]
    team_b_players = player_summary[player_summary["team_id"] == 1]

    shape_a = shape_df[shape_df["team_id"] == 0] if len(shape_df) > 0 else pd.DataFrame()
    shape_b = shape_df[shape_df["team_id"] == 1] if len(shape_df) > 0 else pd.DataFrame()

    fastest_player = player_summary.sort_values("max_speed_kmh", ascending=False).iloc[0]
    furthest_player = player_summary.sort_values("total_distance_m", ascending=False).iloc[0]

    report = {
        "match_name": name,
        "total_tracks": int(df_track["track_id"].nunique()),
        "total_detections": int(len(df_track)),
        "team_classification": {
            "team_a_tracks": int(len(team_a_players)),
            "team_b_tracks": int(len(team_b_players)),
            "referee_gk_tracks": int(len(player_summary[player_summary["team_id"] >= 2])),
        },
        "formations": {
            "team_a": formation_a,
            "team_b": formation_b,
        },
        "physical_workload": {
            "team_a_total_distance_m": round(float(team_a_players["total_distance_m"].sum()), 2),
            "team_b_total_distance_m": round(float(team_b_players["total_distance_m"].sum()), 2),
            "team_a_sprint_distance_m": round(float(team_a_players["sprint_distance_m"].sum()), 2),
            "team_b_sprint_distance_m": round(float(team_b_players["sprint_distance_m"].sum()), 2),
            "furthest_player": {
                "track_id": int(furthest_player["track_id"]),
                "distance_m": float(furthest_player["total_distance_m"]),
                "team_id": int(furthest_player["team_id"]),
            },
            "fastest_player": {
                "track_id": int(fastest_player["track_id"]),
                "top_speed_kmh": float(fastest_player["max_speed_kmh"]),
                "team_id": int(fastest_player["team_id"]),
            },
        },
        "tactical_geometry": {
            "team_a_avg_compactness_m2": round(float(shape_a["hull_area_m2"].mean()), 2) if len(shape_a) > 0 else 0.0,
            "team_b_avg_compactness_m2": round(float(shape_b["hull_area_m2"].mean()), 2) if len(shape_b) > 0 else 0.0,
            "team_a_avg_width_m": round(float(shape_a["width_m"].mean()), 2) if len(shape_a) > 0 else 0.0,
            "team_b_avg_width_m": round(float(shape_b["width_m"].mean()), 2) if len(shape_b) > 0 else 0.0,
            "team_a_avg_length_m": round(float(shape_a["length_m"].mean()), 2) if len(shape_a) > 0 else 0.0,
            "team_b_avg_length_m": round(float(shape_b["length_m"].mean()), 2) if len(shape_b) > 0 else 0.0,
        },
    }

    report_path = out_dir / f"{name}_analytics_summary.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[SUCCESS] Analytics summary report saved to {report_path}")
    print("=" * 80)
    return report


def main():
    parser = argparse.ArgumentParser(description="End-to-End Football Tracking Analytics")
    parser.add_argument("--video", required=True, help="Input video file path")
    parser.add_argument("--tracking", required=True, help="Input tracking CSV path")
    parser.add_argument("--trajectories", default=None, help="Optional precomputed trajectories CSV")
    parser.add_argument("--output-dir", default="outputs/analytics", help="Output directory")
    parser.add_argument("--fps", type=float, default=25.0, help="Frame rate")
    parser.add_argument("--stride", type=int, default=5, help="Sampling stride for color clustering")
    parser.add_argument("--name", default=None, help="Match name tag")
    args = parser.parse_args()

    run_full_analytics(
        video_path=args.video,
        tracking_csv=args.tracking,
        trajectories_csv=args.trajectories,
        output_dir=args.output_dir,
        fps=args.fps,
        sample_stride=args.stride,
        match_name=args.name,
    )


if __name__ == "__main__":
    main()
