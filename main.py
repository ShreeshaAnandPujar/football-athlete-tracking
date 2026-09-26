#!/usr/bin/env python3
"""
⚽ Football Athlete Tracking & Tactical Performance Analytics Pipeline
========================================================================
Lead Researcher & Engineer: Shreesha Anand Pujar
LinkedIn: https://www.linkedin.com/in/shreesha-anand-pujar-ba1b96369/
License: MIT License
========================================================================
Main CLI runner executing end-to-end athlete tracking analytics:
- 1. Self-Test Validation of Analytics Engine
- 2. Player Kinematics (Distance, Speed, Acceleration, Sprint Zones)
- 3. Team Tactical Geometry (Centroids, Convex Hulls, Formations)
- 4. 2D Gaussian Pitch Occupancy & Territorial Dominance Heatmaps
- 5. Benchmark Performance Summary
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Add repo root to Python path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.analytics.pitch_model import FootballPitch
from src.analytics.kinematics import KinematicsAnalyzer
from src.analytics.team_shape import TeamShapeAnalyzer
from src.analytics.heatmaps import HeatmapGenerator


def print_banner():
    banner = r"""
========================================================================================
⚽  FOOTBALL ATHLETE TRACKING & TACTICAL PERFORMANCE ANALYTICS PIPELINE
========================================================================================
  Lead Researcher & Engineer: Shreesha Anand Pujar
  LinkedIn: https://www.linkedin.com/in/shreesha-anand-pujar-ba1b96369/
  Architecture: YOLOv8s + BoT-SORT (ORB GMC) + FIFA Pitch Geometry Calibration
========================================================================================
"""
    print(banner)


def run_unit_tests():
    print("[1/5] Running Analytics Engine Unit & Integration Tests...")
    import tests.test_analytics as t_analytics

    t_analytics.test_jersey_color_extractor()
    t_analytics.test_team_classifier_clustering()
    t_analytics.test_kinematics_analyzer()
    t_analytics.test_heatmap_generator()
    t_analytics.test_team_shape_and_formation()
    print("      ✓ All analytics unit tests passed successfully!\n")


def run_kinematics_pipeline(df: pd.DataFrame, out_dir: Path):
    print("[2/5] Computing Metric Kinematics & Physical Workload...")
    analyzer = KinematicsAnalyzer(fps=25.0, pitch_length=105.0, pitch_width=68.0)
    
    # Compute step kinematics
    kin_df = analyzer.compute_frame_kinematics(df)
    workload_df = analyzer.summarize_player_kinematics(kin_df)

    out_csv = out_dir / "player_kinematics_summary.csv"
    workload_df.to_csv(out_csv, index=False)

    top_fastest = workload_df.sort_values(by="max_speed_kmh", ascending=False).head(5)
    top_distance = workload_df.sort_values(by="total_distance_m", ascending=False).head(5)

    print(f"      ✓ Processed {len(kin_df):,} trajectory positions for {len(workload_df)} athlete tracks.")
    print(f"      ✓ Top Distance Covered: Track #{top_distance.iloc[0]['track_id']} ({top_distance.iloc[0]['total_distance_m']:.2f} m)")
    print(f"      ✓ Peak Sprint Speed: Track #{top_fastest.iloc[0]['track_id']} ({top_fastest.iloc[0]['max_speed_kmh']:.2f} km/h)")
    print(f"      ✓ Saved Kinematics Summary to: {out_csv.relative_to(ROOT_DIR)}\n")
    return kin_df, workload_df


def run_tactical_geometry(df: pd.DataFrame, out_dir: Path):
    print("[3/5] Estimating Tactical Team Shape & Formations...")
    shape_analyzer = TeamShapeAnalyzer(pitch_length=105.0, pitch_width=68.0)

    # If team_id is missing, partition into two teams by longitudinal centroid for demonstration
    if "team_id" not in df.columns:
        track_medians = df.groupby("track_id")["x_pitch"].median()
        team_map = {tid: (0 if med < 52.5 else 1) for tid, med in track_medians.items()}
        df["team_id"] = df["track_id"].map(team_map)

    # Frame 100 tactical snapshot
    sample_frame = df[df["frame"] == 100]
    shape_a = shape_analyzer.compute_frame_shape_metrics(sample_frame, team_id=0)
    shape_b = shape_analyzer.compute_frame_shape_metrics(sample_frame, team_id=1)

    form_a = shape_analyzer.estimate_formation(df, team_id=0)
    form_b = shape_analyzer.estimate_formation(df, team_id=1)

    print(f"      ✓ Team A (Defending Half) - Estimated Formation: {form_a.get('formation', '4-3-3')}")
    if shape_a:
        print(f"        • Compactness (Hull Area): {shape_a['hull_area_m2']:.1f} m² | Width: {shape_a['width_m']:.1f} m | Length: {shape_a['length_m']:.1f} m")

    print(f"      ✓ Team B (Attacking Half) - Estimated Formation: {form_b.get('formation', '4-3-3')}")
    if shape_b:
        print(f"        • Compactness (Hull Area): {shape_b['hull_area_m2']:.1f} m² | Width: {shape_b['width_m']:.1f} m | Length: {shape_b['length_m']:.1f} m")
    print()


def run_pitch_heatmaps(df: pd.DataFrame, out_dir: Path):
    print("[4/5] Generating 2D Gaussian Kernel Density (KDE) Pitch Heatmaps...")
    generator = HeatmapGenerator(pitch_length=105.0, pitch_width=68.0, grid_resolution=1.0)
    heatmaps_dir = out_dir / "heatmaps"
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    # Team A & Team B heatmaps
    team_a_img = heatmaps_dir / "team_a_occupancy.png"
    team_b_img = heatmaps_dir / "team_b_occupancy.png"
    dom_img = heatmaps_dir / "territorial_dominance.png"

    df_a = df[df["team_id"] == 0]
    df_b = df[df["team_id"] == 1]

    generator.generate_team_heatmap(
        df, team_id=0, team_name="Team A (Liverpool FC)", cmap="Reds", output_path=team_a_img
    )
    generator.generate_team_heatmap(
        df, team_id=1, team_name="Team B (Opposition)", cmap="Blues", output_path=team_b_img
    )
    generator.generate_dominance_map(
        df, team_a_id=0, team_b_id=1,
        team_a_name="Team A", team_b_name="Team B",
        output_path=dom_img
    )

    print(f"      ✓ Team A Occupancy Map: {team_a_img.relative_to(ROOT_DIR)}")
    print(f"      ✓ Team B Occupancy Map: {team_b_img.relative_to(ROOT_DIR)}")
    print(f"      ✓ Territorial Dominance Map: {dom_img.relative_to(ROOT_DIR)}\n")


def display_benchmark_summary():
    print("[5/5] Official Benchmark Metrics (SoccerNet Tracking-2023 Validation):")
    bench_file = ROOT_DIR / "outputs" / "evaluation" / "frozen_benchmark.json"
    if bench_file.exists():
        with open(bench_file, "r") as f:
            bench = json.load(f)
        champ = bench.get("official_final_champion", {})
        metrics = champ.get("metrics", {})
        print(f"      ---------------------------------------------------------")
        print(f"      Tracker Architecture:   {champ.get('tracker_type')} (GMC: {champ.get('gmc_method')})")
        print(f"      Match IoU Threshold:    {champ.get('match_thresh')}")
        print(f"      Track Buffer:           {champ.get('track_buffer')} frames")
        print(f"      ---------------------------------------------------------")
        for k, v in metrics.items():
            print(f"      {k:<10}: {v:>8}")
        print(f"      ---------------------------------------------------------")
    print("\n🎉 Pipeline execution completed successfully!\n")


def main():
    print_banner()

    # Step 1: Self-Test
    run_unit_tests()

    # Step 2: Load Data
    trajectories_file = ROOT_DIR / "outputs" / "analytics" / "trajectories" / "liverpool_tracking_champion_metric_trajectories.csv"
    if not trajectories_file.exists():
        print(f"Trajectory dataset not found at {trajectories_file}. Generating...")
        from src.analytics.create_trajectory_dataset import generate_trajectory_dataset
        tracking_csv = ROOT_DIR / "outputs" / "evaluation" / "videos" / "liverpool_highlights" / "tracking.csv"
        generate_trajectory_dataset(tracking_csv, output_csv=trajectories_file)

    df = pd.read_csv(trajectories_file)

    out_dir = ROOT_DIR / "outputs" / "analytics" / "demo_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 3: Kinematics
    kin_df, workload_df = run_kinematics_pipeline(df, out_dir)

    # Step 4: Tactical Geometry
    run_tactical_geometry(kin_df, out_dir)

    # Step 5: Pitch Heatmaps & Dominance
    run_pitch_heatmaps(kin_df, out_dir)

    # Step 6: Benchmark Table
    display_benchmark_summary()


if __name__ == "__main__":
    main()
