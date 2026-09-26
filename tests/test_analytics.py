"""
Comprehensive Unit and Integration Test Suite for Football Analytics.

Tests:
1. Team Classification (Jersey extraction, green grass masking, temporal majority voting)
2. Player Kinematics (Metric distance, speed, acceleration, intensity zones)
3. Pitch Occupancy Heatmaps (KDE density, grid shapes, normalization)
4. Team Shape & Formation (Centroids, convex hulls, formation estimation)
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analytics.heatmaps import HeatmapGenerator
from src.analytics.kinematics import KinematicsAnalyzer
from src.analytics.pitch_model import FootballPitch
from src.analytics.team_classifier import JerseyColorExtractor, TeamClassifier
from src.analytics.team_shape import TeamShapeAnalyzer


def test_jersey_color_extractor():
    extractor = JerseyColorExtractor()

    # Synthetic 100x50 player patch (green pitch border, red torso)
    crop = np.zeros((100, 50, 3), dtype=np.uint8)
    # Background grass (green in BGR: B=30, G=150, R=30)
    crop[:, :] = [30, 150, 30]
    # Jersey torso (red in BGR: B=20, G=20, R=220)
    crop[15:50, 10:40] = [20, 20, 220]

    # Test green masking
    torso_patch = crop[15:50, 10:40]
    mask = extractor.mask_pitch_green(torso_patch)
    assert mask.sum() > 0, "Torso red pixels should not be masked as green grass"

    # Test full extraction
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[200:300, 400:450] = crop
    feat = extractor.extract_features(frame, [400, 200, 450, 300])
    assert feat is not None, "Feature extraction failed on valid player"
    assert len(feat) == 6, f"Expected 6-dim feature vector, got {len(feat)}"
    assert np.all(feat >= 0.0) and np.all(feat <= 1.0), "Features must be normalized in [0, 1]"


def test_team_classifier_clustering():
    clf = TeamClassifier(n_teams=2, include_referee=False)

    # 2 distinct clusters in normalized feature space
    cluster_a = np.random.normal(loc=[0.3, 0.8, 0.7, 0.05, 0.9, 0.8], scale=0.03, size=(20, 6))
    cluster_b = np.random.normal(loc=[0.8, 0.4, 0.4, 0.60, 0.1, 0.9], scale=0.03, size=(20, 6))
    features = np.vstack([cluster_a, cluster_b]).astype(np.float32)

    # Track 1 has samples from cluster A; Track 2 has samples from cluster B
    track_ids = [1] * 20 + [2] * 20
    clf.fit(features, track_ids)

    pred_1 = clf.predict_track(1)
    pred_2 = clf.predict_track(2)
    assert pred_1 != pred_2, "Distinct feature clusters should be assigned to different teams"


def test_kinematics_analyzer():
    fps = 25.0
    analyzer = KinematicsAnalyzer(fps=fps)

    # Synthetic trajectory: Player moving at constant 5 m/s (18 km/h) along X axis
    # In 25 frames (1 second), player covers 5 meters
    frames = list(range(1, 26))
    xs = [10.0 + i * (5.0 / 25.0) for i in range(25)]
    ys = [34.0] * 25
    df = pd.DataFrame({
        "frame": frames,
        "track_id": [1] * 25,
        "x_pitch": xs,
        "y_pitch": ys,
    })

    kin_df = analyzer.compute_frame_kinematics(df)
    assert "speed_kmh" in kin_df.columns
    assert "cumulative_distance_m" in kin_df.columns

    # Final cumulative distance should be approx 4.8 - 5.0 meters
    final_dist = kin_df["cumulative_distance_m"].iloc[-1]
    assert abs(final_dist - 4.8) < 0.5, f"Expected ~4.8m, got {final_dist}"

    # Speed should be ~18 km/h
    mean_speed = kin_df["speed_kmh"].iloc[5:].mean()
    assert abs(mean_speed - 18.0) < 1.0, f"Expected ~18 km/h, got {mean_speed}"

    # Summary aggregation
    summary = analyzer.summarize_player_kinematics(kin_df)
    assert len(summary) == 1
    assert summary["track_id"].iloc[0] == 1
    assert summary["pct_middle_third"].iloc[0] == 0.0 or summary["pct_defensive_third"].iloc[0] == 100.0


def test_heatmap_generator():
    gen = HeatmapGenerator(pitch_length=105.0, pitch_width=68.0)
    xs = np.array([50.0, 52.5, 55.0])
    ys = np.array([30.0, 34.0, 38.0])
    density = gen.compute_density(xs, ys, sigma=2.0)

    assert density.shape == (68, 105), f"Expected (68, 105), got {density.shape}"
    assert density.max() == 1.0, "Peak density should be normalized to 1.0"
    assert density[34, 52] > density[0, 0], "Center spot density should exceed corner"


def test_team_shape_and_formation():
    analyzer = TeamShapeAnalyzer()

    # Synthetic 4-3-3 formation
    # 4 defenders around X=20, 3 midfielders around X=50, 3 forwards around X=80
    xs = [20, 20, 22, 22, 50, 52, 51, 80, 82, 85]
    ys = [12, 26, 42, 58, 20, 34, 48, 18, 34, 52]
    df = pd.DataFrame({
        "frame": [1] * 10,
        "track_id": list(range(1, 11)),
        "team_id": [0] * 10,
        "x_pitch": xs,
        "y_pitch": ys,
    })

    metrics = analyzer.compute_frame_shape_metrics(df, team_id=0)
    assert metrics is not None
    assert abs(metrics["centroid_x"] - 48.4) < 1.0
    assert abs(metrics["length_m"] - 65.0) < 1.0
    assert metrics["hull_area_m2"] > 1000.0

    formation = analyzer.estimate_formation(df, team_id=0)
    assert formation["formation"] == "4-3-3"


if __name__ == "__main__":
    print("Running test_jersey_color_extractor...")
    test_jersey_color_extractor()
    print("Running test_team_classifier_clustering...")
    test_team_classifier_clustering()
    print("Running test_kinematics_analyzer...")
    test_kinematics_analyzer()
    print("Running test_heatmap_generator...")
    test_heatmap_generator()
    print("Running test_team_shape_and_formation...")
    test_team_shape_and_formation()
    print("\nALL ANALYTICS TESTS PASSED SUCCESSFULLY!")
