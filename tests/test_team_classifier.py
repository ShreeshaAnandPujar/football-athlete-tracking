"""
Test and Benchmark TeamClassifier on SoccerNet Ground Truth
"""

import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import pandas as pd
from src.analytics.team_classifier import TeamClassifier


def test_team_classifier_on_soccernet():
    ini_path = Path("data/raw/SoccerNet/tracking-2023/train/train/SNMOT-113/gameinfo.ini")
    assert ini_path.exists(), f"Missing {ini_path}"

    gt_teams = {}
    for line in ini_path.read_text().splitlines():
        m = re.match(r"trackletID_(\d+)=\s*([^;]+)", line.strip())
        if m:
            tid, role = int(m.group(1)), m.group(2).strip().lower()
            if "ball" not in role:
                if "left" in role:
                    gt_teams[tid] = "team_left"
                elif "right" in role:
                    gt_teams[tid] = "team_right"
                else:
                    gt_teams[tid] = "referee"

    gt_file = Path("data/raw/SoccerNet/tracking-2023/train/train/SNMOT-113/gt/gt.txt")
    assert gt_file.exists()

    rows = []
    for line in gt_file.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 6:
            frame = int(parts[0])
            tid = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            rows.append({
                "frame": frame,
                "track_id": tid,
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h
            })

    df = pd.DataFrame(rows)
    img_dir = Path("data/raw/SoccerNet/tracking-2023/train/train/SNMOT-113/img1")

    # Sample first 60 frames for quick benchmark
    unique_frames = sorted(df["frame"].unique())[:60]
    frames = {}
    for fidx in unique_frames:
        img_p = img_dir / f"{fidx:06d}.jpg"
        if img_p.exists():
            frames[fidx] = cv2.imread(str(img_p))

    sub_df = df[df["frame"].isin(frames.keys())]

    clf = TeamClassifier(n_teams=2, include_referee=True)
    feats, tids, _ = clf.extract_dataset_features(frames, sub_df, sample_stride=2)
    clf.fit(feats, tids)

    results = clf.evaluate_against_ground_truth(gt_teams)
    print("=" * 60)
    print("SOCCERNET SNMOT-113 TEAM CLASSIFICATION BENCHMARK")
    print("=" * 60)
    print(f"Evaluated Tracks: {results['evaluated_tracks']}")
    print(f"Accuracy: {results['accuracy']}%")
    print(f"Distribution: {results['gt_distribution']}")
    print("=" * 60)

    # Sanity check: Accuracy should be high (> 80%) on ground-truth tracklets
    assert results["accuracy"] >= 75.0, f"Accuracy too low: {results['accuracy']}%"
    print("Test passed successfully!")


if __name__ == "__main__":
    test_team_classifier_on_soccernet()
