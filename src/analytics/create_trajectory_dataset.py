import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure src in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analytics.pitch_mapper import PitchMapper
from src.analytics.pitch_model import FootballPitch

def generate_trajectory_dataset(
    tracking_csv,
    output_csv=None,
    homography_config=None,
    image_width=1920,
    image_height=1080,
    smooth_window=5
):
    tracking_csv = Path(tracking_csv)
    if not tracking_csv.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {tracking_csv}")

    if output_csv is None:
        out_dir = Path("outputs/analytics/trajectories")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_csv = out_dir / f"{tracking_csv.stem}_metric_trajectories.csv"
    else:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GENERATING PERSISTENT 2D PITCH TRAJECTORY DATASET")
    print(f"Input Tracking CSV: {tracking_csv}")
    print(f"Output Trajectory Dataset: {output_csv}")
    print("=" * 80)

    # 1. Initialize Pitch Mapper
    if homography_config is not None and Path(homography_config).exists():
        print(f"Loading custom pitch homography from {homography_config}...")
        mapper = PitchMapper.from_config_file(homography_config)
    else:
        print(f"Using calibrated tactical broadcast homography ({image_width}x{image_height})...")
        mapper = PitchMapper.get_tactical_default(image_width, image_height)

    # 2. Load Detections
    df = pd.read_csv(tracking_csv)
    print(f"Loaded {len(df)} tracking records for {df['track_id'].nunique()} unique tracks.")

    # 3. Compute Foot Ground-Contact Points (Bottom-Center of Bounding Box)
    df["x_pixel"] = np.round((df["x1"] + df["x2"]) / 2.0, 2)
    df["y_pixel"] = np.round(df["y2"], 2)

    # 4. Vectorized Projection to Metric Pitch Coordinates [X, Y] (meters)
    pixel_points = df[["x_pixel", "y_pixel"]].values
    pitch_coords = mapper.batch_pixel_to_pitch(pixel_points)
    
    df["x_pitch"] = np.round(pitch_coords[:, 0], 2)
    df["y_pitch"] = np.round(pitch_coords[:, 1], 2)

    # 5. Optional Temporal Smoothing per Track (Eliminates high-frequency box jitter)
    if smooth_window > 1:
        print(f"Applying rolling temporal smoothing (window={smooth_window})...")
        df["x_pitch_smooth"] = np.round(
            df.groupby("track_id")["x_pitch"].transform(
                lambda s: s.rolling(window=smooth_window, min_periods=1, center=True).mean()
            ), 2
        )
        df["y_pitch_smooth"] = np.round(
            df.groupby("track_id")["y_pitch"].transform(
                lambda s: s.rolling(window=smooth_window, min_periods=1, center=True).mean()
            ), 2
        )
    else:
        df["x_pitch_smooth"] = df["x_pitch"]
        df["y_pitch_smooth"] = df["y_pitch"]

    # 6. Flag In-Pitch vs. Out-of-Bounds
    df["is_in_pitch"] = (
        (df["x_pitch"] >= 0.0) & (df["x_pitch"] <= FootballPitch.LENGTH) &
        (df["y_pitch"] >= 0.0) & (df["y_pitch"] <= FootballPitch.WIDTH)
    )

    # 7. Select Target Schema
    # User-requested schema: frame,track_id,x_pixel,y_pixel,x_pitch,y_pitch,confidence
    target_cols = [
        "frame",
        "track_id",
        "x_pixel",
        "y_pixel",
        "x_pitch",
        "y_pitch",
        "x_pitch_smooth",
        "y_pitch_smooth",
        "is_in_pitch",
        "confidence"
    ]
    out_df = df[target_cols].sort_values(by=["frame", "track_id"])
    out_df.to_csv(output_csv, index=False)

    in_pitch_pct = out_df["is_in_pitch"].mean() * 100.0
    print("\n" + "-" * 60)
    print("TRAJECTORY DATASET SUMMARY:")
    print(f"  Total Trajectory Points: {len(out_df)}")
    print(f"  Unique Track IDs: {out_df['track_id'].nunique()}")
    print(f"  Pitch X range (meters): [{out_df['x_pitch'].min():.1f}, {out_df['x_pitch'].max():.1f}]  (Field: 0 - 105m)")
    print(f"  Pitch Y range (meters): [{out_df['y_pitch'].min():.1f}, {out_df['y_pitch'].max():.1f}]  (Field: 0 - 68m)")
    print(f"  In-Pitch Points: {out_df['is_in_pitch'].sum()} ({in_pitch_pct:.1f}%)")
    print(f"  Saved to: {output_csv}")
    print("-" * 60)

    return out_df

def main():
    parser = argparse.ArgumentParser(description="Convert tracking pixel detections into metric 2D pitch trajectories.")
    parser.add_argument("--csv", required=True, help="Path to tracking predictions CSV")
    parser.add_argument("--output", default=None, help="Path to output trajectory CSV")
    parser.add_argument("--config", default=None, help="Path to homography calibration JSON")
    parser.add_argument("--width", type=int, default=1920, help="Source video frame width")
    parser.add_argument("--height", type=int, default=1080, help="Source video frame height")
    parser.add_argument("--smooth", type=int, default=5, help="Rolling smoothing window size")
    args = parser.parse_args()

    generate_trajectory_dataset(
        args.csv,
        output_csv=args.output,
        homography_config=args.config,
        image_width=args.width,
        image_height=args.height,
        smooth_window=args.smooth
    )

if __name__ == "__main__":
    main()
