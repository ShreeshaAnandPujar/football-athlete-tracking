"""
Player Movement & Kinematics Module for Football Analytics.

Computes physical tracking metrics from metric pitch trajectories:
- Metric distance (meters) and cumulative distance
- Instantaneous speed (m/s and km/h) with rolling temporal smoothing
- Instantaneous acceleration (m/s^2)
- FIFA physical intensity breakdown (Walking, Jogging, Running, Sprinting)
- Spatial pitch zone occupancy (Longitudinal Thirds & Lateral Channels)
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd


# Standard Physical Intensity Zones (km/h)
INTENSITY_THRESHOLDS = {
    "walking": (0.0, 7.2),       # < 2.0 m/s
    "jogging": (7.2, 14.4),      # 2.0 - 4.0 m/s
    "running": (14.4, 19.8),     # 4.0 - 5.5 m/s
    "sprinting": (19.8, 45.0),   # >= 5.5 m/s
}

# Maximum biologically possible player sprint speed (11.5 m/s = 41.4 km/h)
MAX_PLAUSIBLE_SPEED_MPS = 11.5


class KinematicsAnalyzer:
    """
    Computes kinematics and physical workload metrics from metric pitch coordinates.
    """

    def __init__(
        self,
        fps: float = 25.0,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        speed_smooth_window: int = 5,
        max_speed_clamp_mps: float = MAX_PLAUSIBLE_SPEED_MPS,
    ):
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.smooth_window = speed_smooth_window
        self.max_speed_clamp_mps = max_speed_clamp_mps

    def compute_frame_kinematics(self, trajectory_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes frame-level step distance, speed, acceleration, and zones.
        Assumes trajectory_df contains: 'frame', 'track_id', 'x_pitch', 'y_pitch'
        (uses smoothed coordinates 'x_pitch_smooth', 'y_pitch_smooth' if present).
        """
        df = trajectory_df.copy()

        x_col = "x_pitch_smooth" if "x_pitch_smooth" in df.columns else "x_pitch"
        y_col = "y_pitch_smooth" if "y_pitch_smooth" in df.columns else "y_pitch"

        # Ensure sorted by track_id and frame
        df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)

        # Delta time and frame step
        df["frame_diff"] = df.groupby("track_id")["frame"].diff().fillna(1)
        df["dt_step"] = df["frame_diff"] * self.dt

        # Spatial delta
        df["dx"] = df.groupby("track_id")[x_col].diff().fillna(0.0)
        df["dy"] = df.groupby("track_id")[y_col].diff().fillna(0.0)

        # Step distance in meters
        df["step_dist_raw"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)

        # Sanity check: clamp teleports / occlusion jumps exceeding max sprint speed
        max_dist_step = self.max_speed_clamp_mps * df["dt_step"]
        df["is_teleport"] = df["step_dist_raw"] > (max_dist_step * 1.5)
        df["step_dist_m"] = np.where(df["is_teleport"], 0.0, df["step_dist_raw"])

        # Cumulative distance per track
        df["cumulative_distance_m"] = df.groupby("track_id")["step_dist_m"].cumsum().round(2)

        # Instantaneous speed (m/s)
        raw_speed_mps = np.where(df["is_teleport"], 0.0, df["step_dist_m"] / df["dt_step"])
        df["speed_mps_raw"] = np.clip(raw_speed_mps, 0.0, self.max_speed_clamp_mps)

        # Rolling average speed per track
        df["speed_mps"] = (
            df.groupby("track_id")["speed_mps_raw"]
            .transform(lambda s: s.rolling(self.smooth_window, min_periods=1).mean())
            .round(2)
        )
        df["speed_kmh"] = (df["speed_mps"] * 3.6).round(2)

        # Acceleration (m/s^2)
        df["dv"] = df.groupby("track_id")["speed_mps"].diff().fillna(0.0)
        df["acceleration_mps2"] = (
            np.where(df["is_teleport"], 0.0, df["dv"] / df["dt_step"])
            .clip(-8.0, 8.0)  # Max physical human deceleration/acceleration clamp
            .round(2)
        )

        # Intensity zone assignment
        df["intensity_zone"] = self._assign_intensity_zones(df["speed_kmh"])

        # Tactical pitch zone assignment
        df["pitch_third"] = self._assign_pitch_thirds(df[x_col])
        df["pitch_channel"] = self._assign_pitch_channels(df[y_col])

        # Clean up temporary columns
        drop_cols = ["frame_diff", "dt_step", "dx", "dy", "step_dist_raw", "speed_mps_raw", "dv"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        return df

    def _assign_intensity_zones(self, speed_kmh_series: pd.Series) -> pd.Series:
        """Categorizes speeds into physical intensity bands."""
        conditions = [
            (speed_kmh_series < 7.2),
            (speed_kmh_series >= 7.2) & (speed_kmh_series < 14.4),
            (speed_kmh_series >= 14.4) & (speed_kmh_series < 19.8),
            (speed_kmh_series >= 19.8),
        ]
        choices = ["walking", "jogging", "running", "sprinting"]
        return pd.Series(np.select(conditions, choices, default="walking"), index=speed_kmh_series.index)

    def _assign_pitch_thirds(self, x_series: pd.Series) -> pd.Series:
        """Categorizes X coordinates into Longitudinal Thirds (Defensive, Middle, Attacking)."""
        third_len = self.pitch_length / 3.0
        conditions = [
            (x_series < third_len),
            (x_series >= third_len) & (x_series < 2.0 * third_len),
            (x_series >= 2.0 * third_len),
        ]
        choices = ["defensive_third", "middle_third", "attacking_third"]
        return pd.Series(np.select(conditions, choices, default="middle_third"), index=x_series.index)

    def _assign_pitch_channels(self, y_series: pd.Series) -> pd.Series:
        """Categorizes Y coordinates into 5 Tactical Lateral Channels."""
        conditions = [
            (y_series < 20.0),
            (y_series >= 20.0) & (y_series < 30.0),
            (y_series >= 30.0) & (y_series < 38.0),
            (y_series >= 38.0) & (y_series < 48.0),
            (y_series >= 48.0),
        ]
        choices = ["left_wing", "half_space_left", "center", "half_space_right", "right_wing"]
        return pd.Series(np.select(conditions, choices, default="center"), index=y_series.index)

    def summarize_player_kinematics(self, annotated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates workload statistics per player (track_id):
        - total_distance_m
        - max_speed_kmh
        - avg_speed_kmh
        - sprint_distance_m
        - sprint_count
        - pitch zone percentages
        """
        summary_rows = []

        grouped = annotated_df.groupby("track_id")
        for tid, group in grouped:
            total_dist = float(group["step_dist_m"].sum())
            max_speed = float(group["speed_kmh"].max())
            avg_speed = float(group["speed_kmh"].mean())

            # Sprint metrics (V >= 19.8 km/h)
            sprint_mask = group["intensity_zone"] == "sprinting"
            sprint_dist = float(group.loc[sprint_mask, "step_dist_m"].sum())
            # Count consecutive sprint bouts
            sprint_bouts = (sprint_mask.astype(int).diff() == 1).sum()

            # Time distribution across intensity zones
            n_frames = max(1, len(group))
            zone_counts = group["intensity_zone"].value_counts()
            pct_walking = round((zone_counts.get("walking", 0) / n_frames) * 100.0, 1)
            pct_jogging = round((zone_counts.get("jogging", 0) / n_frames) * 100.0, 1)
            pct_running = round((zone_counts.get("running", 0) / n_frames) * 100.0, 1)
            pct_sprinting = round((zone_counts.get("sprinting", 0) / n_frames) * 100.0, 1)

            # Spatial distribution across pitch thirds
            third_counts = group["pitch_third"].value_counts()
            pct_def = round((third_counts.get("defensive_third", 0) / n_frames) * 100.0, 1)
            pct_mid = round((third_counts.get("middle_third", 0) / n_frames) * 100.0, 1)
            pct_att = round((third_counts.get("attacking_third", 0) / n_frames) * 100.0, 1)

            team_id = int(group["team_id"].iloc[0]) if "team_id" in group.columns else -1
            team_name = str(group["team_name"].iloc[0]) if "team_name" in group.columns else "Unknown"

            summary_rows.append({
                "track_id": int(tid),
                "team_id": team_id,
                "team_name": team_name,
                "total_frames": int(n_frames),
                "total_distance_m": round(total_dist, 2),
                "max_speed_kmh": round(max_speed, 2),
                "avg_speed_kmh": round(avg_speed, 2),
                "sprint_distance_m": round(sprint_dist, 2),
                "sprint_bouts": int(sprint_bouts),
                "pct_walking": pct_walking,
                "pct_jogging": pct_jogging,
                "pct_running": pct_running,
                "pct_sprinting": pct_sprinting,
                "pct_defensive_third": pct_def,
                "pct_middle_third": pct_mid,
                "pct_attacking_third": pct_att,
            })

        summary_df = pd.DataFrame(summary_rows)
        return summary_df.sort_values("total_distance_m", ascending=False).reset_index(drop=True)
