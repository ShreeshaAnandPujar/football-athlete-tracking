"""
Team Shape & Formation Estimation Module for Football Analytics.

Computes tactical team geometric metrics per frame and across match phases:
- Team Centroid (center of mass)
- Team Length (longitudinal stretch) and Width (lateral spread)
- Team Compactness (Convex Hull Area in m^2, average dispersion radius)
- Defensive Line and Attacking Line heights
- Tactical Formation Classification (e.g. 4-3-3, 4-4-2, 4-2-3-1, 3-5-2)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from sklearn.cluster import KMeans


# Recognized tactical formations
KNOWN_FORMATIONS = {
    "4-3-3": [4, 3, 3],
    "4-4-2": [4, 4, 2],
    "4-2-3-1": [4, 2, 3, 1],
    "3-5-2": [3, 5, 2],
    "3-4-3": [3, 4, 3],
    "5-3-2": [5, 3, 2],
    "5-4-1": [5, 4, 1],
    "4-1-4-1": [4, 1, 4, 1],
}


class TeamShapeAnalyzer:
    """
    Analyzes collective team spatial organization, tactical lines, and formations.
    """

    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

    def compute_frame_shape_metrics(
        self,
        frame_df: pd.DataFrame,
        team_id: int,
    ) -> Optional[Dict[str, float]]:
        """
        Computes geometric shape metrics for a given team in a single frame.
        Requires at least 3 players to form a 2D hull.
        """
        x_col = "x_pitch_smooth" if "x_pitch_smooth" in frame_df.columns else "x_pitch"
        y_col = "y_pitch_smooth" if "y_pitch_smooth" in frame_df.columns else "y_pitch"

        team_pts = frame_df[frame_df["team_id"] == team_id][[x_col, y_col]].dropna().values
        n_players = len(team_pts)

        if n_players < 2:
            return None

        xs = team_pts[:, 0]
        ys = team_pts[:, 1]

        centroid_x = float(np.mean(xs))
        centroid_y = float(np.mean(ys))

        length = float(np.max(xs) - np.min(xs))
        width = float(np.max(ys) - np.min(ys))

        # Average dispersion radius from centroid
        dispersion_radius = float(np.mean(np.sqrt((xs - centroid_x) ** 2 + (ys - centroid_y) ** 2)))

        # Convex Hull Area (m^2)
        hull_area = 0.0
        hull_vertices = []
        if n_players >= 3:
            try:
                hull = ConvexHull(team_pts)
                hull_area = float(hull.volume)  # For 2D, volume is area
                hull_vertices = team_pts[hull.vertices].tolist()
            except QhullError:
                # Degenerate collinear points
                hull_area = 0.0

        # Defensive and attacking lines (sorted longitudinally)
        sorted_xs = np.sort(xs)
        k_line = min(4, max(1, n_players // 3))
        # Lowest X line and highest X line
        low_line_x = float(np.mean(sorted_xs[:k_line]))
        high_line_x = float(np.mean(sorted_xs[-k_line:]))
        inter_line_dist = high_line_x - low_line_x

        return {
            "n_players": n_players,
            "centroid_x": round(centroid_x, 2),
            "centroid_y": round(centroid_y, 2),
            "length_m": round(length, 2),
            "width_m": round(width, 2),
            "hull_area_m2": round(hull_area, 2),
            "dispersion_radius_m": round(dispersion_radius, 2),
            "low_line_x": round(low_line_x, 2),
            "high_line_x": round(high_line_x, 2),
            "inter_line_dist_m": round(inter_line_dist, 2),
            "hull_vertices": hull_vertices,
        }

    def analyze_sequence_team_shapes(self, trajectory_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes team shape metrics across all frames in a trajectory dataset.
        Returns a DataFrame indexed by (frame, team_id).
        """
        if "team_id" not in trajectory_df.columns:
            raise ValueError("trajectory_df must contain 'team_id'.")

        records = []
        teams = [t for t in trajectory_df["team_id"].unique() if t in [0, 1]]

        for frame_idx, frame_data in trajectory_df.groupby("frame"):
            for tid in teams:
                metrics = self.compute_frame_shape_metrics(frame_data, tid)
                if metrics is not None:
                    metrics["frame"] = frame_idx
                    metrics["team_id"] = tid
                    metrics.pop("hull_vertices", None)  # keep tabular
                    records.append(metrics)

        return pd.DataFrame(records)

    def estimate_formation(
        self,
        trajectory_df: pd.DataFrame,
        team_id: int,
        n_outfield_target: int = 10,
    ) -> Dict[str, Union[str, List[int], Dict[str, float]]]:
        """
        Estimates the tactical formation of a team by analyzing player average positions.
        """
        x_col = "x_pitch_smooth" if "x_pitch_smooth" in trajectory_df.columns else "x_pitch"
        y_col = "y_pitch_smooth" if "y_pitch_smooth" in trajectory_df.columns else "y_pitch"

        team_data = trajectory_df[trajectory_df["team_id"] == team_id]
        if len(team_data) == 0:
            return {"formation": "Unknown", "line_counts": [], "confidence": 0.0}

        # Compute mean position and count per track
        grouped = team_data.groupby("track_id")
        player_means_x = grouped[x_col].mean()
        player_means_y = grouped[y_col].mean()
        player_counts = grouped.size()

        # Keep players with sufficient appearances
        max_c = float(player_counts.max())
        min_frames = max(1, int(max_c * 0.15))
        active_mask = player_counts >= min_frames
        active_tids = player_counts[active_mask].index

        active_players = pd.DataFrame({
            x_col: player_means_x.loc[active_tids],
            y_col: player_means_y.loc[active_tids],
        })

        n_players = len(active_players)
        if n_players < 7:
            # Not enough players to confidently estimate 11v11 formation
            return {
                "formation": f"Small-Sided ({n_players}p)",
                "line_counts": [n_players],
                "confidence": 0.3,
                "n_active_players": n_players,
            }

        # Determine attacking direction:
        # Sort players along X
        sorted_players = active_players.sort_values(x_col).reset_index()

        # If team has 11 players, the extreme player is usually the Goalkeeper
        # Identify Goalkeeper as the most isolated player on the deep end
        pts_x = sorted_players[x_col].values
        if n_players >= 10:
            # Exclude the deepest player (goalkeeper)
            outfield_x = pts_x[1:]
        else:
            outfield_x = pts_x

        n_outfield = len(outfield_x)

        # Cluster outfield players along X into 3 lines (Defenders, Midfielders, Attackers)
        best_formation = "4-3-3"
        best_diff = 999
        predicted_lines = []

        for n_lines in [3, 4]:
            if n_outfield < n_lines:
                continue
            km = KMeans(n_clusters=n_lines, n_init=10, random_state=42)
            line_labels = km.fit_predict(outfield_x.reshape(-1, 1))

            # Order cluster centroids longitudinally
            order = np.argsort(km.cluster_centers_.flatten())
            counts_per_line = [int(np.sum(line_labels == c)) for c in order]

            # Compare against known formations
            for name, template in KNOWN_FORMATIONS.items():
                if len(template) == n_lines:
                    diff = sum(abs(c - t) for c, t in zip(counts_per_line, template))
                    if diff < best_diff:
                        best_diff = diff
                        best_formation = name
                        predicted_lines = counts_per_line

        formation_str = "-".join(map(str, predicted_lines))
        confidence = max(0.2, round(1.0 - (best_diff / max(1, n_outfield)), 2))

        return {
            "formation": best_formation,
            "detected_line_counts": predicted_lines,
            "line_str": formation_str,
            "confidence": confidence,
            "n_outfield_analyzed": n_outfield,
        }
