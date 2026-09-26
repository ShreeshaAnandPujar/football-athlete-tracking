"""
Pitch Occupancy Heatmap Module for Football Player Tracking.

Computes 2D Gaussian Kernel Density Estimation (KDE) pitch occupancy:
- Individual player heatmaps
- Team-level collective territorial presence
- Territorial dominance / control differential map (Team A vs. Team B)
- Publication-quality tactical figures overlaid on FIFA standard pitch markings
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


class HeatmapGenerator:
    """
    Generates spatial density heatmaps and territorial dominance maps on a FIFA pitch.
    """

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        grid_resolution: float = 1.0,  # 1 meter per bin
        default_sigma: float = 2.5,    # smoothing radius in meters
    ):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.grid_res = grid_resolution
        self.default_sigma = default_sigma

        # Grid dimensions (bins)
        self.nx = int(round(pitch_length / grid_resolution))
        self.ny = int(round(pitch_width / grid_resolution))

    def compute_density(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        sigma: Optional[float] = None,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Computes 2D smoothed spatial density grid.
        Returns 2D array of shape (ny, nx).
        """
        if sigma is None:
            sigma = self.default_sigma

        # Filter points within or near pitch boundary
        valid = (
            (x_coords >= -2.0)
            & (x_coords <= self.pitch_length + 2.0)
            & (y_coords >= -2.0)
            & (y_coords <= self.pitch_width + 2.0)
        )
        x_clean = np.clip(x_coords[valid], 0.0, self.pitch_length - 1e-4)
        y_clean = np.clip(y_coords[valid], 0.0, self.pitch_width - 1e-4)
        w_clean = weights[valid] if weights is not None else None

        # 2D Histogram
        hist, _, _ = np.histogram2d(
            x_clean,
            y_clean,
            bins=[self.nx, self.ny],
            range=[[0.0, self.pitch_length], [0.0, self.pitch_width]],
            weights=w_clean,
        )

        # Transpose so rows = Y (width), columns = X (length)
        density = hist.T

        # Gaussian smoothing
        sigma_bins = sigma / self.grid_res
        smoothed = gaussian_filter(density, sigma=sigma_bins)

        # Normalize to [0, 1] if not all zeros
        max_val = smoothed.max()
        if max_val > 1e-8:
            smoothed = smoothed / max_val

        return smoothed

    def _draw_tactical_pitch_ax(self, ax: plt.Axes, bg_color: str = "#143820", line_color: str = "#E0E0E0"):
        """Draws standard FIFA pitch lines on a matplotlib Axes."""
        ax.set_facecolor(bg_color)

        # Outer pitch boundary
        pitch_rect = patches.Rectangle(
            (0, 0), self.pitch_length, self.pitch_width,
            linewidth=1.8, edgecolor=line_color, facecolor="none", zorder=3
        )
        ax.add_patch(pitch_rect)

        # Halfway line
        ax.plot([52.5, 52.5], [0, 68], color=line_color, linewidth=1.5, zorder=3)

        # Center circle & spot
        center_circle = patches.Circle((52.5, 34), 9.15, edgecolor=line_color, facecolor="none", linewidth=1.5, zorder=3)
        center_spot = patches.Circle((52.5, 34), 0.6, color=line_color, zorder=3)
        ax.add_patch(center_circle)
        ax.add_patch(center_spot)

        # Left Penalty Box (18-yard)
        left_box = patches.Rectangle((0, 34 - 20.16), 16.5, 40.32, edgecolor=line_color, facecolor="none", linewidth=1.5, zorder=3)
        left_spot = patches.Circle((11, 34), 0.6, color=line_color, zorder=3)
        left_arc = patches.Arc((11, 34), 18.3, 18.3, angle=0, theta1=307, theta2=53, color=line_color, linewidth=1.5, zorder=3)
        left_6yd = patches.Rectangle((0, 34 - 9.16), 5.5, 18.32, edgecolor=line_color, facecolor="none", linewidth=1.2, zorder=3)
        ax.add_patch(left_box)
        ax.add_patch(left_spot)
        ax.add_patch(left_arc)
        ax.add_patch(left_6yd)

        # Right Penalty Box
        right_box = patches.Rectangle((105 - 16.5, 34 - 20.16), 16.5, 40.32, edgecolor=line_color, facecolor="none", linewidth=1.5, zorder=3)
        right_spot = patches.Circle((105 - 11, 34), 0.6, color=line_color, zorder=3)
        right_arc = patches.Arc((105 - 11, 34), 18.3, 18.3, angle=0, theta1=127, theta2=233, color=line_color, linewidth=1.5, zorder=3)
        right_6yd = patches.Rectangle((105 - 5.5, 34 - 9.16), 5.5, 18.32, edgecolor=line_color, facecolor="none", linewidth=1.2, zorder=3)
        ax.add_patch(right_box)
        ax.add_patch(right_spot)
        ax.add_patch(right_arc)
        ax.add_patch(right_6yd)

        # Axis styling
        ax.set_xlim(-3, self.pitch_length + 3)
        ax.set_ylim(-3, self.pitch_width + 3)
        ax.set_aspect("equal")
        ax.invert_yaxis()  # Top-down view (0,0 top-left)
        ax.axis("off")

    def generate_player_heatmap(
        self,
        trajectory_df: pd.DataFrame,
        track_id: int,
        title: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None,
        cmap: str = "magma",
        dpi: int = 150,
    ) -> np.ndarray:
        """
        Generates and saves an individual player pitch occupancy heatmap.
        """
        x_col = "x_pitch_smooth" if "x_pitch_smooth" in trajectory_df.columns else "x_pitch"
        y_col = "y_pitch_smooth" if "y_pitch_smooth" in trajectory_df.columns else "y_pitch"

        player_data = trajectory_df[trajectory_df["track_id"] == track_id]
        if len(player_data) == 0:
            raise ValueError(f"No trajectory data found for track_id={track_id}")

        xs = player_data[x_col].values
        ys = player_data[y_col].values
        density = self.compute_density(xs, ys)

        fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0e1713")
        self._draw_tactical_pitch_ax(ax)

        # Overlay heatmap
        im = ax.imshow(
            density,
            origin="upper",
            extent=[0, self.pitch_length, self.pitch_width, 0],
            cmap=cmap,
            alpha=0.75,
            zorder=2,
        )

        team_name = player_data["team_name"].iloc[0] if "team_name" in player_data.columns else ""
        header = title or f"Pitch Occupancy Heatmap: Player #{track_id} {f'({team_name})' if team_name else ''}"
        ax.set_title(
            header,
            color="#FFFFFF",
            fontsize=15,
            fontweight="bold",
            pad=15,
            family="sans-serif",
        )

        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.5)
        cbar.set_label("Occupancy Density", color="#FFFFFF", fontsize=11)
        cbar.ax.tick_params(colors="#FFFFFF")

        plt.tight_layout()

        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(out_p), dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")

        plt.close(fig)
        return density

    def generate_team_heatmap(
        self,
        trajectory_df: pd.DataFrame,
        team_id: int,
        team_name: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None,
        cmap: str = "YlOrRd",
        dpi: int = 150,
    ) -> np.ndarray:
        """
        Generates and saves team-level collective spatial presence heatmap.
        """
        x_col = "x_pitch_smooth" if "x_pitch_smooth" in trajectory_df.columns else "x_pitch"
        y_col = "y_pitch_smooth" if "y_pitch_smooth" in trajectory_df.columns else "y_pitch"

        if "team_id" not in trajectory_df.columns:
            raise ValueError("trajectory_df must contain 'team_id' column.")

        team_data = trajectory_df[trajectory_df["team_id"] == team_id]
        if len(team_data) == 0:
            raise ValueError(f"No trajectory data found for team_id={team_id}")

        t_name = team_name or (str(team_data["team_name"].iloc[0]) if "team_name" in team_data.columns else f"Team {team_id}")
        xs = team_data[x_col].values
        ys = team_data[y_col].values
        density = self.compute_density(xs, ys)

        fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0e1713")
        self._draw_tactical_pitch_ax(ax)

        im = ax.imshow(
            density,
            origin="upper",
            extent=[0, self.pitch_length, self.pitch_width, 0],
            cmap=cmap,
            alpha=0.75,
            zorder=2,
        )

        ax.set_title(
            f"Collective Pitch Occupancy: {t_name}",
            color="#FFFFFF",
            fontsize=15,
            fontweight="bold",
            pad=15,
            family="sans-serif",
        )

        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.5)
        cbar.set_label("Occupancy Density", color="#FFFFFF", fontsize=11)
        cbar.ax.tick_params(colors="#FFFFFF")

        plt.tight_layout()

        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(out_p), dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")

        plt.close(fig)
        return density

    def generate_dominance_map(
        self,
        trajectory_df: pd.DataFrame,
        team_a_id: int = 0,
        team_b_id: int = 1,
        team_a_name: str = "Team A",
        team_b_name: str = "Team B",
        output_path: Optional[Union[str, Path]] = None,
        dpi: int = 150,
    ) -> np.ndarray:
        """
        Computes territorial dominance differential:
            Diff = Density(Team A) - Density(Team B)
        Plotted with diverging colormap:
            Positive (Red) -> Team A Dominance
            Negative (Blue) -> Team B Dominance
            Zero (Neutral) -> Contested Zone
        """
        x_col = "x_pitch_smooth" if "x_pitch_smooth" in trajectory_df.columns else "x_pitch"
        y_col = "y_pitch_smooth" if "y_pitch_smooth" in trajectory_df.columns else "y_pitch"

        data_a = trajectory_df[trajectory_df["team_id"] == team_a_id]
        data_b = trajectory_df[trajectory_df["team_id"] == team_b_id]

        density_a = self.compute_density(data_a[x_col].values, data_a[y_col].values) if len(data_a) > 0 else np.zeros((self.ny, self.nx))
        density_b = self.compute_density(data_b[x_col].values, data_b[y_col].values) if len(data_b) > 0 else np.zeros((self.ny, self.nx))

        # Differential map
        diff = density_a - density_b
        max_abs = max(0.01, float(np.max(np.abs(diff))))

        fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0e1713")
        self._draw_tactical_pitch_ax(ax)

        im = ax.imshow(
            diff,
            origin="upper",
            extent=[0, self.pitch_length, self.pitch_width, 0],
            cmap="coolwarm",
            vmin=-max_abs,
            vmax=max_abs,
            alpha=0.8,
            zorder=2,
        )

        ax.set_title(
            f"Territorial Dominance Map: {team_a_name} (Red) vs {team_b_name} (Blue)",
            color="#FFFFFF",
            fontsize=15,
            fontweight="bold",
            pad=15,
            family="sans-serif",
        )

        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.5)
        cbar.set_label(f"← {team_b_name} Control | Contested | {team_a_name} Control →", color="#FFFFFF", fontsize=11)
        cbar.ax.tick_params(colors="#FFFFFF")

        plt.tight_layout()

        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(out_p), dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")

        plt.close(fig)
        return diff
