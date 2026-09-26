import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from src.analytics.pitch_model import FootballPitch

class PitchMapper:
    """
    Transforms 2D video pixel coordinates into official FIFA pitch coordinates (meters).
    Pitch dimensions: 105.0m (length, X) x 68.0m (width, Y).
    """

    def __init__(self, homography_matrix=None):
        if homography_matrix is not None:
            self.H = np.array(homography_matrix, dtype=np.float64)
            self.H_inv = np.linalg.inv(self.H)
        else:
            self.H = None
            self.H_inv = None

    @classmethod
    def from_correspondences(cls, pixel_points, pitch_points):
        """
        Computes planar homography from 4 or more (pixel, pitch_meter) correspondences.
        Args:
            pixel_points: list of [x_pixel, y_pixel]
            pitch_points: list of [X_pitch_meters, Y_pitch_meters]
        """
        src_pts = np.array(pixel_points, dtype=np.float32).reshape(-1, 1, 2)
        dst_pts = np.array(pitch_points, dtype=np.float32).reshape(-1, 1, 2)
        
        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            raise ValueError("Failed to compute valid homography matrix from given points.")
        return cls(homography_matrix=H)

    @classmethod
    def from_config_file(cls, config_path):
        """Loads calibration from a JSON or YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {config_path}")
        
        with open(config_path, "r") as f:
            data = json.load(f)
            
        if "homography_matrix" in data:
            return cls(homography_matrix=data["homography_matrix"])
        elif "pixel_points" in data and "pitch_points" in data:
            return cls.from_correspondences(data["pixel_points"], data["pitch_points"])
        else:
            raise ValueError("Calibration file must contain 'homography_matrix' or ('pixel_points', 'pitch_points').")

    def save_calibration(self, output_path):
        """Saves current homography matrix to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "homography_matrix": self.H.tolist(),
            "pitch_length": FootballPitch.LENGTH,
            "pitch_width": FootballPitch.WIDTH
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def pixel_to_pitch(self, x_pixel, y_pixel):
        """
        Maps a single image pixel point to metric pitch coordinate [X_meters, Y_meters].
        """
        if self.H is None:
            raise RuntimeError("Homography matrix not initialized.")
        pt = np.array([[[float(x_pixel), float(y_pixel)]]], dtype=np.float64)
        projected = cv2.perspectiveTransform(pt, self.H)[0][0]
        return float(projected[0]), float(projected[1])

    def pitch_to_pixel(self, x_pitch, y_pitch):
        """
        Maps a metric pitch point [X_meters, Y_meters] back to image pixel [x_pixel, y_pixel].
        """
        if self.H_inv is None:
            raise RuntimeError("Inverse homography matrix not initialized.")
        pt = np.array([[[float(x_pitch), float(y_pitch)]]], dtype=np.float64)
        projected = cv2.perspectiveTransform(pt, self.H_inv)[0][0]
        return float(projected[0]), float(projected[1])

    def batch_pixel_to_pitch(self, points_array):
        """
        Vectorized transformation of an array of (N, 2) pixel coordinates.
        Returns (N, 2) pitch coordinates.
        """
        if self.H is None:
            raise RuntimeError("Homography matrix not initialized.")
        pts = np.array(points_array, dtype=np.float64).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)
        return projected

    def is_on_pitch(self, x_pitch, y_pitch, margin=3.0):
        """Checks if a metric pitch point is within pitch playing boundary (with tolerance margin)."""
        return (-margin <= x_pitch <= FootballPitch.LENGTH + margin) and \
               (-margin <= y_pitch <= FootballPitch.WIDTH + margin)

    @classmethod
    def get_tactical_default(cls, image_width=1920, image_height=1080):
        """
        Generates a robust default planar homography for standard full-pitch tactical broadcast views.
        Uses standard camera elevation (25 deg tilt, elevated gantry) calibrated to 1080p canvas.
        """
        # 4 reference anchor points in image canvas (near touchline and far touchline)
        src_points = [
            [image_width * 0.10, image_height * 0.88],  # Bottom-left (near touchline)
            [image_width * 0.90, image_height * 0.88],  # Bottom-right (near touchline)
            [image_width * 0.82, image_height * 0.22],  # Top-right (far touchline)
            [image_width * 0.18, image_height * 0.22],  # Top-left (far touchline)
            [image_width * 0.50, image_height * 0.88],  # Halfway near
            [image_width * 0.50, image_height * 0.22],  # Halfway far
        ]
        dst_points = [
            [5.0, 65.0],    # Near bottom-left
            [100.0, 65.0],  # Near bottom-right
            [100.0, 3.0],   # Far top-right
            [5.0, 3.0],     # Far top-left
            [52.5, 65.0],   # Halfway near
            [52.5, 3.0],    # Halfway far
        ]
        return cls.from_correspondences(src_points, dst_points)
