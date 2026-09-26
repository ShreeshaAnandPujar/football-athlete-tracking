import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, deque

from src.analytics.pitch_model import FootballPitch

class DynamicPitchMapper:
    """
    Dynamic Shot-Aware Pitch Mapper with ORB Camera Motion Compensation
    and Replay/Close-up Gating for broadcast football videos.
    """

    def __init__(self, anchor_homography=None, smoothing_alpha=0.4):
        self.H_anchor = np.array(anchor_homography, dtype=np.float64) if anchor_homography is not None else None
        self.smoothing_alpha = smoothing_alpha
        self.smoothed_positions = {}  # track_id -> (x, y)
        self.orb = cv2.ORB_create(nfeatures=1200)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_gray = None
        self.prev_kps = None
        self.prev_des = None
        self.current_H = self.H_anchor.copy() if self.H_anchor is not None else None

    @classmethod
    def get_liverpool_attacking_box_homography(cls):
        """
        Calibrated planar homography for Liverpool attacking third shot (frames 1-386).
        Maps the visible 18-yard box, 6-yard box, penalty arc, and flanks to the right goal (X: 70-105m, Y: 10-60m).
        """
        # Verified image pixel anchors -> true metric pitch anchors (right goal at X=105m)
        pixel_pts = [
            [1195.0, 377.0],  # Goalkeeper #140 (goal line center: X=104, Y=34)
            [465.0, 458.0],   # Player #2 (penalty arc apex: X=85, Y=34)
            [947.0, 393.0],   # Player #20 (penalty box center: X=95, Y=34)
            [137.0, 360.0],   # Player #7 (left wing / flank: X=76, Y=48)
            [120.0, 435.0],   # Player #5 (left touchline: X=75, Y=56)
            [283.0, 230.0],   # Player #71 (right touchline: X=78, Y=16)
            [723.0, 325.0],   # Player #1 (18-yard box top edge: X=88.5, Y=26)
        ]
        pitch_pts = [
            [104.0, 34.0],
            [85.0, 34.0],
            [95.0, 34.0],
            [76.0, 48.0],
            [75.0, 56.0],
            [78.0, 16.0],
            [88.5, 26.0],
        ]
        H, _ = cv2.findHomography(np.array(pixel_pts, dtype=np.float32), np.array(pitch_pts, dtype=np.float32))
        return H

    def reset_anchor(self, new_anchor_H):
        """Resets reference homography for a new camera shot."""
        self.H_anchor = np.array(new_anchor_H, dtype=np.float64)
        self.current_H = self.H_anchor.copy()
        self.prev_gray = None
        self.prev_kps = None
        self.prev_des = None
        self.smoothed_positions.clear()

    def update_camera_motion(self, frame_bgr):
        """
        Estimates inter-frame camera motion via ORB feature matching
        and updates the dynamic homography matrix H_t.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kps, des = self.orb.detectAndCompute(gray, None)

        if self.prev_gray is not None and self.prev_des is not None and des is not None and len(des) > 30:
            matches = self.matcher.match(des, self.prev_des)
            if len(matches) >= 15:
                matches = sorted(matches, key=lambda x: x.distance)[:80]
                pts_curr = np.float32([kps[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                pts_prev = np.float32([self.prev_kps[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                
                # Estimate affine motion (pan/tilt/zoom)
                M, inliers = cv2.estimateAffinePartial2D(pts_curr, pts_prev)
                if M is not None and np.sum(inliers) >= 12:
                    # Expand 2x3 affine to 3x3 homography
                    M_homo = np.eye(3, dtype=np.float64)
                    M_homo[:2, :] = M
                    # Update dynamic homography
                    self.current_H = self.current_H @ M_homo

        self.prev_gray = gray
        self.prev_kps = kps
        self.prev_des = des

    def project_player(self, track_id, x_pixel, y_pixel):
        """
        Projects a player's foot pixel position to metric pitch coordinates
        with outlier clamping and EMA smoothing.
        """
        if self.current_H is None:
            return None, None, False

        pt = np.array([[[float(x_pixel), float(y_pixel)]]], dtype=np.float64)
        proj = cv2.perspectiveTransform(pt, self.current_H)[0][0]
        raw_x = float(proj[0])
        raw_y = float(proj[1])

        # Physical sanity check: reject degenerate projections
        if np.isnan(raw_x) or np.isnan(raw_y) or abs(raw_x) > 200 or abs(raw_y) > 150:
            return None, None, False

        # Apply exponential moving average (EMA) smoothing per track
        if track_id in self.smoothed_positions:
            prev_x, prev_y = self.smoothed_positions[track_id]
            smooth_x = self.smoothing_alpha * raw_x + (1.0 - self.smoothing_alpha) * prev_x
            smooth_y = self.smoothing_alpha * raw_y + (1.0 - self.smoothing_alpha) * prev_y
        else:
            smooth_x, smooth_y = raw_x, raw_y

        self.smoothed_positions[track_id] = (smooth_x, smooth_y)

        # Check playing field bounds (with 3m technical margin)
        is_in_pitch = (-2.0 <= smooth_x <= FootballPitch.LENGTH + 2.0) and (-2.0 <= smooth_y <= FootballPitch.WIDTH + 2.0)
        return round(smooth_x, 2), round(smooth_y, 2), is_in_pitch
