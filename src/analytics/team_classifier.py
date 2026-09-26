"""
Team Classifier Module for Football Player Tracking.

Extracts jersey color representations from player bounding boxes,
filters out pitch grass background, performs unsupervised clustering
(K-Means / GMM) to separate teams and officials, and applies temporal
majority voting across tracklets to guarantee identity-consistent team assignment.
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class JerseyColorExtractor:
    """
    Extracts representative jersey color features from player bounding boxes,
    suppressing pitch background grass.
    """

    def __init__(
        self,
        torso_y_range: Tuple[float, float] = (0.15, 0.50),
        torso_x_range: Tuple[float, float] = (0.18, 0.82),
        min_pixels: int = 15,
    ):
        self.torso_y_range = torso_y_range
        self.torso_x_range = torso_x_range
        self.min_pixels = min_pixels

    def get_torso_crop(
        self,
        frame: np.ndarray,
        bbox: Union[List[float], Tuple[float, float, float, float], np.ndarray]
    ) -> Optional[np.ndarray]:
        """Crops the player torso region from the bounding box [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        h_frame, w_frame = frame.shape[:2]

        x1 = max(0, min(w_frame - 1, x1))
        x2 = max(0, min(w_frame, x2))
        y1 = max(0, min(h_frame - 1, y1))
        y2 = max(0, min(h_frame, y2))

        w = x2 - x1
        h = y2 - y1

        if w < 6 or h < 12:
            return None

        ty1 = y1 + int(h * self.torso_y_range[0])
        ty2 = y1 + int(h * self.torso_y_range[1])
        tx1 = x1 + int(w * self.torso_x_range[0])
        tx2 = x1 + int(w * self.torso_x_range[1])

        ty1 = max(0, min(h_frame - 1, ty1))
        ty2 = max(ty1 + 2, min(h_frame, ty2))
        tx1 = max(0, min(w_frame - 1, tx1))
        tx2 = max(tx1 + 2, min(w_frame, tx2))

        crop = frame[ty1:ty2, tx1:tx2]
        return crop if crop.size > 0 else None

    def mask_pitch_green(self, bgr_crop: np.ndarray) -> np.ndarray:
        """
        Creates a boolean mask where True indicates NON-pitch pixels.
        Grass is identified in HSV space (Hue ~ 30 to 88).
        """
        hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
        lower_green = np.array([30, 35, 30], dtype=np.uint8)
        upper_green = np.array([88, 255, 255], dtype=np.uint8)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        jersey_mask = green_mask == 0
        return jersey_mask

    def extract_features(self, frame: np.ndarray, bbox: Union[List[float], np.ndarray]) -> Optional[np.ndarray]:
        """
        Extracts a feature vector for the player jersey:
        - Median CIELAB [L*, a*, b*] (perceptually uniform color difference)
        - Median HSV [H, S, V]
        Returns: feature array of shape (6,) or None if invalid crop.
        """
        crop = self.get_torso_crop(frame, bbox)
        if crop is None:
            return None

        jersey_mask = self.mask_pitch_green(crop)
        valid_bgr = crop[jersey_mask]

        if len(valid_bgr) < self.min_pixels:
            ch, cw = crop.shape[:2]
            inner = crop[ch // 4 : 3 * ch // 4, cw // 4 : 3 * cw // 4]
            if inner.size == 0:
                return None
            valid_bgr = inner.reshape(-1, 3)

        if len(valid_bgr) < self.min_pixels:
            return None

        valid_bgr_img = valid_bgr.reshape(-1, 1, 3).astype(np.uint8)
        lab = cv2.cvtColor(valid_bgr_img, cv2.COLOR_BGR2LAB).reshape(-1, 3)
        hsv = cv2.cvtColor(valid_bgr_img, cv2.COLOR_BGR2HSV).reshape(-1, 3)

        lab_median = np.median(lab, axis=0)  # [L, a, b]
        hsv_median = np.median(hsv, axis=0)  # [H, S, V]

        features = np.array([
            lab_median[0] / 255.0,  # Lightness (0-1)
            lab_median[1] / 255.0,  # Green-Red (0-1)
            lab_median[2] / 255.0,  # Blue-Yellow (0-1)
            hsv_median[0] / 180.0,  # Hue (0-1)
            hsv_median[1] / 255.0,  # Saturation (0-1)
            hsv_median[2] / 255.0,  # Value (0-1)
        ], dtype=np.float32)

        return features


class TeamClassifier:
    """
    Classifies players into teams and officials using jersey color clustering
    and temporal majority voting across tracklets.
    """

    def __init__(self, n_teams: int = 2, include_referee: bool = True, random_state: int = 42):
        self.n_teams = n_teams
        self.include_referee = include_referee
        self.k = n_teams + (1 if include_referee else 0)
        self.random_state = random_state
        self.extractor = JerseyColorExtractor()
        self.cluster_model: Optional[KMeans] = None
        self.track_teams: Dict[int, int] = {}
        self.team_names: Dict[int, str] = {0: "Team A", 1: "Team B", 2: "Referee / GK"}
        self.team_colors: Dict[int, Tuple[int, int, int]] = {
            0: (40, 40, 220),     # Red (BGR)
            1: (240, 160, 40),    # Sky Blue (BGR)
            2: (30, 220, 240),    # Yellow (BGR)
        }

    def extract_dataset_features(
        self,
        video_path_or_frames: Union[str, Path, List[np.ndarray], Dict[int, np.ndarray]],
        tracking_df: pd.DataFrame,
        sample_stride: int = 3,
        max_samples_per_track: int = 50,
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        """
        Samples bounding boxes from video frames and extracts jersey features.
        Uses fast sequential video streaming to avoid expensive frame seeks.
        """
        features_list = []
        track_ids = []
        frame_indices = []

        is_video_file = isinstance(video_path_or_frames, (str, Path))
        grouped = tracking_df.groupby("frame")
        active_frames = set(grouped.groups.keys())
        target_frames = {f for f in active_frames if f % sample_stride == 0}

        track_counts = Counter()

        if is_video_file:
            cap = cv2.VideoCapture(str(video_path_or_frames))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path_or_frames}")

            frame_idx = 0
            max_target = max(target_frames) if target_frames else 0

            while cap.isOpened() and frame_idx <= max_target:
                ret, frame_img = cap.read()
                if not ret or frame_img is None:
                    break
                frame_idx += 1

                if frame_idx in target_frames:
                    frame_data = grouped.get_group(frame_idx)
                    for _, row in frame_data.iterrows():
                        tid = int(row["track_id"])
                        if track_counts[tid] >= max_samples_per_track:
                            continue

                        bbox = [row["x1"], row["y1"], row["x2"], row["y2"]]
                        feat = self.extractor.extract_features(frame_img, bbox)
                        if feat is not None:
                            features_list.append(feat)
                            track_ids.append(tid)
                            frame_indices.append(frame_idx)
                            track_counts[tid] += 1

            cap.release()
        elif isinstance(video_path_or_frames, dict):
            for frame_idx in sorted(target_frames):
                frame_img = video_path_or_frames.get(frame_idx)
                if frame_img is None:
                    continue
                frame_data = grouped.get_group(frame_idx)
                for _, row in frame_data.iterrows():
                    tid = int(row["track_id"])
                    if track_counts[tid] >= max_samples_per_track:
                        continue
                    bbox = [row["x1"], row["y1"], row["x2"], row["y2"]]
                    feat = self.extractor.extract_features(frame_img, bbox)
                    if feat is not None:
                        features_list.append(feat)
                        track_ids.append(tid)
                        frame_indices.append(frame_idx)
                        track_counts[tid] += 1
        else:
            for frame_idx in sorted(target_frames):
                if frame_idx - 1 < len(video_path_or_frames):
                    frame_img = video_path_or_frames[frame_idx - 1]
                    frame_data = grouped.get_group(frame_idx)
                    for _, row in frame_data.iterrows():
                        tid = int(row["track_id"])
                        if track_counts[tid] >= max_samples_per_track:
                            continue
                        bbox = [row["x1"], row["y1"], row["x2"], row["y2"]]
                        feat = self.extractor.extract_features(frame_img, bbox)
                        if feat is not None:
                            features_list.append(feat)
                            track_ids.append(tid)
                            frame_indices.append(frame_idx)
                            track_counts[tid] += 1

        if len(features_list) == 0:
            raise ValueError("No valid player jersey features could be extracted.")

        return np.array(features_list), track_ids, frame_indices

    def fit(
        self,
        features: np.ndarray,
        track_ids: List[int],
    ):
        """
        Fits K-Means clustering model to feature vectors and computes
        temporal majority vote per track_id.
        """
        weights = np.array([2.5, 2.0, 2.0, 1.5, 1.0, 1.0], dtype=np.float32)
        scaled_features = features * weights

        self.cluster_model = KMeans(
            n_clusters=self.k,
            n_init=15,
            random_state=self.random_state
        )
        labels = self.cluster_model.fit_predict(scaled_features)

        track_votes: Dict[int, List[int]] = {}
        for tid, lbl in zip(track_ids, labels):
            track_votes.setdefault(tid, []).append(lbl)

        raw_track_teams = {}
        for tid, votes in track_votes.items():
            most_common = Counter(votes).most_common(1)[0][0]
            raw_track_teams[tid] = int(most_common)

        cluster_counts = Counter(raw_track_teams.values())
        sorted_clusters = [c for c, _ in cluster_counts.most_common()]

        team_mapping = {}
        if len(sorted_clusters) >= 2:
            team_mapping[sorted_clusters[0]] = 0
            team_mapping[sorted_clusters[1]] = 1
            for extra_idx, c in enumerate(sorted_clusters[2:], start=2):
                team_mapping[c] = extra_idx
        else:
            team_mapping = {c: c for c in sorted_clusters}

        self.track_teams = {tid: team_mapping[orig_lbl] for tid, orig_lbl in raw_track_teams.items()}

    def predict_track(self, track_id: int) -> int:
        """Returns the classified team ID (0: Team A, 1: Team B, 2: Referee/GK)."""
        return self.track_teams.get(track_id, 0)

    def annotate_dataframe(self, tracking_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds 'team_id' and 'team_name' columns to tracking DataFrame.
        """
        df = tracking_df.copy()
        df["team_id"] = df["track_id"].map(lambda tid: self.predict_track(int(tid)))
        df["team_name"] = df["team_id"].map(lambda cid: self.team_names.get(cid, f"Team {cid}"))
        return df

    def evaluate_against_ground_truth(
        self,
        ground_truth_teams: Dict[int, str]
    ) -> Dict[str, float]:
        """
        Evaluates clustering accuracy against ground truth annotations (e.g. from gameinfo.ini).
        Finds optimal bipartite alignment between predicted clusters and GT classes.
        """
        from itertools import permutations

        common_tids = [tid for tid in ground_truth_teams if tid in self.track_teams]
        if not common_tids:
            return {"accuracy": 0.0, "evaluated_tracks": 0}

        y_true = [ground_truth_teams[tid] for tid in common_tids]
        y_pred = [self.track_teams[tid] for tid in common_tids]

        unique_gt = list(set(y_true))
        gt_to_idx = {name: i for i, name in enumerate(unique_gt)}
        y_true_idx = [gt_to_idx[name] for name in y_true]

        n_classes = max(len(unique_gt), max(y_pred) + 1)
        best_acc = 0.0

        for perm in permutations(range(n_classes)):
            mapped_pred = [perm[p] if p < len(perm) else p for p in y_pred]
            acc = sum(1 for p, t in zip(mapped_pred, y_true_idx) if p == t) / len(y_true_idx)
            if acc > best_acc:
                best_acc = acc

        return {
            "accuracy": round(float(best_acc) * 100.0, 2),
            "evaluated_tracks": len(common_tids),
            "gt_distribution": dict(Counter(y_true)),
        }
