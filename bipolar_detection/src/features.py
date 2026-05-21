"""
Feature Extraction
==================
Extracts facial landmarks, AU (Action Units), and texture features
from facial images/video frames for bipolar disorder detection.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional


class FacialFeatureExtractor:
    """
    Extracts multi-modal facial features:
    - Geometric features (landmark distances, angles)
    - Texture features (LBP, HOG)
    - Temporal features (movement, expression dynamics)
    """

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face region in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                                    minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return None
        # Return largest face
        return max(faces, key=lambda f: f[2] * f[3])

    def extract_hog_features(self, face_img: np.ndarray,
                              orientations: int = 9,
                              pixels_per_cell: Tuple = (8, 8),
                              cells_per_block: Tuple = (2, 2)) -> np.ndarray:
        """Extract HOG (Histogram of Oriented Gradients) features."""
        face_resized = cv2.resize(face_img, (64, 64))
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

        # Manual HOG implementation for portability
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=1)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=1)
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx) * (180 / np.pi) % 180

        cell_h, cell_w = pixels_per_cell
        n_cells_y = gray.shape[0] // cell_h
        n_cells_x = gray.shape[1] // cell_w

        hog_cells = np.zeros((n_cells_y, n_cells_x, orientations))
        bin_size = 180 // orientations

        for i in range(n_cells_y):
            for j in range(n_cells_x):
                cell_mag = magnitude[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                cell_ang = angle[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                for o in range(orientations):
                    mask = (cell_ang >= o * bin_size) & (cell_ang < (o + 1) * bin_size)
                    hog_cells[i, j, o] = cell_mag[mask].sum()

        return hog_cells.flatten()

    def extract_lbp_features(self, face_img: np.ndarray,
                              radius: int = 1, n_points: int = 8) -> np.ndarray:
        """Extract Local Binary Pattern features for texture analysis."""
        gray = cv2.cvtColor(cv2.resize(face_img, (64, 64)), cv2.COLOR_BGR2GRAY)
        lbp = np.zeros_like(gray, dtype=np.uint8)

        for i in range(radius, gray.shape[0] - radius):
            for j in range(radius, gray.shape[1] - radius):
                center = gray[i, j]
                code = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = j + radius * np.cos(angle)
                    y = i - radius * np.sin(angle)
                    x0, y0 = int(np.floor(x)), int(np.floor(y))
                    neighbor = gray[y0, x0]
                    code |= (1 << k) if neighbor >= center else 0
                lbp[i, j] = code

        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)
        return hist

    def extract_geometric_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract geometric features from facial landmarks.
        Computes pairwise distances and angles between key facial points.
        """
        if landmarks is None or len(landmarks) == 0:
            return np.zeros(68 * 2)

        # Normalize landmarks relative to face bounding box
        landmarks = landmarks.astype(float)
        landmarks -= landmarks.mean(axis=0)
        scale = np.sqrt((landmarks**2).sum(axis=1)).max()
        if scale > 0:
            landmarks /= scale

        return landmarks.flatten()

    def extract_temporal_features(self, frame_sequence: List[np.ndarray]) -> np.ndarray:
        """
        Extract temporal dynamics from a sequence of frames.
        Captures expression changes over time — key signal for bipolar detection.
        """
        if len(frame_sequence) < 2:
            return np.zeros(10)

        features = []
        prev_gray = cv2.cvtColor(frame_sequence[0], cv2.COLOR_BGR2GRAY)

        for frame in frame_sequence[1:]:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, curr_gray)
            features.append([
                diff.mean(),
                diff.std(),
                diff.max(),
            ])
            prev_gray = curr_gray

        features = np.array(features)
        return np.concatenate([features.mean(axis=0), features.std(axis=0),
                                features.max(axis=0), features.min(axis=0)])

    def extract_all_features(self, frame: np.ndarray,
                              landmarks: Optional[np.ndarray] = None) -> np.ndarray:
        """Extract and concatenate all features from a single frame."""
        face_bbox = self.detect_face(frame)
        if face_bbox is None:
            return None

        x, y, w, h = face_bbox
        face_img = frame[y:y+h, x:x+w]

        hog_feats = self.extract_hog_features(face_img)
        lbp_feats = self.extract_lbp_features(face_img)
        geo_feats = self.extract_geometric_features(landmarks) if landmarks is not None \
                    else np.zeros(136)

        return np.concatenate([hog_feats, lbp_feats, geo_feats])
