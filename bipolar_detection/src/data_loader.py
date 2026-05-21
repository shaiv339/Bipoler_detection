"""
Data Loader
===========
Loads and preprocesses facial image datasets for bipolar disorder detection.
Supports standard image folder structure and CSV-based metadata.
"""

import os
import numpy as np
import cv2
from typing import Tuple, List
from src.features import FacialFeatureExtractor


# Expected dataset structure:
# data/
#   train/
#     control/        <- healthy subjects
#     bipolar_manic/  <- manic episode
#     bipolar_depressive/ <- depressive episode
#   test/
#     ...


LABEL_MAP = {
    "control": 0,
    "bipolar_manic": 1,
    "bipolar_depressive": 2
}


def load_dataset(data_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load dataset from folder structure and extract features.
    Returns X (features), y (labels), class_names.
    """
    extractor = FacialFeatureExtractor()
    X, y = [], []
    class_names = list(LABEL_MAP.keys())

    for split in ["train", "test"]:
        split_path = os.path.join(data_path, split)
        if not os.path.exists(split_path):
            continue

        for class_name, label in LABEL_MAP.items():
            class_path = os.path.join(split_path, class_name)
            if not os.path.exists(class_path):
                continue

            image_files = [f for f in os.listdir(class_path)
                           if f.lower().endswith((".jpg", ".jpeg", ".png"))]

            print(f"  Loading {len(image_files)} images from {class_name}...")

            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                frame = cv2.imread(img_path)
                if frame is None:
                    continue

                features = extractor.extract_all_features(frame)
                if features is not None:
                    X.append(features)
                    y.append(label)

    if len(X) == 0:
        print("WARNING: No images found. Generating synthetic data for demonstration.")
        return _generate_synthetic_data()

    return np.array(X), np.array(y), class_names


def _generate_synthetic_data(n_samples: int = 300, n_features: int = 1000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate synthetic data for testing the pipeline without a real dataset.
    Simulates facial feature distributions across 3 mood states.
    """
    np.random.seed(42)
    class_names = list(LABEL_MAP.keys())
    n_per_class = n_samples // len(class_names)

    X_parts, y_parts = [], []
    for i, class_name in enumerate(class_names):
        # Each class has a slightly different feature distribution
        mean = np.random.randn(n_features) * (i + 1) * 0.3
        X_class = np.random.randn(n_per_class, n_features) + mean
        y_class = np.full(n_per_class, i)
        X_parts.append(X_class)
        y_parts.append(y_class)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    # Shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx], class_names
