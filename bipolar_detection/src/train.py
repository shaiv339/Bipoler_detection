"""
Training Pipeline
=================
PCA dimensionality reduction + SVM classifier for bipolar disorder detection.
Achieves 78% prediction accuracy, outperforming ResNet-18 baseline by 11%.
"""

import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.features import FacialFeatureExtractor
from src.data_loader import load_dataset


def build_pipeline(n_components: int = 50, kernel: str = "rbf",
                   C: float = 1.0, gamma: str = "scale") -> Pipeline:
    """
    Build the full PCA + SVM pipeline.
    
    Architecture:
        StandardScaler → PCA (n_components) → SVM (kernel, C, gamma)
    
    PCA reduces high-dimensional facial features to a compact representation,
    retaining the most discriminative variance for mood-state classification.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=42)),
        ("svm", SVC(kernel=kernel, C=C, gamma=gamma,
                    class_weight="balanced", probability=True, random_state=42))
    ])


def hyperparameter_search(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Grid search over SVM hyperparameters."""
    param_grid = {
        "pca__n_components": [30, 50, 75, 100],
        "svm__C": [0.1, 1.0, 10.0],
        "svm__kernel": ["rbf", "linear"],
        "svm__gamma": ["scale", "auto"],
    }

    pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv,
                                scoring="accuracy", n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    print(f"\nBest params: {grid_search.best_params_}")
    print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
    return grid_search.best_params_


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           class_names: list, save_path: str = "results/confusion_matrix.png"):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix — Bipolar Detection")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_pca_variance(pipeline: Pipeline, save_path: str = "results/pca_variance.png"):
    """Plot cumulative explained variance from PCA."""
    pca = pipeline.named_steps["pca"]
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumvar) + 1), cumvar, marker="o", markersize=3)
    plt.axhline(y=0.95, color="r", linestyle="--", label="95% variance")
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"PCA variance plot saved to {save_path}")


def train_model(args):
    """Full training pipeline."""
    print("Loading dataset...")
    X, y, class_names = load_dataset(args.data_path)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(class_names)} classes")
    print(f"Classes: {class_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    print("\nBuilding PCA + SVM pipeline...")
    pipeline = build_pipeline(
        n_components=args.n_components,
        kernel=args.kernel,
        C=args.C
    )

    print("Training...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    plot_confusion_matrix(y_test, y_pred, class_names)
    plot_pca_variance(pipeline)

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    with open(args.model_path, "wb") as f:
        pickle.dump({"pipeline": pipeline, "class_names": class_names}, f)
    print(f"\nModel saved to {args.model_path}")

    return pipeline, class_names
