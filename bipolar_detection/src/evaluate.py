"""
Evaluation
==========
Evaluates trained SVM model and compares against ResNet-18 baseline.
"""

import pickle
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                              roc_auc_score, confusion_matrix)
from src.data_loader import load_dataset


def evaluate_model(args):
    """Load saved model and evaluate on test set."""
    with open(args.model_path, "rb") as f:
        saved = pickle.load(f)

    pipeline = saved["pipeline"]
    class_names = saved["class_names"]

    print("Loading test data...")
    X, y, _ = load_dataset(args.data_path)

    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)

    acc = accuracy_score(y, y_pred)
    print(f"\nAccuracy: {acc:.4f} ({acc*100:.1f}%)")

    # ResNet-18 baseline comparison (from paper)
    resnet_baseline = 0.67
    improvement = (acc - resnet_baseline) / resnet_baseline * 100
    print(f"ResNet-18 Baseline: {resnet_baseline*100:.1f}%")
    print(f"Improvement over baseline: +{improvement:.1f}%")

    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=class_names))

    try:
        auc = roc_auc_score(y, y_prob, multi_class="ovr", average="macro")
        print(f"Macro AUC-ROC: {auc:.4f}")
    except Exception:
        pass

    return acc
