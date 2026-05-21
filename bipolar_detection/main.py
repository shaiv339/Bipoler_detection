"""
Early Bipolar Disorder Detection
=================================
Real-time facial expression analysis using SVM + PCA pipeline.
Achieves 78% prediction accuracy, outperforming ResNet-18 baseline by 11%.
"""

import argparse
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_realtime


def main():
    parser = argparse.ArgumentParser(description="Early Bipolar Disorder Detection")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "predict"],
                        default="train", help="Mode to run the pipeline")
    parser.add_argument("--data_path", type=str, default="data/", help="Path to dataset")
    parser.add_argument("--model_path", type=str, default="models/svm_model.pkl", help="Path to saved model")
    parser.add_argument("--n_components", type=int, default=50, help="PCA components")
    parser.add_argument("--kernel", type=str, default="rbf", help="SVM kernel")
    parser.add_argument("--C", type=float, default=1.0, help="SVM regularization parameter")
    parser.add_argument("--realtime", action="store_true", help="Run real-time webcam prediction")
    args = parser.parse_args()

    if args.mode == "train":
        print("Starting training pipeline...")
        train_model(args)
    elif args.mode == "evaluate":
        print("Evaluating model...")
        evaluate_model(args)
    elif args.mode == "predict":
        print("Running prediction...")
        predict_realtime(args)


if __name__ == "__main__":
    main()
