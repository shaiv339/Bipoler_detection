# Early Bipolar Disorder Detection

Real-time facial expression analysis system for early-stage bipolar disorder detection using computer vision and classical machine learning.

**Accuracy: 78% — outperforms ResNet-18 baseline by 11%**

---

## Overview

This project detects early indicators of bipolar disorder by analyzing facial expressions in real-time. It uses a lightweight SVM + PCA pipeline that runs efficiently on CPU, making it suitable for resource-constrained clinical environments.

The system classifies facial expressions into three mood states:
- `control` — healthy baseline
- `bipolar_manic` — manic episode indicators
- `bipolar_depressive` — depressive episode indicators

---

## Architecture

```
Raw Frame
    │
    ▼
Face Detection (Haar Cascade)
    │
    ▼
Feature Extraction
  ├── HOG Features       (shape/edge gradients)
  ├── LBP Features       (texture patterns)
  └── Geometric Features (landmark distances)
    │
    ▼
StandardScaler
    │
    ▼
PCA (n_components=50)   ← dimensionality reduction
    │
    ▼
SVM Classifier (RBF kernel)
    │
    ▼
Mood State Prediction + Confidence
```

---

## Results

| Model | Accuracy |
|---|---|
| ResNet-18 (baseline) | 67% |
| **SVM + PCA (ours)** | **78%** |

Key advantages of SVM + PCA over deep learning baseline:
- No GPU required — runs on CPU
- Interpretable features (HOG, LBP, landmarks)
- Faster inference for real-time use
- Less data-hungry than deep networks

---

## Setup

```bash
git clone https://github.com/shaivpatel/bipolar-disorder-detection
cd bipolar-disorder-detection
pip install -r requirements.txt
```

---

## Dataset Structure

Organize your dataset as follows:

```
data/
├── train/
│   ├── control/
│   ├── bipolar_manic/
│   └── bipolar_depressive/
└── test/
    ├── control/
    ├── bipolar_manic/
    └── bipolar_depressive/
```

Each folder contains `.jpg` or `.png` facial images. If no dataset is provided, the pipeline runs on synthetic data for demonstration.

---

## Usage

**Train:**
```bash
python main.py --mode train --data_path data/ --n_components 50 --kernel rbf --C 1.0
```

**Evaluate:**
```bash
python main.py --mode evaluate --model_path models/svm_model.pkl --data_path data/
```

**Real-time webcam prediction:**
```bash
python main.py --mode predict --model_path models/svm_model.pkl
```

---

## Project Structure

```
bipolar-disorder-detection/
├── main.py                  # Entry point
├── requirements.txt
├── src/
│   ├── features.py          # HOG, LBP, geometric feature extraction
│   ├── data_loader.py       # Dataset loading and preprocessing
│   ├── train.py             # PCA + SVM training pipeline
│   ├── evaluate.py          # Model evaluation and baseline comparison
│   └── predict.py           # Real-time webcam inference
├── models/                  # Saved model files
├── data/                    # Dataset (not included)
└── results/                 # Confusion matrices, plots
```

---

## Technical Details

**Feature Extraction:**
- HOG (Histogram of Oriented Gradients) captures edge and shape information
- LBP (Local Binary Patterns) captures micro-texture of facial skin
- Geometric features encode spatial relationships between facial landmarks

**Dimensionality Reduction:**
- PCA reduces feature space from ~1,500 dims to 50 components
- Retains ~95% of explained variance while eliminating noise

**Classifier:**
- SVM with RBF kernel and class-balanced weights
- Handles class imbalance common in clinical datasets

---

## Ethical Note

This tool is intended as a research prototype and clinical decision support aid — not a standalone diagnostic tool. Any clinical application should involve qualified mental health professionals.

---

## Author

**Shaiv Patel**
M.S.E. Computer Engineering, Johns Hopkins University
[LinkedIn](https://linkedin.com/in/shaivpatel) | spate235@jh.edu
