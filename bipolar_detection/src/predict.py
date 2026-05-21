"""
Real-time Prediction
====================
Runs live webcam-based facial expression analysis for mood state prediction.
"""

import cv2
import pickle
import numpy as np
from src.features import FacialFeatureExtractor

MOOD_COLORS = {
    "control":             (0, 200, 0),
    "bipolar_manic":       (0, 165, 255),
    "bipolar_depressive":  (0, 0, 220),
}


def predict_realtime(args):
    """Run real-time prediction from webcam feed."""
    with open(args.model_path, "rb") as f:
        saved = pickle.load(f)

    pipeline   = saved["pipeline"]
    class_names = saved["class_names"]
    extractor  = FacialFeatureExtractor()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    print("Running real-time prediction. Press 'q' to quit.")
    frame_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_buffer.append(frame.copy())
        if len(frame_buffer) > 30:
            frame_buffer.pop(0)

        features = extractor.extract_all_features(frame)
        label, confidence = "No face detected", 0.0

        if features is not None:
            proba = pipeline.predict_proba([features])[0]
            pred_idx = np.argmax(proba)
            label = class_names[pred_idx]
            confidence = proba[pred_idx]

        face_bbox = extractor.detect_face(frame)
        if face_bbox is not None:
            x, y, w, h = face_bbox
            color = MOOD_COLORS.get(label, (200, 200, 200))
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        color = MOOD_COLORS.get(label, (200, 200, 200))
        cv2.putText(frame, f"State: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Bipolar Disorder Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
