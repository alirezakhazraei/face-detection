"""
FaceDetector: a small wrapper around OpenCV's DNN face detection model.

The model is an SSD (Single Shot Detector) with a ResNet-10 backbone,
trained by the OpenCV team. It expects a 300x300 input and outputs a
list of detections, each with a confidence score and a bounding box.
"""

import os
from typing import List, Tuple

import cv2
import numpy as np

# Model input size the network was trained on - don't change this,
# it must match the .prototxt architecture.
INPUT_SIZE = (300, 300)

# Mean subtraction values used during training (BGR order).
# These must match what the model expects, otherwise accuracy drops.
MEAN_VALUES = (104.0, 177.0, 123.0)

Detection = Tuple[int, int, int, int, float]  # (x1, y1, x2, y2, confidence)


class FaceDetector:
    def __init__(self, prototxt_path: str, model_path: str, confidence_threshold: float = 0.5):
        if not os.path.exists(prototxt_path):
            raise FileNotFoundError(
                f"Model config not found at {prototxt_path}. "
                "Run 'python download_model.py' first."
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model weights not found at {model_path}. "
                "Run 'python download_model.py' first."
            )

        self.confidence_threshold = confidence_threshold
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Runs face detection on a single BGR frame (as read by cv2.VideoCapture
        or cv2.imread) and returns a list of (x1, y1, x2, y2, confidence)
        bounding boxes for every face above the confidence threshold.
        """
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, INPUT_SIZE),
            scalefactor=1.0,
            size=INPUT_SIZE,
            mean=MEAN_VALUES,
            swapRB=False,
            crop=False,
        )

        self.net.setInput(blob)
        raw_detections = self.net.forward()

        faces: List[Detection] = []
        # raw_detections shape: [1, 1, N, 7] - the last dim is
        # [batch_id, class_id, confidence, x1, y1, x2, y2] (x/y normalized 0-1)
        for i in range(raw_detections.shape[2]):
            confidence = float(raw_detections[0, 0, i, 2])
            if confidence < self.confidence_threshold:
                continue

            box = raw_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            # Clip to frame bounds - detections near edges can go slightly out of range
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            faces.append((x1, y1, x2, y2, confidence))

        return faces
