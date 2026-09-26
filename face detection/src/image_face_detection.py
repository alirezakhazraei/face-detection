"""
Face detection on a static image file (bonus script - reuses the same
FaceDetector class as the webcam version).

Usage:
    python src/image_face_detection.py path/to/photo.jpg
"""

import os
import sys

import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.face_detector import FaceDetector

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
PROTOTXT_PATH = os.path.join(MODELS_DIR, "deploy.prototxt")
CAFFEMODEL_PATH = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

CONFIDENCE_THRESHOLD = 0.5


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python src/image_face_detection.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: file not found: {image_path}")
        sys.exit(1)

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: could not read image (unsupported format?): {image_path}")
        sys.exit(1)

    detector = FaceDetector(PROTOTXT_PATH, CAFFEMODEL_PATH, CONFIDENCE_THRESHOLD)
    faces = detector.detect(frame)

    print(f"Found {len(faces)} face(s).")

    for (x1, y1, x2, y2, confidence) in faces:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{confidence * 100:.1f}%"
        cv2.putText(
            frame, label, (x1, max(y1 - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
        )

    output_path = os.path.splitext(image_path)[0] + "_detected.jpg"
    cv2.imwrite(output_path, frame)
    print(f"Saved result to {output_path}")


if __name__ == "__main__":
    main()
