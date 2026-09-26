"""
Real-time face detection from the webcam using OpenCV's DNN face detector.

Usage:
    python src/webcam_face_detection.py

Controls:
    q  -  quit
    s  -  save a screenshot of the current frame to screenshots/
"""

import os
import sys
import time

import cv2

# Allow running this file directly (python src/webcam_face_detection.py)
# by adding the project root to the path so `from src...` imports work.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.face_detector import FaceDetector

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
PROTOTXT_PATH = os.path.join(MODELS_DIR, "deploy.prototxt")
CAFFEMODEL_PATH = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

SCREENSHOTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "screenshots"
)

CONFIDENCE_THRESHOLD = 0.5
BOX_COLOR = (0, 255, 0)  # BGR - green
TEXT_COLOR = (0, 255, 0)


def draw_detections(frame, faces) -> None:
    for (x1, y1, x2, y2, confidence) in faces:
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"{confidence * 100:.1f}%"
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        cv2.putText(
            frame, label, (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 2,
        )


def draw_fps(frame, fps: float) -> None:
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
    )


def draw_face_count(frame, count: int) -> None:
    h = frame.shape[0]
    cv2.putText(
        frame, f"Faces: {count}", (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
    )


def main() -> None:
    detector = FaceDetector(PROTOTXT_PATH, CAFFEMODEL_PATH, CONFIDENCE_THRESHOLD)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam (index 0). Is it connected and free?")
        sys.exit(1)

    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

    prev_time = time.time()
    print("Press 'q' to quit, 's' to save a screenshot.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)  # mirror view, feels more natural

            faces = detector.detect(frame)
            draw_detections(frame, faces)

            current_time = time.time()
            fps = 1.0 / (current_time - prev_time) if current_time != prev_time else 0.0
            prev_time = current_time
            draw_fps(frame, fps)
            draw_face_count(frame, len(faces))

            cv2.imshow("Face Detection - press 'q' to quit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                filename = os.path.join(
                    SCREENSHOTS_DIR, f"screenshot_{int(time.time())}.png"
                )
                cv2.imwrite(filename, frame)
                print(f"Saved screenshot: {filename}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
