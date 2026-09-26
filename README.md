# Real-Time Face Detection

Real-time face detection from a webcam using OpenCV's DNN module, built as
an internship-level computer vision project in Python.

## How it works

Uses OpenCV's pretrained deep learning face detector: an **SSD (Single Shot
Detector) with a ResNet-10 backbone**, trained by the OpenCV team
specifically for face detection. It's a Caffe model, loaded via
`cv2.dnn.readNetFromCaffe`.

This is more accurate than the classic Haar Cascade approach (handles
angled faces, partial occlusion, and varied lighting much better), while
still being simple enough to explain in an interview: "I load a pretrained
Caffe model, feed each webcam frame through it as a 300x300 blob, and draw
boxes around detections above a confidence threshold."

## Project structure

```
face-detection/
  download_model.py              Downloads the pretrained model files (run once)
  requirements.txt
  models/                        deploy.prototxt + .caffemodel go here (gitignored - see below)
  src/
    face_detector.py             FaceDetector class - loads the model, runs detection on a frame
    webcam_face_detection.py     Main script - real-time detection from your webcam
    image_face_detection.py      Bonus script - runs detection on a single image file
  screenshots/                   Saved screenshots land here (created automatically)
```

`face_detector.py` is deliberately separate from the webcam script: the
`FaceDetector` class only knows how to detect faces in a single frame (numpy
array in, list of boxes out). It doesn't know or care whether that frame
came from a webcam, a video file, or a photo - which is why the same class
powers both `webcam_face_detection.py` and `image_face_detection.py`.

## Getting Started

### 1. Prerequisites
- Python 3.9+
- A webcam

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the model files
```bash
python download_model.py
```
This fetches two files from OpenCV's official repositories into `models/`:
- `deploy.prototxt` - describes the network architecture (layers)
- `res10_300x300_ssd_iter_140000.caffemodel` - the pretrained weights (~10MB)

These are **not committed to git** (see `.gitignore`) since they're binary
model weights, not source code - `download_model.py` is what your graders
or teammates run to fetch them.

### 4. Run it
```bash
python src/webcam_face_detection.py
```

**Controls:**
- `q` - quit
- `s` - save a screenshot of the current frame to `screenshots/`

The window shows a green box around each detected face with its confidence
score, plus a live FPS counter and face count.

### Bonus: detect faces in a photo instead
```bash
python src/image_face_detection.py path/to/photo.jpg
```
Saves the result as `photo_detected.jpg` next to the original.

## Tuning

`CONFIDENCE_THRESHOLD` in `webcam_face_detection.py` (default `0.5`) controls
how confident the model must be before drawing a box. Lower it to catch more
faces (at the risk of false positives), raise it to be stricter.

## Pushing this to GitHub

```bash
git init
git add .
git commit -m "Initial commit: real-time face detection with OpenCV DNN"
git branch -M main
git remote add origin <your-empty-github-repo-url>
git push -u origin main
```

Since the model files are gitignored, anyone cloning your repo just needs to
run `python download_model.py` before using it - keeps the repo small and
fast to clone, which is exactly what you want on GitHub.

Suggested follow-up commits:
```bash
git commit -m "Add face-count logging to a CSV file"
git commit -m "Add support for detecting faces in a video file"
git commit -m "Add simple face-blurring mode for privacy demo"
```
